# agent-testing-poc

LangGraph 에이전트를 PR 단위로 테스트하는 방식을 검증하기 위한 샘플.
대상은 사내 문서를 검색해 답하는 정보탐색 에이전트(검색 백엔드는 mock). 실제 과제 repo에 옮길 때 `backends.py`만 실제 검색 API로 바꾸면 된다.

## 테스트 3단계

| 단계 | 위치 | LLM | 실행 시점 | 잡는 것 |
|---|---|---|---|---|
| L1 | `tests/unit` | 없음 | 모든 커밋/PR | 도구, guard, 라우팅 함수, 골든셋 정합성 |
| L2 | `tests/flows` | 가짜 모델(스크립트된 응답) | 모든 PR | 그래프 경로, 루프 상한, DeepEval 결정적 메트릭(ToolCorrectness, ToolPermission) |
| L3 | `tests/l3` | 실제 LLM + judge | PR(smoke, 비차단) / nightly(전체) | 프롬프트·모델 변경에 따른 행동 변화 |

프롬프트를 바꿔도 L1/L2는 통과한다(가짜 모델이라 프롬프트를 안 읽음). 프롬프트 회귀는 L3에서만 잡힌다. 이건 한계가 아니라 역할 분담이다.

## 실행

```bash
pip install -r requirements.txt
python -m pytest tests -m "l1 or l2" -q          # LLM 불필요

export LLM_API_KEY=...            # OpenAI 호환 엔드포인트
export LLM_BASE_URL=...           # 사내 vLLM이면 필수
export LLM_MODEL=...
export JUDGE_MODEL=...            # 생략하면 LLM_MODEL과 동일
SMOKE=1 python -m pytest tests/l3 -m l3 -q       # smoke 8건
python -m pytest tests/l3 -m l3 -q               # 전체 24건
```

## 회귀 판정은 상대값으로

judge 점수는 실행마다 흔들린다. 절대 점수로 막으면 flaky해지므로 baseline 대비 하락으로 판단한다.

1. `python scripts/variance.py --runs 5 --smoke` 로 우리 모델+judge의 흔들림(sd)을 먼저 측정한다.
2. 마진은 측정한 sd의 2배 이상으로 잡는다 (`scripts/compare.py --margin`, 기본 0.10은 임시값).
3. 신뢰할 만한 실행 결과를 `baseline/l3.jsonl`로 복사해 커밋한다. 없으면 compare는 통과 처리한다.
4. 결정적 체크(도구 호출 정확성, 허용 목록, 인용)는 baseline에서 통과했는데 지금 실패하면 무조건 실패.

## 폐쇄망에서 쓸 때

- GitHub-hosted runner는 사내망에 접근 못한다. `.github/workflows/agent-tests.yml`의 L3 job `runs-on`을 self-hosted runner 라벨로 바꾼다. L1/L2는 외부 접근이 필요 없어 그대로 둬도 된다.
- DeepEval은 기본이 OpenAI judge다. `evalkit/live.py`의 `OpenAICompatJudge`가 사내 vLLM 같은 OpenAI 호환 엔드포인트를 judge로 쓴다.
- Confident AI(클라우드 대시보드)는 쓰지 않는다. baseline은 repo의 jsonl로 관리한다.
- 실제 에이전트가 k8s에서만 뜬다면 L3는 PR 체크가 아니라 배포 후 k8s Job으로 돌리는 쪽이 현실적이다. 이 샘플의 `tests/l3`를 그대로 Job 이미지에서 실행하면 된다.

## 구성

```
src/infoagent/   state/graph/nodes/tools/backends (샘플 에이전트)
prompts/         시스템 프롬프트 (코드와 분리, 변경 시 L3로 확인)
evalkit/         runner(실행 결과 추출), scoring(골든 케이스 채점), stubs(NoLLM), live(실제 LLM 연결)
data/golden.json 24건 (smoke 8건, 태그: hr/it/fin/sec/no-answer/safety/multi)
scripts/         variance.py, compare.py
```

## 알아둘 점

- `ToolCorrectnessMetric`은 judge를 호출하지 않지만 생성 시점에 모델 객체를 만들어서 키가 없으면 에러가 난다. `NoLLM` 스텁으로 우회했고, 실수로 LLM이 호출되면 즉시 에러가 나게 해뒀다.
- 가짜 모델은 `FakeMessagesListChatModel`을 쓴다 (`FakeListChatModel`은 문자열만 받아서 tool_calls를 못 넣는다).
