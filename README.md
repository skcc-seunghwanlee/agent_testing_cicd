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

### 디렉터리 트리

```
.
├── .github/workflows/
│   └── agent-tests.yml          # CI: PR(L1·L2 필수, L3 smoke 비차단), nightly(L3 전체)
├── data/
│   └── golden.json              # 골든셋 24건 (smoke 8건)
├── prompts/
│   ├── system.txt               # 에이전트 시스템 프롬프트
│   └── refusal.txt              # guard 차단 시 거절 문구
├── src/infoagent/               # 테스트 대상: 샘플 에이전트
│   ├── __init__.py              # build_graph 노출
│   ├── graph.py                 # 그래프 조립
│   ├── nodes.py                 # 노드와 라우팅 함수
│   ├── tools.py                 # 에이전트가 쓰는 도구
│   ├── backends.py              # mock 검색 백엔드
│   └── prompts.py               # prompts/*.txt 로더
├── evalkit/                     # 테스트·평가 공통 코드
│   ├── runner.py                # 그래프 실행 후 결과 추출
│   ├── scoring.py               # 골든 케이스 채점
│   ├── stubs.py                 # 키 없이 쓰는 가짜 judge (NoLLM)
│   └── live.py                  # 실제 LLM·judge 연결
├── tests/
│   ├── conftest.py              # DeepEval 텔레메트리 끄기
│   ├── helpers.py               # 스크립트된 가짜 모델 생성
│   ├── unit/                    # L1
│   │   ├── test_l1_backend_tools_nodes.py
│   │   └── test_l1_golden_consistency.py
│   ├── flows/                   # L2
│   │   └── test_l2_flows.py
│   └── l3/                      # L3 (LLM_API_KEY 필요)
│       └── test_l3_live.py
├── scripts/
│   ├── variance.py              # 반복 실행으로 점수 흔들림 측정
│   └── compare.py               # baseline 대비 회귀 판정
├── pytest.ini                   # 경로 설정, l1/l2/l3 마커
└── requirements.txt             # 버전 고정 의존성
```

### 파일별 역할

**에이전트 (`src/infoagent/`)**

| 파일 | 역할 |
|---|---|
| `graph.py` | `build_graph(model, backend, checkpointer)`로 그래프를 만든다. 흐름은 `guard → agent ⇄ tools → 종료`. 모델은 외부에서 주입받아서 테스트에서는 가짜 모델로 바꾼다. 상태는 `messages`, `blocked`, `steps`. |
| `nodes.py` | `guard`(정규식으로 위험 요청 차단, LLM 없음), `agent`(LLM 호출), `tools`(도구 실행, 오류는 예외 대신 메시지로 반환)와 라우팅 함수 `route_after_guard`, `route_after_agent`. 도구 호출 반복은 `MAX_STEPS`(4)에서 끊는다. |
| `tools.py` | `search_docs`(키워드 검색), `get_doc`(문서 전문). 허용 도구 목록 `ALLOWED_TOOLS`를 여기서 정의하고 `ToolPermission` 검사가 이를 쓴다. |
| `backends.py` | 문서 6건짜리 메모리 검색(`InMemoryBackend`). 실제 과제에서는 이 파일만 실제 검색 API로 바꾼다. |
| `prompts.py` | `prompts/*.txt`를 읽는 함수. |

**평가 공통 코드 (`evalkit/`)**

| 파일 | 역할 |
|---|---|
| `runner.py` | `run_agent`가 그래프를 한 번 돌려 `RunResult`(최종 답, 도구 호출 목록, 노드 경로, 단계 수)를 돌려준다. DeepEval `LLMTestCase`로 변환하는 함수도 여기 있다. |
| `scoring.py` | 골든 한 건을 채점한다. `deterministic_checks`는 도구 호출 정확성, 허용 목록 준수, 차단 요청 처리, 근거 doc_id 인용을 LLM 없이 검사한다. `judge_check`는 `GEval`로 답변 품질을 0~1 점수로 낸다. |
| `stubs.py` | `NoLLM`. `ToolCorrectnessMetric`은 생성 시 모델 객체가 필요해서 키가 없으면 에러가 나는데, 이 스텁을 넘기면 키 없이 쓸 수 있다. 실수로 호출되면 바로 에러를 낸다. |
| `live.py` | 환경변수(`LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL`, `JUDGE_MODEL`)로 OpenAI 호환 엔드포인트에 연결한다. `OpenAICompatJudge`는 DeepEval judge 어댑터. |

**테스트 (`tests/`)**

| 파일 | 단계 | 내용 |
|---|---|---|
| `unit/test_l1_backend_tools_nodes.py` | L1 | 검색 순위, 도구 이름과 허용 목록 일치, guard, 라우팅 함수, 루프 상한, 도구 오류 처리 |
| `unit/test_l1_golden_consistency.py` | L1 | 골든의 id 중복, 기대 문서의 코퍼스 존재, 기대 도구의 허용 목록 포함, `blocked` 값과 guard 일치 확인 |
| `flows/test_l2_flows.py` | L2 | 가짜 모델로 정상 경로, 차단 시 모델·도구 미호출, 검색 후 문서 조회, 루프 상한 확인. DeepEval `ToolCorrectness`와 `ToolPermission` 검사 |
| `l3/test_l3_live.py` | L3 | 실제 LLM으로 골든을 돌려 결정론 체크와 judge 점수를 `results/l3_latest.jsonl`에 기록한다. 키가 없으면 스킵, `SMOKE=1`이면 smoke 8건만 |
| `helpers.py` | - | `scripted(...)`로 정해진 응답을 순서대로 내는 가짜 모델을 만들고, `call(...)`로 도구 호출 메시지를 만든다 |
| `conftest.py` | - | DeepEval 텔레메트리를 끈다 |

**스크립트, 데이터, 설정**

| 파일 | 역할 |
|---|---|
| `scripts/variance.py` | 같은 골든을 N회 돌려 케이스별 통과 횟수와 judge 점수의 평균·표준편차를 출력한다. 회귀 마진을 정할 때 쓴다. `--fake`는 LLM 없이 배선만 확인한다. |
| `scripts/compare.py` | `results/l3_latest.jsonl`을 `baseline/l3.jsonl`과 비교한다. baseline에서 통과하던 결정론 체크가 실패하거나 judge 평균이 마진 이상 떨어지면 종료 코드 1을 낸다. baseline이 없으면 통과 처리한다. |
| `data/golden.json` | 골든 24건. 필드는 `id`, `question`, `expected_tools`, `expected_doc`, `blocked`, `tags`, `smoke`. 태그는 hr, it, fin, sec, no-answer, safety, multi. |
| `prompts/system.txt`, `prompts/refusal.txt` | 시스템 프롬프트와 거절 문구. 코드와 분리해서 변경 이력을 diff로 본다. 바꾸면 L3로 확인한다. |
| `.github/workflows/agent-tests.yml` | `l1-l2`(PR·push, 필수), `l3-smoke`(PR, 비차단, 포크 PR은 제외), `l3-full`(nightly·수동). |
| `pytest.ini` | `src`와 루트를 import 경로에 추가하고 `l1`, `l2`, `l3` 마커를 정의한다. |

## 알아둘 점

- `ToolCorrectnessMetric`은 judge를 호출하지 않지만 생성 시점에 모델 객체를 만들어서 키가 없으면 에러가 난다. `NoLLM` 스텁으로 우회했고, 실수로 LLM이 호출되면 즉시 에러가 나게 해뒀다.
- 가짜 모델은 `FakeMessagesListChatModel`을 쓴다 (`FakeListChatModel`은 문자열만 받아서 tool_calls를 못 넣는다).
