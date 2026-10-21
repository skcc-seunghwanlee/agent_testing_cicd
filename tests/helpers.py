from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage


def call(name: str, **args) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"call-{name}-{len(args)}"}])


def scripted(*msgs) -> FakeMessagesListChatModel:
    """Fake chat model returning the given AIMessages in order."""
    return FakeMessagesListChatModel(responses=list(msgs))
