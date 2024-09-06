from typing import List, Tuple, cast
import pytest

from vllm.entrypoints.chat_utils import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageToolCallParam,
)


from .utils import CONFIGS, ToolCallParserConfig


# run this for each tool call regex config
@pytest.fixture(params=CONFIGS.keys())
def tool_call_messages(request):
    config = CONFIGS[request.param]

    messages = config["messages"]
    for message in messages:
        if message["role"] == "assistant":
            assistant = cast(ChatCompletionAssistantMessageParam, message)
            yield (config, list(assistant["tool_calls"]))


@pytest.fixture()
def tool_call_fixture(
    tool_call_messages: Tuple[
        ToolCallParserConfig, List[ChatCompletionMessageToolCallParam]
    ]
):
    config, tool_calls = tool_call_messages

    regex = config["regex"]
    format_tool_call = config["format_tool_call"]

    strings = []
    for tool_call in tool_calls:
        string = format_tool_call(tool_call)
        strings.append(string)

    yield ("".join(strings), regex, tool_calls)
