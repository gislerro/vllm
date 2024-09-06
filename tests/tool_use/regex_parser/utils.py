import re
from typing import Callable, Dict, Sequence, TypedDict


from vllm.entrypoints.chat_utils import (
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCallParam,
)

from ..utils import MESSAGES_WITH_TOOL_RESPONSE, MESSAGES_WITH_PARALLEL_TOOL_RESPONSE

TEST_CASES = [MESSAGES_WITH_TOOL_RESPONSE, MESSAGES_WITH_PARALLEL_TOOL_RESPONSE]


class ToolCallParserConfig(TypedDict):
    regex: re.Pattern[str]
    format_tool_call: Callable[[ChatCompletionMessageToolCallParam], str]
    messages: Sequence[ChatCompletionMessageParam]


CONFIGS: Dict[str, ToolCallParserConfig] = {
    "Single Tool Call": {
        "regex": re.compile(
            r"<function=(?P<name>[^>]+)>(?P<arguments>.*?)<\/function>"
        ),
        "format_tool_call": lambda tool_call: f"<function={tool_call['function']['name']}>{tool_call['function']['arguments']}</function>",
        "messages": MESSAGES_WITH_TOOL_RESPONSE,
    },
    "Parallel Tool Call": {
        "regex": re.compile(
            r"<tool_call=(?P<name>[^>]+)>(?P<arguments>.*?)<\/tool_call>"
        ),
        "format_tool_call": lambda tool_call: f"<tool_call={tool_call['function']['name']}>{tool_call['function']['arguments']}</tool_call>",
        "messages": MESSAGES_WITH_PARALLEL_TOOL_RESPONSE,
    },
}
