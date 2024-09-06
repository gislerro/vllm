import re

from vllm.entrypoints.chat_utils import (
    ChatCompletionMessageToolCallParam,
)


from typing import List, Tuple
from vllm.entrypoints.openai.tool_parsers.regex_tool_parser import parse_tool_calls


def test_parse_tool_calls(
    tool_call_fixture: Tuple[
        str, re.Pattern[str], List[ChatCompletionMessageToolCallParam]
    ]
):

    tool_call_str, tool_call_regex, tool_calls = tool_call_fixture

    parsed_tool_calls = parse_tool_calls(tool_call_str, tool_call_regex)

    assert len(parsed_tool_calls) == len(tool_calls)

    for i, parsed_tool_call in enumerate(parsed_tool_calls):
        assert parsed_tool_call.type == "function"
        assert parsed_tool_call.function.name == tool_calls[i]["function"]["name"]
        assert (
            parsed_tool_call.function.arguments
            == tool_calls[i]["function"]["arguments"]
        )
