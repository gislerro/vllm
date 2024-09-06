import pytest

from vllm.entrypoints.openai.tool_parsers.regex_tool_parser import (
    validate_tool_call_regex,
)


@pytest.mark.parametrize(
    "regex_str, expected",
    [
        (r"(?P<name>[^>]+)(?P<arguments>.*?)", True),
        (r"<function=(?P<name>[^>]+)>(?P<arguments>.*?)<\/function>", True),
        (r"<tool_call=(?P<name>[^>]+)>(?P<arguments>.*?)<\/tool_call>", True),
        (
            r"MY very WeIRd(?P<name>/>(?P<name>[^>]+)!>tool calling(?P<arguments>.*?)template<end_of_text>",
            True,
        ),
        (r"(?P<names>[^>]+)(?P<arguments>.*?)", False),
        (r"(?P<name>[^>]+)(?P<argument>.*?)", False),
        (r"<function=(?P<name>[^>]+)>(?P<parameters>.*?)<\/function>", False),
        (r"(?P<names>[^>]+)>(?P<args>.*?)<\/tool_call", False),
    ],
)
def test_validate_tool_call_regex(regex_str, expected):
    assert validate_tool_call_regex(regex_str) == expected
