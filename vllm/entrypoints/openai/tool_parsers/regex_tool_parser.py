import re
from typing import List, Sequence, Union

from vllm.entrypoints.openai.protocol import (
    DeltaMessage,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.entrypoints.openai.tool_parsers.abstract_tool_parser import ToolParser
from vllm.logger import init_logger
from vllm.transformers_utils.tokenizer import AnyTokenizer


logger = init_logger(__name__)


# matches any regex with two named groups: 'name' and 'arguments'
TOOL_CALL_META_REGEX = re.compile(
    r"(?=.*?\(\?P<name>.*?\))(?=.*?\(\?P<arguments>.*?\))"
)


def validate_tool_call_regex(tool_call_regex: str) -> bool:
    match = re.search(TOOL_CALL_META_REGEX, tool_call_regex)
    return match is not None


def parse_tool_calls(content: str, tool_call_regex: re.Pattern[str]) -> List[ToolCall]:
    tool_call_matches = tool_call_regex.finditer(content)
    tool_calls: List[ToolCall] = []
    for match in tool_call_matches:
        name = match.group("name")
        arguments = match.group("arguments")

        tool_calls.append(
            ToolCall(
                type="function",
                function=FunctionCall(name=name, arguments=arguments),
            )
        )

    return tool_calls


class RegexToolParser(ToolParser):

    tool_call_regex: re.Pattern[str]

    def __init__(self, tokenizer: AnyTokenizer, tool_call_regex: str):
        super().__init__(tokenizer)

        if not validate_tool_call_regex(tool_call_regex):
            raise RuntimeError(f"Invalid user regex: {tool_call_regex}")

        self.tool_call_regex = re.compile(tool_call_regex)

    def extract_tool_calls(self, model_output: str):

        tool_calls = parse_tool_calls(model_output, self.tool_call_regex)

        return ExtractedToolCallInformation(
            tools_called=len(tool_calls) > 0,
            tool_calls=tool_calls,
            content=model_output if len(model_output) > 0 else None,
        )

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> Union[DeltaMessage, None]:
        raise NotImplementedError("Streaming not supported for RegexToolParser")
