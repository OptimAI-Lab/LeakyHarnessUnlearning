import os
import re
from typing import Union

from langchain.agents.agent import AgentOutputParser
from langchain.schema import AgentAction, AgentFinish

FINAL_ANSWER_ACTION = "Final Answer:"


class CustomMRKLOutputParser(AgentOutputParser):
    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        if FINAL_ANSWER_ACTION in text:
            return AgentFinish(
                {
                    "output": text.split(FINAL_ANSWER_ACTION)[-1].strip(),
                    "Final Thought": text.rsplit(FINAL_ANSWER_ACTION, 1)[0].strip(),
                 }, text
            )
        # \s matches against tab/newline/whitespace
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, text, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse LLM output: `{text}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        return AgentAction(action, action_input.strip(" ").strip('"'), text)

class CustomMRKLOutputParser2(AgentOutputParser):
    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        final_answer_action = "ASSISTANT Response:"
        if final_answer_action in text:
            return AgentFinish(
                {
                    "output": text.split(final_answer_action)[-1].strip(),
                    "Final Thought": text.rsplit(final_answer_action, 1)[0].strip(),
                 }, text
            )
        # \s matches against tab/newline/whitespace
        regex = r"ASSISTANT\s*Action\s*\d*\s*:(.*?)\nASSISTANT\s*Action\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, text, re.DOTALL)
        if not match:
            # Chat-completion models may return the final response directly
            # after an observation even when the prompt requests the legacy
            # "ASSISTANT Response:" prefix.  Preserve strict action parsing,
            # but accept non-action text as the terminal user-facing answer.
            if os.getenv("TOOLALPACA_LENIENT_PARSE", "0") == "1" \
                    and text.strip() and "ASSISTANT Action" not in text:
                return AgentFinish(
                    {"output": text.strip(), "Final Thought": ""},
                    text,
                )
            raise ValueError(f"Could not parse LLM output: `{text}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        return AgentAction(action, action_input.strip(" ").strip('"'), text)
