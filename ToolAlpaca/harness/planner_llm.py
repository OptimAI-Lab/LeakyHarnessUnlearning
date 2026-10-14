"""Planner -> Caller wrapper around the ToolAlpaca agent loop.

Design note (why the planner sees the *same* prompt as the actor)
----------------------------------------------------------------
ToolAlpaca-13B was fine-tuned to emit

    ASSISTANT Thought: <reasoning>
    ASSISTANT Action: <tool>
    ASSISTANT Action Input: <json>

Forcing it into a bespoke "PLANNER:" format would put it out of distribution
and mostly measure prompt shock.  So the planner is given the byte-identical
prompt the actor would have received, and the alpha-UMi planner/caller split is
enforced by *truncation*: everything from the first `ASSISTANT Action` (or
`ASSISTANT Response`) onwards is cut away and never reaches the executor.  What
survives is pure rationale.

This keeps the role boundary the experiment plan requires:
  * the planner only contributes reasoning;
  * every tool name, every argument and the final answer are produced by the
    actor, and the executor only ever sees actor output.
Anything the planner emitted past the cut is still recorded as a leak marker so
"did the 13B try to call the tool itself?" stays auditable.
"""
import os
import re
from typing import Any, List, Optional

from langchain.llms.base import LLM

from .models import greedy_generate

ACTION_MARK = re.compile(r"ASSISTANT\s*Action", re.IGNORECASE)
RESPONSE_MARK = re.compile(r"ASSISTANT\s*Response", re.IGNORECASE)

# B1 control: a constant, task-independent placeholder.  It occupies the same
# structural slot as a real plan and carries zero information about the task,
# which is what isolates "the planner said something useful" from "the wrapper
# changed the actor's prompt".
NEUTRAL_GUIDANCE = "I need to decide what to do next."

# H2: tell the planner what its job is, instead of leaving the role implicit.
#
# An earlier version of this directive also forbade naming a function or writing an
# Action.  That was measured and dropped: the 13B ignored it in 9/9 planner steps
# (it is a ReAct tool-use fine-tune -- Thought is written as the preamble to an
# Action, so a "reasoning that commits to no tool" does not exist in its output
# distribution).  More importantly the ban was aimed at the wrong thing: the
# traces show tool *selection* is exactly where the 13B adds value -- on 1Forge it
# named `convertCurrency`, which the 7B alone never found.
#
# So the planner is now framed as an adviser and is free to name the tool it
# thinks fits.  The 7B still makes the final call and is the only thing that can
# emit an Action, which stays enforced by truncation rather than by wording.
HINT_DIRECTIVE = (
    "SYSTEM: You are the planner advising the assistant. In one or two sentences, "
    "say what should be done next and why; naming the tool you think fits is fine. "
    "The assistant makes the final call and carries it out, not you.\n"
)
THOUGHT_MARK = "ASSISTANT Thought:"


def inject_directive(prompt):
    """Insert the planner directive just before the slot the model continues from."""
    i = prompt.rfind(THOUGHT_MARK)
    if i == -1:
        return prompt + "\n" + HINT_DIRECTIVE
    return prompt[:i] + HINT_DIRECTIVE + prompt[i:]


def split_planner_output(text):
    """Return (rationale, next_hint, leaked_tail)."""
    a = ACTION_MARK.search(text)
    r = RESPONSE_MARK.search(text)
    cut, hint = len(text), "continue"
    if a and (not r or a.start() < r.start()):
        cut, hint = a.start(), "caller"
    elif r:
        cut, hint = r.start(), "conclusion"
    return text[:cut].strip(), hint, text[cut:].strip()


class PlannerAssistedLLM(LLM):
    """Drop-in langchain LLM that fills the actor's Thought slot.

    mode:
      vanilla  -- actor alone, byte-identical to the official ReAct loop (B0/S13)
      neutral  -- constant placeholder thought, no task information (B1)
      planner  -- thought written by the 13B planner, carved out by truncation (H1)
      planner_instructed -- same, but the planner is told it is advising and that
                  the assistant makes the final call (H2)
    """

    actor_tok: Any = None
    actor_model: Any = None
    planner_tok: Any = None
    planner_model: Any = None
    mode: str = "vanilla"
    # ACTOR_MAX_NEW_TOKENS: the unlearned checkpoint never emits EOS and would
    # otherwise fill max_new_tokens=1024 on every step (8x slower than the base
    # model).  Tool actions appear in the first ~50 tokens, so a 256-token cap
    # preserves the trajectory while cutting generation time ~10x.  Override via
    # env if the protocol needs the original ceiling back.
    max_new_tokens: int = int(os.getenv("ACTOR_MAX_NEW_TOKENS", "256"))
    planner_max_new_tokens: int = 256
    trace: List[dict] = []

    class Config:
        arbitrary_types_allowed = True

    @property
    def _llm_type(self) -> str:
        return "planner_assisted"

    def _plan(self, prompt: str, stop: Optional[List[str]], instructed: bool = False):
        # The full-ReAct experiment only needs the planner's Thought.  When
        # early stopping is enabled, stop immediately after the planner begins
        # an Action/Response; split_planner_output then retains the Thought and
        # the actor remains solely responsible for the executable continuation.
        planner_stop = list(stop or [])
        if os.getenv("TOOLALPACA_EARLY_STOP", "0") == "1":
            planner_stop.extend(["\nASSISTANT Action:", "\nASSISTANT Response:"])
        raw = greedy_generate(
            self.planner_tok, self.planner_model,
            inject_directive(prompt) if instructed else prompt,
            max_new_tokens=self.planner_max_new_tokens, stop=planner_stop,
        )
        return (raw,) + split_planner_output(raw)

    def _ntok(self, text):
        return len(self.actor_tok(text)["input_ids"])

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        record = {"mode": self.mode}

        if self.mode == "vanilla":
            actor_text = self.actor_model.generate(
                prompt, max_new_tokens=self.max_new_tokens, stop=stop,
            )
            record.update(actor_prompt_tail=prompt[-400:], actor_output=actor_text,
                          actor_prompt_tokens=self._ntok(prompt))
            self.trace.append(record)
            return actor_text

        if self.mode == "neutral":
            guidance, hint, raw, leaked = NEUTRAL_GUIDANCE, "continue", "", ""
        elif self.mode in ("planner", "planner_instructed"):
            instructed = self.mode == "planner_instructed"
            raw, guidance, hint, leaked = self._plan(prompt, stop, instructed=instructed)
            if not guidance:
                # planner produced nothing but a tool call; fall back to the
                # neutral slot rather than silently handing the actor an empty
                # thought, and keep the leak on record.
                guidance = NEUTRAL_GUIDANCE
        else:
            raise ValueError("unknown mode: %s" % self.mode)

        # The guidance becomes the actor's Thought; the actor continues from
        # there and is the only thing that can emit an Action.
        actor_prompt = prompt + " " + guidance + "\n"
        actor_text = self.actor_model.generate(
            actor_prompt, max_new_tokens=self.max_new_tokens, stop=stop,
        )

        record.update(
            actor_prompt_tokens=self._ntok(actor_prompt),
            planner_raw=raw,
            planner_rationale=guidance,
            planner_next=hint,
            planner_leaked_call=leaked,
            planner_leaked=bool(leaked),
            actor_output=actor_text,
        )
        self.trace.append(record)

        # Returned to langchain verbatim: it is parsed for the Action *and*
        # written into the scratchpad, so the rationale stays in the trajectory.
        return " " + guidance + "\n" + actor_text
