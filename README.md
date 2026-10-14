<div align="center">

# When Unlearning Meets the Harness: Helpful Assistance, Harmful Recovery

</div>

## Abstract

Large language model (LLM) unlearning aims to mitigate privacy and safety risks by selectively removing targeted knowledge or capabilities while preserving useful behavior. In deployment, LLM agents rely on harnesses that provide reasoning guidance and output repair to improve task execution. However, the security issues of this assistance remain insufficiently understood, leaving a substantial gap between model-level unlearning and robustness in deployed agents. Using tool unlearning as a concrete setting, we identify a failure mode *harness-mediated recovery*: harness assistance intended to improve task performance restores apparently suppressed tool-use capabilities without changing model parameters, optimizing adversarial prompts, or retrieving forgotten content. Experiments on ToolAlpaca and the Berkeley Function Calling Leaderboard (BFCL) reveal two mechanisms for this recovery: reasoning assistance before generation elicits suppressed tool calls, while schema repair after generation reconstructs executable calls from malformed outputs. Therefore, we formulate unlearning robustness at the level of the complete model--harness system and introduce the *Harness Recovery Rate* (HRR) to quantify recovery among forget-set examples that fail under the default interface. We further propose *Harness-Aware Robust Unlearning* (HARU), which incorporates harness-recoverable behaviors into the unlearning objective. Across benchmarks, HARU strengthens robustness to harness-mediated recovery and achieves strong forgetting while preserving substantial retained-tool utility. Our findings further demonstrate that harness environments constitute a critical and underexplored security dimension, where deployment-time interactions can fundamentally alter intended security properties.

## Get Started

We implement tool unlearning on ToolAlpaca and the Berkeley Function Calling Leaderboard (BFCL) to evaluate whether harness assistance causes unlearned agents to leak forgotten tool-use capabilities. We also implement **Harness-Aware Robust Unlearning (HARU)** for more robust tool unlearning.

For setup, evaluation, and unlearning instructions, see:

- **ToolAlpaca:** [Get started](ToolAlpaca/README.md)
