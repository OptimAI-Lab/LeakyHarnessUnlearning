# When Unlearning Meets the Harness: Helpful Assistance, Harmful Recovery

ToolAlpaca evaluation, unlearning baselines, and **Harness-Aware Robust Unlearning (HARU)**.

## Setup

Run commands from `ToolAlpaca/` with Linux, Python 3.10, and CUDA GPUs.

```bash
cd ToolAlpaca
conda create -n leaky-harness python=3.10 -y
conda activate leaky-harness
pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Evaluation uses GPT-3.5 Turbo (`gpt-3.5-turbo`) for the ToolAlpaca benchmark
simulator and GPT-4 (`gpt-4-0613`) for judging. Set your API key:

```bash
export OPENAI_API_KEY=your-key
```

## Evaluation

We evaluate tool unlearning with and without harness assistance.
`TangQiaoYu/ToolAlpaca-7B` is the original model, and
`TangQiaoYu/ToolAlpaca-13B` is the default harness model.

Each command starts the simulator, evaluates all examples, and judges the results.

**Single model:**

```bash
python evaluate.py --mode single --model TangQiaoYu/ToolAlpaca-7B
```

**With the 13B harness:** the harness supplies reasoning; the evaluated model executes tool calls.

```bash
python evaluate.py --mode harness --model TangQiaoYu/ToolAlpaca-7B
```

Replace `--model` with the checkpoint you want to evaluate. Single-model evaluation
uses one GPU; harness evaluation uses two. The default split is `forget`; add
`--split retain` for retained tools. Results are saved
to `outputs/<split>_<mode>/judge.json`; use `--output` to choose a new run directory.

## Unlearning baselines

Train from the original 7B model (eight GPUs by default; set `NPROC` to change):

```bash
bash tool_unlearn/baselines/run.sh graddiff checkpoints/graddiff --epochs 5
```

| Method | Objective |
| --- | --- |
| `ga` / `graddiff` | Gradient ascent / gradient ascent + retain CE |
| `npo` | Negative preference optimization |
| `tooldelete_sft` / `tooldelete_dpo` | ToolDelete SFT / DPO |

Replace `graddiff` with the desired method and choose a new output directory.
For ToolDelete, run:

```bash
TOOLFREE_MODEL=lmsys/vicuna-7b-v1.1 bash tool_unlearn/baselines/run.sh tooldelete_sft checkpoints/tooldelete_sft --epochs 5
```

This downloads Vicuna and generates targets automatically. For DPO, replace
`tooldelete_sft` with `tooldelete_dpo` in both the method and output directory.

Evaluate the trained model:

```bash
python evaluate.py --mode single --model checkpoints/graddiff/model --output outputs/graddiff_single
python evaluate.py --mode harness --model checkpoints/graddiff/model --output outputs/graddiff_harness
```

## HARU

HARU unlearns for one epoch, mines ten harness-assisted first actions per forget
query, and continues unlearning for two epochs. The reference stays fixed to the
original model. Training and mining use eight GPUs by default.

```bash
python tool_unlearn/haru.py --out_dir checkpoints/haru \
  --retain_loss ce --forget_weight 2 --retain_weight 2
```

Override `--model_dir` or `--planner_model_dir` to use different models.
Evaluate the final checkpoint:

```bash
python evaluate.py --mode single --model checkpoints/haru/final --output outputs/haru_single
python evaluate.py --mode harness --model checkpoints/haru/final --output outputs/haru_harness
```

## Data and source

Frozen forget/retain splits are in `tool_unlearn/data/`; general evaluation data
is in `data/`. Training and evaluation demonstrations are disjoint.
