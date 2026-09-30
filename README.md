<div align="center">

# When Forgetting Is Not Catastrophic: On the Mechanics of Spurious Forgetting

**Vedant Palit, Florent Draye, Nicolas Zucchet, Zhijing Jin, Bernhard Schölkopf**

[![arXiv](https://img.shields.io/badge/arXiv-coming%20soon-b31b1b.svg)](#citation)
[![Python](https://img.shields.io/badge/python-3.10%2B-3776ab.svg?logo=python&logoColor=white)](https://www.python.org/)
[![JAX](https://img.shields.io/badge/JAX-0.4-a259ff.svg)](https://github.com/jax-ml/jax)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4%2B-ee4c2c.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![uv](https://img.shields.io/badge/managed%20with-uv-de5fe9.svg)](https://docs.astral.sh/uv/)

</div>

<p align="center">
  <img src="assets/fig1.png" alt="Catastrophic versus spurious forgetting" width="100%">
</p>

<p align="center"><em>
A model is pretrained on two non-overlapping sets of facts, A and B, and then finetuned on new facts.
<b>Left:</b> finetuning on C, catastrophic forgetting: A and B decline slowly together.
<b>Right:</b> finetuning on B′, spurious forgetting: accuracy on A collapses, partly recovers, then erodes, while B is retained.
</em></p>

This repository contains the code for all experiments in the paper, at three scales:

| Setting | Framework | Code |
| --- | --- | --- |
| Minimal model (associative memory) | JAX | [`toy_final_iclr/`](toy_final_iclr) |
| Transformer trained from scratch | JAX / Flax | [`src/`](src), [`scripts/`](scripts) |
| Pretrained language model (OLMo 2 1B) | PyTorch | [`llm/`](llm), [`llm_real/`](llm_real) |

## Installation

The minimal model and the transformer use JAX; install them and the plotting tools with [uv](https://docs.astral.sh/uv/):

```bash
uv sync --group analysis
mkdir -p plots
```

The pretrained language model uses PyTorch, in a separate environment:

```bash
uv venv .venv-llm
uv pip install --python .venv-llm --group llm
```

Run all commands from the repository root unless stated otherwise. The JAX commands below assume the environment is active (or prefix them with `uv run`), and the language-model commands use `.venv-llm/bin/python`.

## Minimal model

Collapse, recovery and erosion of the old facts in the associative memory:

```bash
cd toy_final_iclr
python paper_base_runs.py --seeds 0-9 --out paper_base.json
python plot_paper_base.py
```

## Transformer

Pretrain on the two sets of old facts, then finetune on new facts whose answers lie in one region (`disjoint`), or in both regions for comparison (`all_values`):

```bash
python -m src.experiments.mlpfree_pretrain
python -m src.experiments.mlpfree_injection --pretrain_step 16000 --inject_seed 0 --inject_condition disjoint
python -m src.experiments.mlpfree_injection --pretrain_step 16000 --inject_seed 0 --inject_condition all_values
```

## Pretrained language model

Select the old facts OLMo 2 1B knows, build the new facts, and finetune:

```bash
.venv-llm/bin/python -m llm.gate_a_set --out llm/out/gate.json
.venv-llm/bin/python -m llm.build_b --n_people 250 --out_dir llm/out
.venv-llm/bin/python -m llm.inject --lr 1e-5 --steps 400
```

## Citation

If you find this work useful, please cite:

```bibtex
@article{palit2026forgetting,
  title   = {When Forgetting Is Not Catastrophic: On the Mechanics of Spurious Forgetting},
  author  = {Palit, Vedant and Draye, Florent and Zucchet, Nicolas and Jin, Zhijing and Sch{\"o}lkopf, Bernhard},
  journal = {arXiv preprint arXiv:XXXX.XXXXX},
  year    = {2026}
}
```
