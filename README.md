# A multi-agentic framework for real-time, autonomous freeform metasurface design

**Publication out now in** [Science Advances](https://www.science.org/doi/10.1126/sciadv.adx8006)

This repository hosts the resources accompanying our study, **[A multi-agentic framework for real-time, autonomous freeform metasurface design](https://www.science.org/doi/10.1126/sciadv.adx8006)**.

---

## New: Temporal Modulation Extensions

> **Contributed by [Rashedul Albab](https://github.com/rashedulalbab)**
>
> This fork extends MetaChat with **temporal modulation capabilities** for time-varying metasurface design — enabling the surrogate solver and agentic design stack to handle dynamic, time-switched electromagnetic responses.

### At a Glance

| # | Contribution | Type | Key Files |
|:-:|---|---|---|
| 1 | [Temporal FiLM Conditioning](#contribution-1-temporal-film-conditioning) | Modified | `learners.py`, `dataloader.py`, `train.py`, `config.yaml` |
| 2 | [Physics-Informed Temporal Loss](#contribution-2-physics-informed-temporal-loss) | New | `temporal_physics.py` |
| 3 | [Backward-Compatible Checkpoint Loading](#contribution-3-backward-compatible-checkpoint-loading) | Modified | `learners.py` |
| 4 | [TemporalModulationAgent (AIM)](#contribution-4-temporalmodulationagent-aim) | New | `temporal_modulation_agent.py` |
| 5 | [Temporal Design API Tool](#contribution-5-temporal-design-api-tool) | New | `temporal_design.py` |

### Contribution 1: Temporal FiLM Conditioning

Extended the FiLM conditioning layer from 2 conditions (wavelength, angle) to **3 conditions** (wavelength, angle, **time/switch state**), enabling the surrogate solver to model time-varying electromagnetic responses. The new temporal input `t` represents the normalized switch state of the metasurface, allowing the network to predict field distributions at any point during a temporal modulation cycle.

**Modified files**: `multi_film_angle_dec_fwdadj_sample_learners.py`, `multi_film_angle_dec_fwdadj_sample_otf_dataloader.py`, `multi_film_angle_dec_fwdadj_sample_otf_train.py`, `config.yaml`

### Contribution 2: Physics-Informed Temporal Loss

Created a new `temporal_physics.py` module that adds two physics-informed loss terms to enforce Maxwell's temporal boundary conditions during training:
- **D/B continuity loss**: Enforces continuity of displacement field **D = εE** and magnetic flux density **B = μH** at temporal switching boundaries.
- **Frozen eigenmode loss**: During the inductive state, penalizes changes in the normalized spatial eigenmode profile — only amplitude may vary.

**New file**: `film-waveynet/source_code/temporal_physics.py`

### Contribution 3: Backward-Compatible Checkpoint Loading

Implemented zero-padding weight upgrade utilities (`upgrade_film_state_dict`, `load_legacy_checkpoint`) that allow pretrained 2-condition `best_model.pt` checkpoints to be loaded directly into the new 3-condition architecture **without retraining**. The time column weights are initialized to zero, guaranteeing mathematically identical outputs to the original model.

**Modified file**: `multi_film_angle_dec_fwdadj_sample_learners.py`

### Contribution 4: TemporalModulationAgent (AIM)

Created a new AIM agent (`TemporalModulationAgent`) specialized for temporal metasurface design that:
- Accepts natural-language prompts like *"Design a metasurface that suppresses voltage across load between 7–17 ns"*
- Autonomously derives optimal exponential control signal parameters **σ(t) = σ₀ · exp(-t/τ)**
- Invokes the temporal FiLM WaveY-Net for multi-timestep field simulation
- Outputs optimized geometry + control signal parameters

**New file**: `metachat-aim/agent/temporal_modulation_agent.py`

### Contribution 5: Temporal Design API Tool

Built a new `TemporalDesignAPI` tool for the AIM agent stack with two key methods:
- `design_temporal_metasurface()` — Simulate metasurface response at multiple time steps with configurable control signals (exponential, step, linear)
- `optimize_control_signal()` — Gradient-based optimization of (σ₀, τ) parameters over a target suppression window

**New file**: `metachat-aim/tools/design/temporal_design.py`

### Verification

All contributions have been verified with **37 automated tests** (import checks, tensor shape validation, mathematical correctness, API routing, and cross-module integration) — all passing on CPU without GPU, training data, or API keys. See `verify_contributions.py` for details.

---

## Overview

We present *MetaChat*, a multi-agentic computer-aided design framework, which combines agency with millisecond-speed deep learning surrogate solvers to automate and accelerate photonics design. MetaChat is capable of performing complex freeform design tasks in nearly real-time, as opposed to the days-to-weeks required by the manual use of conventional computing methods and resources.

Near real-time, multi-objective, multi-wavelength autonomous metasurface design is enabled by two key contributions:
- **Agentic Iterative Monologue (AIM):** *A novel agentic system designed to seamlessly automate multiple-agent collaboration, human-designer interaction, and computational tools*
- **FiLM WaveY-Net:** *A semi-general fullwave surrogate solver, which supports conditional fullwave modeling—enabling simulations with variable conditions, including source angle, wavelength, material, and device topology—while maintaining high fidelity to the governing physics*

![MetaChat framework overview](figs/fig1.png)

## Repository structure

- `metachat-aim/`: Source code for the AIM agentic design stack.
- `film-waveynet/`: Source code for the FiLM WaveY-Net surrogate solver, including scripts for training and inference (pretrained weights downloadable via [Zenodo](https://zenodo.org/records/15802727), training and validation data downloadable via [Stanford Digital Repository](https://purl.stanford.edu/dq123fg9049); see below).
- `web-app/`: Code for an example web app, which includes a frontend and a GPU server backend, to run MetaChat.

## Data availability

All data used for training and validation in the study and referenced by the code here (dielectric structures, sources, Ex, Ey, and Hz fields) can be downloaded from the [Stanford Digital Repository](https://purl.stanford.edu/dq123fg9049). Further information can be found on the [Metanet Page](http://metanet.stanford.edu/search/metachat/). The pretrained `best_model.pt` checkpoint is hosted on [Zenodo](https://zenodo.org/records/15802727).

## Reference

The following BibTeX entry can be used to cite MetaChat, this code, and data:

```
@article{lupoiu2025multiagentic,
	title = {A multi-agentic framework for real-time, autonomous freeform metasurface design},
	volume = {11},
	url = {https://www.science.org/doi/full/10.1126/sciadv.adx8006},
	doi = {10.1126/sciadv.adx8006},
	language = {en},
	number = {44},
	journal = {Science Advances},
	author = {Lupoiu, Robert and Shao, Yixuan and Dai, Tianxiang and Mao, Chenkai and Edée, Kofi and Fan, Jonathan A.},
	year = {2025},
}
```

## Contributors

- **Original authors**: Robert Lupoiu, Yixuan Shao, Tianxiang Dai, Chenkai Mao, Kofi Edée, Jonathan A. Fan
- **Temporal modulation extensions**: Rashedul Albab

## Contact

Corresponding author: jonfan@stanford.edu

If you have any questions or need help setting up either AIM or FilM WaveY-Net, don't hesitate to reach out!
