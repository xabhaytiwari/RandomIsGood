# Does Smarter Acquisition Beat Random Sampling?
### A Fully-Controlled Study of Submodular, Density-Aware Active Learning for Fine-Grained Classification

**Abhay Tiwari** — Department of Computer Science and Engineering, Indian Institute of Information Technology, Raichur
`ad23b1001@iiitr.ac.in`

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22807079.svg)](https://doi.org/10.5281/zenodo.22807079)


---

## TL;DR

We build a multi-modal active learning pipeline for 10,000-class iNaturalist classification — batched CELF submodular optimization, density-weighted outlier-robust sampling, taxonomy-aware hard-negative mining, and an audited pseudo-labeling loop. Then we ask the question most active learning papers don't fully control for: **once training epochs, rounds, and label budget are all matched, does the acquisition strategy actually beat random sampling?**

**The honest answer: no.** Random sampling outperforms our full pipeline at every one of 5 active learning rounds, by 1.8–4.9 points Top-1 accuracy (48.18% ± 0.18% vs. 50.52% ± 0.14% at the final round, 3 seeds per arm). The same pipeline clearly *does* beat other engineered acquisition heuristics under identical conditions — uncertainty-only sampling by 3.4 points, CoreSet-style diversity alone by 7.1 points — so the components aren't broken, but sophistication among heuristics doesn't translate into an advantage over no heuristic at all in this setting. This is consistent with a small but growing body of work (Mittal et al., Munjal et al.) showing many published active learning methods fail to beat random once fairly compared.

Independent of that result, we also contribute:
- A **measured** (not asserted) 14–96× GPU memory reduction from batched CELF against a dense O(N²) baseline.
- A **leakage-audited** pseudo-labeling loop maintaining 89–96% accuracy against ground truth frozen before training, with train/test disjointness enforced at runtime.
- An **acquisition-strategy dispatcher** supporting 4 interchangeable strategies (ours, CoreSet-style, uncertainty-only, random) behind one interface, so running a matched-budget random-sampling control is a config change, not a second engineering project.

## Repository Contents

```
.
├── paper.pdf                          # Full paper, camera-ready
├── paper_sources/
│   ├── paper.tex                      # LaTeX source (IEEEtran)
│   ├── make_figures.py                # Regenerates all figures from the results JSON
│   └── figs/                          # All figures (PDF), including two supplementary
│                                       #   ones (fig1/fig2, ours_full-only trajectory)
│                                       #   not embedded in the compiled paper
├── final_script.py                    # Full training/experiment pipeline
├── hf_backup.py                       # Cross-session checkpoint backup (Hugging Face Hub)
└── README.md
```

## Reproducing the Experiments

The pipeline runs a registry of experiments (the full method, 7 component ablations, and
4 baselines) across multiple seeds, with automatic checkpointing and resume:

```bash
pip install torch transformers scikit-learn numpy matplotlib huggingface_hub
python final_script.py
```

Key experiments in `final_script.py`'s `EXPERIMENTS` registry:
- `ours_full` — full pipeline, 3 seeds, 5 rounds, 40+40 epochs/round
- `baseline_random_matched_epochs` — random acquisition, otherwise identical fidelity to `ours_full` (the controlled comparison)
- `ablate_no_*` — seven single-component ablations (reduced schedule: 3 rounds, 15+15 epochs, 1 seed, to keep compute tractable)
- `baseline_uncertainty`, `baseline_coreset` — competing engineered acquisition strategies
- `baseline_supervised_random_175k` — single-shot supervised control (no AL loop)

Results are written to `al_results_all_experiments.json`. Regenerate all figures with:
```bash
cd paper_sources && python make_figures.py
```

### Surviving interrupted sessions (Kaggle / free-tier GPU)
`hf_backup.py` pushes every checkpoint to a private Hugging Face Hub dataset repo the moment
it's written, and restores everything at the start of a new session. One-time setup:
1. Create a free HF account and a write-access token.
2. Set it as an environment variable (`HF_TOKEN`) or a Kaggle Secret.
3. Change `HF_REPO_ID` in `hf_backup.py` to your own `username/repo-name`.

## Citation

If you use this code or reference this work, please cite:

```bibtex
@misc{tiwari2026activelearning,
  author = {Tiwari, Abhay and Jallu, Ramesh K.},
  title  = {Does Smarter Acquisition Beat Random Sampling? A Fully-Controlled
            Study of Submodular, Density-Aware Active Learning for
            Fine-Grained Classification},
  year   = {2026},
  doi    = {PENDING},
  url    = {https://github.com/xabhaytiwari/RandomIsGood}
}
```

See also [`CITATION.cff`](CITATION.cff) for machine-readable citation metadata (GitHub renders
a "Cite this repository" button from this automatically).

## License

Code is released under the MIT License (see [`LICENSE`](LICENSE)). The paper text and figures
are released under CC-BY 4.0 unless noted otherwise.

## Status

This work is currently an independent preprint. An arXiv submission is in progress pending
endorsement (arXiv requires endorsement from an established author in `cs.LG`/`cs.CV` for
first-time submitters as of their January 2026 policy update). This repository and its Zenodo
archive are the canonical citable version in the meantime.
