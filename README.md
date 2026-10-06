# GNN-Guided Antibiotic Discovery

**Uncertainty-aware Graph Attention Networks for DNA Gyrase B (GyrB) inhibitor discovery — with an honest account of where the model fails.**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/christophergaughan/GNN-Antibiotics/blob/main/notebooks/GNN_Antibiotic_Discovery_GyrB.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](#reproducing)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)

**Christopher L. Gaughan, Ph.D.** — *AntibodyML Consulting LLC*

---

## TL;DR

An end-to-end, reproducible pipeline from ChEMBL bioactivity data to ranked, multi-objective
antibiotic candidates against **DNA Gyrase Subunit B** (GyrB; ChEMBL target `CHEMBL240`,
structure PDB `4DUH`):

> GAT activity model → ensemble uncertainty → generative expansion → drug-likeness/ADMET
> filtering → AutoDock Vina docking → Pareto-based lead optimization.

It is built to report **honest generalization**, not flattering metrics.

## Why this repo reads differently

Most ML drug-discovery demos quote a single random-split AUC and stop. This one does the
opposite: it measures the gap between random and scaffold splits, shows that known clinical
inhibitors are *missed* by the model, and reports a weak ML–docking correlation. Those are
not blemishes in the writeup — they **are** the result, and they are the point.

## Headline numbers (scaffold split, 5-model ensemble)

| Metric | Scaffold split | Random split |
|---|---|---|
| Test AUC | **0.64** (± 0.03) | 0.70 (± 0.02) |
| Test accuracy | 0.79 | 0.76 |
| Honest generalization gap | **+0.055 AUC** | — |

Dataset: ~1,000 GyrB compounds, **232 active / 768 inactive** (≈3:1 imbalance). The scaffold
(Murcko) split forces prediction on *entirely unseen chemotypes*; the random split lets
structurally similar molecules leak across train/test and inflates the AUC. The ~0.055 gap is
the cost of that leakage — the number a random-split-only report would quietly hide.

![Scaffold vs. random split comparison](figures/scaffold_split_comparison.png)
*Scaffold-split AUC vs. the inflated random-split AUC — the "honest vs. flattering" gap.*

![ROC curves](figures/roc_curves.png)
*Ensemble ROC under the scaffold split.*

## Pipeline

1. **Data** — Paginated ChEMBL query (`CHEMBL240`, IC50), deduplicated to most-potent per
   compound, standardized to nM, quality-filtered for valid SMILES/values.
2. **Splitting** — Bemis–Murcko scaffold split for generalization to novel chemical classes.
3. **Representation** — 9-D atom features (atomic number, degree, formal charge, hybridization,
   aromaticity, H-count, in-ring, ring size, chirality) + bond features → PyTorch Geometric graphs.
4. **Model** — 3-layer **Graph Attention Network** (4 heads, BatchNorm, ELU, dropout) with
   global attention pooling; attention weights retained for interpretability.
5. **Ensemble** — 5 seeds, ≤150 epochs, early stopping (patience 20) → mean prediction **plus
   standard deviation as an epistemic-uncertainty estimate** (no point estimate goes unqualified).
6. **Generation** — BRICS fragment recombination: **2,000 → 110** drug-like → **2** with P > 0.5.
7. **Multi-objective filtering** — SAScore, QED, Lipinski; ADMET flags (hERG, CYP, AMES,
   PAINS/Brenk).
8. **Docking** — GyrB (`4DUH`), custom rigid-receptor PDBQT, AutoDock Vina: **20/20** docked,
   scores **−6.87 to −5.79 kcal/mol**.
9. **Lead optimization** — 32 medicinal-chemistry analogs (bioisosteres, halogen scans,
   heteroatom walks); Pareto front over ML score × docking score.

## Model architecture

```
GATAntibiotics(
  Input: 9 atomic features (atomic number, degree, formal charge, hybridization,
         aromaticity, H-count, ring membership, ring size, chirality)

  (input_proj): Linear(9 -> 128)
  (gat_layers): 3 x GATConv(128, 32, heads=4) + residual connections
  (batch_norms): 3 x BatchNorm1d(128)
  (gate_nn):    Global attention pooling -> Linear(128 -> 64) -> ELU -> Linear(64 -> 1) -> Sigmoid
  (classifier): Linear(128 -> 64) -> ELU -> Dropout -> Linear(64 -> 1)

  Parameters: 68,994
)
```

## Results, in pictures

**Data & training**

![Data distribution](figures/data_distribution.png)
*Activity-class balance and property distributions of the GyrB set (≈3:1 inactive:active).*

![Training curves](figures/training_curves.png)
*Ensemble training with early stopping (patience 20).*

**Generation & drug-likeness**

![Generated candidate distribution](figures/generated_candidates_distribution.png)
*Predicted-activity distribution of BRICS-generated candidates (2,000 → 110 drug-like → 2 with P > 0.5).*

![Synthetic accessibility](figures/synthetic_accessibility.png)
*SAScore filtering; BRICS recombination can produce synthetically unrealistic molecules, so SAScore ≤ 4 removes the worst offenders.*

**ADMET & docking**

![ADMET profile](figures/admet_profile.png)
*ADMET flags across the top candidates: PAINS 0/50, Brenk 50/50 (pyridine N-oxide motif — see notebook caveat), hERG high-risk 30/50, AMES all low.*

![Docking validation](figures/docking_validation.png)
*AutoDock Vina into GyrB (`4DUH`): 20/20 docked, −6.87 to −5.79 kcal/mol. ML–docking correlation is only r = 0.291 (see below).*

![Pareto frontier](figures/pareto_frontier.png)
*ML score × docking score kept as separate axes — a Pareto front, not a collapsed "validated" number.*

![Lead structures](figures/lead_structures.png)
*Pareto-optimal leads; 32 analogs explored (12/30 improved ML, 8/30 docking, 4/30 both).*

## What the model gets wrong (read this before trusting the plots)

- **It misses every known inhibitor it was shown.** Novobiocin, chlorobiocin, a coumermycin
  fragment, a pyrrolamide, and an aminopyrimidine all scored as *inactive* (P ≈ 0.05–0.18).
  This is a training-distribution bias — ChEMBL GyrB data skews toward synthetic HTS/optimization
  series, while the aminocoumarins are natural products underrepresented in training.
- **Its "confident" calls are barely above chance.** Under 3:1 imbalance, even top scaffold-split
  predictions sit at P ≈ 0.47–0.48, and 5 of the top 6 are false positives.
- **ML and docking disagree (r = 0.291).** Docking here is a *triage filter and an orthogonal
  sanity check*, not a confirmation of activity.
- **There is no wet-lab validation.** Every candidate is a hypothesis for synthesis
  prioritization — nothing in this repo demonstrates binding or efficacy.

![Known GyrB inhibitors scored inactive](figures/attention_known_inhibitors.png)
*The model calls known GyrB inhibitors inactive — an honest readout of its applicability domain, not a fluke.*

![Attention: actives vs. inactives](figures/attention_comparison.png)
*Attention still lands on chemically sensible atoms (fluorines, heterocyclic N, aromatic cores), even where the call is wrong.*

### On docking, specifically

Docking scores **rank**; they do not **confirm**. A −6.9 kcal/mol Vina score is a reason to look
closer, not evidence of binding. ML score and docking score are kept as **separate axes** (hence
the Pareto treatment) precisely because they disagree — collapsing them into one "validated"
number would be the exact overclaim this pipeline is designed to avoid.

## Usage

**Training**

```python
from models import GATAntibiotics, train_model

model = GATAntibiotics(
    input_dim=9, hidden_dim=128, num_layers=3, num_heads=4, dropout=0.3
).to(device)

model, history = train_model(model, train_loader, val_loader, epochs=150, patience=20)
```

**Inference with uncertainty**

```python
def ensemble_predict(models, smiles, device):
    """Prediction + epistemic uncertainty from the ensemble."""
    graph = mol_to_graph(smiles, label=0)
    batch = Batch.from_data_list([graph]).to(device)
    preds = []
    for m in models:
        m.eval()
        with torch.no_grad():
            preds.append(torch.sigmoid(m(batch)).item())
    return float(np.mean(preds)), float(np.std(preds))

mean, std = ensemble_predict(ensemble_models, "CC1=CC=C(C=C1)C2=CN=C(N=C2N)N", device)
print(f"P(active) = {mean:.3f} ± {std:.3f}")
```

**Attention extraction (interpretability)**

```python
model.eval()
out, attention_weights, gate = model(batch, return_attention=True)
# attention_weights: per-layer (num_edges, num_heads) tensors
# gate: (num_atoms, 1) global attention-pooling weights
```

## Reproducing

Colab-first by design (reference environment: **A100, High-RAM**). Open the notebook with the
Colab badge above; it self-installs dependencies and mounts Google Drive for persistence.
`requirements.txt` is a reference version floor rather than the primary install path — for an
exact lock, capture `pip freeze` from the Colab runtime (torch / torch-geometric builds are
CUDA-specific). ChEMBL data is pulled live at run time, so no dataset is vendored here.

## Repo layout

```
notebooks/
  GNN_Antibiotic_Discovery_GyrB.ipynb   # main v2 pipeline (this README describes it)
  level1_proof_of_concept.ipynb         # Level-1 GCN prototype it builds on
figures/                                 # plots exported from the notebook (see extract_figures.py)
extract_figures.py                       # regenerates figures/ from the notebook outputs
requirements.txt                         # reference dependency versions
LICENSE                                  # MIT
```

## Methods & references

*Bibliographic details below are given in good faith from memory; verify against Paperpile
before any manuscript use.*

1. Veličković P, Cucurull G, Casanova A, Romero A, Liò P, Bengio Y. "Graph Attention Networks."
   *International Conference on Learning Representations (ICLR)*, 2018. arXiv:1710.10903.
2. Fey M, Lenssen JE. "Fast Graph Representation Learning with PyTorch Geometric." *ICLR Workshop
   on Representation Learning on Graphs and Manifolds*, 2019. arXiv:1903.02428.
3. Bemis GW, Murcko MA. "The properties of known drugs. 1. Molecular frameworks." *Journal of
   Medicinal Chemistry*, 1996;39(15):2887–2893. doi:10.1021/jm9602928.
4. Degen J, Wegscheid-Gerlach C, Zaliani A, Rarey M. "On the Art of Compiling and Using
   'Drug-Like' Chemical Fragment Spaces." *ChemMedChem*, 2008;3(10):1503–1507.
   doi:10.1002/cmdc.200800178.
5. Ertl P, Schuffenhauer A. "Estimation of synthetic accessibility score of drug-like molecules
   based on molecular complexity and fragment contributions." *Journal of Cheminformatics*,
   2009;1:8. doi:10.1186/1758-2946-1-8.
6. Bickerton GR, Paolini GV, Besnard J, Muresan S, Hopkins AL. "Quantifying the chemical beauty
   of drugs." *Nature Chemistry*, 2012;4(2):90–98. doi:10.1038/nchem.1243.
7. Trott O, Olson AJ. "AutoDock Vina: improving the speed and accuracy of docking with a new
   scoring function, efficient optimization, and multithreading." *Journal of Computational
   Chemistry*, 2010;31(2):455–461. doi:10.1002/jcc.21334.
8. Eberhardt J, Santos-Martins D, Tillack AF, Forli S. "AutoDock Vina 1.2.0: New Docking Methods,
   Expanded Force Field, and Python Bindings." *Journal of Chemical Information and Modeling*,
   2021;61(8):3891–3898. doi:10.1021/acs.jcim.1c00203.
9. Zdrazil B, Felix E, Hunter F, et al. "The ChEMBL Database in 2023: a drug discovery platform
   spanning multiple bioactivity data types and time periods." *Nucleic Acids Research*,
   2024;52(D1):D1180–D1192. doi:10.1093/nar/gkad1004.
10. Landrum G, et al. "RDKit: Open-source cheminformatics." https://www.rdkit.org
11. Ramsundar B, Eastman P, Walters P, Pande V, Leswing K, Wu Z. *Deep Learning for the Life
    Sciences.* O'Reilly Media, 2019. (DeepChem)

## License

MIT © Christopher L. Gaughan. See [LICENSE](LICENSE).

---

*This is a computational hypothesis-generation pipeline for research and portfolio purposes.
It is not a validated drug-discovery result and makes no therapeutic claims.*
