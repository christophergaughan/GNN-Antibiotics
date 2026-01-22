# GNN-Guided Antibiotic Discovery: A Rigorous Computational Pipeline

**Targeting DNA Gyrase Subunit B (GyrB) with Uncertainty-Aware Graph Attention Networks**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Overview

This project implements a **scientifically rigorous** pipeline for Graph Neural Network-guided antibiotic discovery. Unlike typical ML demonstrations that prioritize metrics on random splits, this work emphasizes:

- **Honest generalization estimates** via scaffold-based splitting
- **Uncertainty quantification** through ensemble methods
- **Mechanistic interpretability** via attention weight visualization
- **Multi-modal validation** combining ML predictions, ADMET profiling, and molecular docking

### Target: DNA Gyrase Subunit B (GyrB)

DNA gyrase is a validated antibacterial target with decades of clinical precedent. GyrB, the ATPase subunit, offers opportunities for:
- Overcoming fluoroquinolone resistance (different binding site than GyrA)
- Novel mechanism of action (ATP-competitive inhibition)
- Reduced cross-resistance with existing antibiotics

---

## Key Results

### Honest vs. Inflated Metrics

| Split Type | Test AUC | Interpretation |
|------------|----------|----------------|
| **Scaffold** | 0.638 ± 0.03 | Generalization to novel chemotypes |
| Random | 0.695 ± 0.02 | Inflated by scaffold memorization |
| **Δ** | **+0.057** | The "bullshit gap" |

> *"Random splits overestimate performance by ~9% relative AUC. When prioritizing compounds for synthesis, that 9% is the difference between a productive campaign and chasing ghosts."*

### Attention Visualization

The model learns chemically meaningful features:

![Attention Comparison](figures/attention_comparison.png)

- **True Actives**: Consistent biaryl scaffolds with heterocyclic termini
- **Attention Focus**: Fluorines, heterocyclic nitrogens, aromatic cores
- **Failure Modes**: Transparent and interpretable

### Literature Validation

![Known Inhibitors](figures/attention_known_inhibitors.png)

Known GyrB inhibitors (novobiocin, chlorobiocin) are predicted as inactive—revealing the training data's applicability domain. The attention patterns still highlight pharmacophore-relevant atoms (coumarin carbonyls, lactone oxygens), demonstrating the model learned real chemistry despite limited training coverage.

---

## Pipeline Architecture

```
ChEMBL GyrB IC50 Data (n=1,000)
        ↓
Scaffold-Based Splitting (Murcko decomposition)
        ↓
GAT Ensemble Training (5 models, different seeds)
        ↓
Uncertainty-Aware Predictions (mean ± std)
        ↓
Molecular Generation + Filtering
    • Chemical validity (RDKit)
    • Drug-likeness (QED > 0.4, Lipinski)
    • Synthetic accessibility (SAScore < 4)
    • Novelty check (not in ChEMBL/PubChem)
        ↓
Multi-Objective Profiling
    • ADMET predictions (hERG, CYP, toxicity)
    • Selectivity (GyrB vs human TopoII)
        ↓
Docking Validation (AutoDock Vina → PDB:4DUH)
        ↓
Pareto-Ranked Candidates with Full Mechanistic Rationale
```

---

## What Makes This Different

| Common Practice | This Pipeline |
|-----------------|---------------|
| Random train/test splits | **Scaffold splits** to test generalization to novel chemotypes |
| Single model, point estimates | **Ensemble of 5 GATs** with uncertainty quantification |
| Black-box predictions | **Attention-based interpretability** validated against known SAR |
| Activity prediction only | **Multi-objective profiling**: ADMET, selectivity, synthetic accessibility |
| ML metrics in isolation | **Orthogonal validation**: docking into GyrB crystal structure |

---

## Model Architecture

```
GATAntibiotics(
  Input: 9 atomic features (atomic num, degree, charge, hybridization, 
         aromaticity, H-count, ring membership, ring size, chirality)
  
  (input_proj): Linear(9 → 128)
  
  (gat_layers): 3 × GATConv(128, 32, heads=4) + residual connections
  (batch_norms): 3 × BatchNorm1d(128)
  
  (gate_nn): Global attention pooling
      Linear(128 → 64) → ELU → Linear(64 → 1) → Sigmoid
  
  (classifier): Linear(128 → 64) → ELU → Dropout → Linear(64 → 1)
  
  Parameters: 68,994
)
```

---

## Repository Structure

```
GNN_antibiotics/
├── v2_rigorous/
│   ├── models/
│   │   ├── gat_scaffold_model_1-5.pt    # Trained ensemble (scaffold split)
│   │   └── gat_random_model_1-5.pt      # Comparison models (random split)
│   ├── data/
│   │   ├── train_scaffold.csv
│   │   ├── val_scaffold.csv
│   │   └── test_scaffold.csv
│   ├── figures/
│   │   ├── data_distribution.png
│   │   ├── ensemble_training_results.png
│   │   ├── attention_top_predictions.png
│   │   ├── attention_comparison.png
│   │   └── attention_known_inhibitors.png
│   ├── candidates/                       # Generated molecules (Section 9)
│   ├── docking/                          # Vina results (Section 11)
│   └── results_summary.json
├── GNN_Antibiotics_v2_Rigorous_Pipeline.ipynb
├── GNN_Generation_of_new_antibiotics.ipynb  # Level 1 prototype
└── README.md
```

---

## Installation

```bash
# Clone repository
git clone https://github.com/[username]/GNN_antibiotics.git
cd GNN_antibiotics

# Install dependencies
pip install torch torch-geometric
pip install rdkit
pip install chembl-webresource-client
pip install deepchem
pip install meeko vina  # For docking
pip install py3Dmol     # For visualization
```

### Google Colab (Recommended)

The notebook is optimized for **Colab with A100 GPU**. Upload `GNN_Antibiotics_v2_Rigorous_Pipeline.ipynb` and run cells sequentially.

---

## Usage

### Training

```python
from models import GATAntibiotics, train_model

# Initialize model
model = GATAntibiotics(
    input_dim=9,
    hidden_dim=128,
    num_layers=3,
    num_heads=4,
    dropout=0.3
).to(device)

# Train with early stopping
model, history = train_model(
    model, 
    train_loader, 
    val_loader,
    epochs=150,
    patience=20
)
```

### Inference with Uncertainty

```python
def ensemble_predict(models, smiles, device):
    """Get prediction with uncertainty from ensemble."""
    graph = mol_to_graph(smiles, label=0)
    batch = Batch.from_data_list([graph]).to(device)
    
    preds = []
    for model in models:
        model.eval()
        with torch.no_grad():
            pred = torch.sigmoid(model(batch)).item()
        preds.append(pred)
    
    return np.mean(preds), np.std(preds)

# Example
mean, std = ensemble_predict(ensemble_models, "CC1=CC=C(C=C1)C2=CN=C(N=C2N)N", device)
print(f"P(active) = {mean:.3f} ± {std:.3f}")
```

### Attention Extraction

```python
# Get attention weights for interpretability
model.eval()
out, attention_weights, gate = model(batch, return_attention=True)

# attention_weights: list of (num_edges, num_heads) tensors per layer
# gate: (num_atoms, 1) global attention pooling weights
```

---

## Results Summary

### Phase 1: Model Training ✓
- Scaffold split AUC: **0.638 ± 0.03** (honest)
- Random split AUC: 0.695 ± 0.02 (inflated)
- 5-model ensemble with uncertainty quantification

### Phase 2: Attention Analysis ✓
- Chemically meaningful attention patterns
- Model attends to: fluorines, heterocyclic N, aromatic cores
- Known GyrB inhibitors reveal applicability domain limits

### Phase 3: Molecular Generation (In Progress)
- SAScore filtering for synthetic accessibility
- QED filtering for drug-likeness
- Novelty verification against ChEMBL/PubChem

### Phase 4: ADMET Profiling (In Progress)
- hERG liability prediction
- CYP inhibition (2D6, 3A4)
- Predicted toxicity endpoints

### Phase 5: Docking Validation (In Progress)
- AutoDock Vina against GyrB (PDB: 4DUH)
- Binding pose analysis
- Convergent evidence mapping

---

## Key Insights

### What the Model Learned
1. **Biaryl + heterocycle motifs** correlate with GyrB activity
2. **Fluorination** is a positive signal (consistent with medicinal chemistry)
3. **Extended aromatic systems** receive high attention

### What the Model Doesn't Know
1. **Aminocoumarin chemotypes** (novobiocin, chlorobiocin) - not in training data
2. **Pyrrolamide scaffolds** - underrepresented in ChEMBL
3. **3D binding geometry** - 2D graph can't capture shape complementarity

### Implications for Deployment
- Use ensemble std to flag uncertain predictions
- Implement applicability domain detection
- Combine with docking for 3D validation
- Human-in-the-loop review for final prioritization

---

## Citation

If you use this work, please cite:

```bibtex
@software{gaughan2026gnn_antibiotics,
  author = {Gaughan, Christopher L.},
  title = {GNN-Guided Antibiotic Discovery: A Rigorous Computational Pipeline},
  year = {2026},
  url = {https://github.com/[username]/GNN_antibiotics}
}
```

---

## Author

**Christopher L. Gaughan, Ph.D.**  
Chemical and Biochemical Engineering, Rutgers University

Expertise: Computational antibody developability, molecular dynamics simulations, ML for drug discovery, bioprocess development.

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Acknowledgments

- ChEMBL database for bioactivity data
- PyTorch Geometric team
- RDKit community
