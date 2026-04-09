# Connectome-Constrained Spiking Olfactory Pathway Model

Spiking neural network model of the larval *Drosophila* olfactory pathway,
constrained by the complete Winding et al. (2023) connectome. All connectivity
(368 neurons, ~14,600 chemical synapses, ~1,400 gap junction pairs) is fixed
to the connectome; only ~449 scalar parameters are learned.

**Paper**: "Connectome-Constrained Spiking Neural Networks Reproduce Emergent
Computations in the Larval *Drosophila* Olfactory Pathway" (CCN 2026).
See [`98_CCN_Revisions.pdf`](98_CCN_Revisions.pdf) for the submitted revision.

## Architecture

```
OR (rate, 21 types) -> ORN (LIF, 42) -> LN (LIF, 108) -> PN (LIF, 72) -> KC (2-comp, 144) <- APL (graded, 2) -> Decoder (28 odors)
                                         ^___v gap junctions              ^___v KC-KC recurrent
```

- **ORN, LN, PN**: Leaky integrate-and-fire with surrogate gradients
- **KC**: Two-compartment (dendrite + axon) with learnable soma conductance
- **APL**: Graded (non-spiking) global inhibition
- **Gap junctions**: LN-LN (354 pairs), PN-PN sister (929), eLN-PN (103)
- **Non-AD contacts**: 7 types (2,200 synapses) with weak initialization
- **STD**: Tsodyks-Markram short-term depression on all chemical synapses
- **Noise**: 6 biologically motivated sources (OR 30% CV, membrane 1mV, background 15pA, synaptic 25%, threshold 1mV, receptor 10%)

## Setup

### 1. Create conda environment

```bash
conda env create -f environment.yml
conda activate connectome-paper
```

### 2. Reproduce paper results (pre-computed data exists)

```bash
jupyter lab notebooks/paper_figures.ipynb
```

Run all cells. The notebook loads pre-trained models and pre-computed results from
`results/`, computes all metrics at runtime, generates all figures, and prints
every value cited in the paper. No training required.

### 3. Train from scratch (optional)

```bash
python run_training.py                # Canonical models (5 seeds, ~2h)
python run_ablation.py                # Retrained ablations (~4h)
python run_posthoc_ablation.py        # Post-hoc ablations (~30min)
python run_std_ablation.py            # STD ablation (~2h)
python run_odor_mixtures.py           # Mixture coding (~1h)
python run_honegger_metric.py         # Honegger sub-additivity (~30min)
python run_teacher_consistency.py     # Teacher/student analysis (~5min)
python run_task_complexity.py         # Task complexity (~4h)
python run_training_energy_only.py    # Energy constraint variants (~8h)
```

All scripts save JSON results to `results/` subdirectories. The notebook
detects existing results and skips recomputation.

## Repository Structure

```
ccn_s_connectome_revisions/
|-- __init__.py                   # Package exports
|-- layers.py                     # Neuron layers (LIF, TwoCompartmentKC, APL, etc.)
|-- model.py                      # SpikingConnectomeConstrainedModel (full pipeline)
|-- dataset.py                    # Dataset loading (Kreher 2008 OR responses)
|-- environment.yml               # Conda environment specification
|-- 98_CCN_Revisions.pdf          # Submitted revision of the paper
|-- README.md                     # This file
|
|-- data/                         # Bundled connectome + ORN data
|   |-- kreher2008/               # ORN response matrices (Kreher et al. 2008)
|   |-- winding2023/              # Collapsed connectivity matrices (Winding et al. 2023)
|   +-- winding2023_compartments/ # Compartment-resolved (aa, ad, da, dd)
|
|-- analysis/                     # Analysis subpackage
|   |-- __init__.py               # Re-exports all analysis functions
|   |-- utils.py                  # Shared primitives (cosine similarity, noisy forward pass)
|   |-- compute.py                # Metrics (decorrelation, Mancini test, concentration invariance)
|   +-- plotting.py               # Plot functions (each returns fig, accepts show kwarg)
|
|-- notebooks/
|   +-- paper_figures.ipynb       # Main notebook: trains models, computes metrics, generates figures
|
|-- run_training.py               # Canonical training (5 seeds x 300 epochs)
|-- run_training_energy_only.py   # C7: Energy constraint variants (9 conditions x 5 seeds)
|-- run_ablation.py               # C1/C3: Retrained ablations + LN threshold sensitivity
|-- run_posthoc_ablation.py       # C1: Post-hoc ablations (gap junctions, APL disabled at eval)
|-- run_std_ablation.py           # STD: Short-term depression ablation (retrained + post-hoc)
|-- run_odor_mixtures.py          # C2: Odor mixture KC coding analysis (post-hoc)
|-- run_honegger_metric.py     # C2: Honegger-style per-KC sub-additivity metric
|-- run_teacher_consistency.py    # C5: Teacher/student parameter consistency analysis
|-- run_task_complexity.py        # C6: Task complexity / KC threshold scaling
|
|-- results/                      # Pre-computed results (loaded by notebook)
|   |-- all_connections_nonad_canonical/  # Canonical models + results (5 seeds)
|   |-- energy_only_c7/                  # C7 energy constraint results (9 x 5 seeds)
|   |-- ablations_c7/                    # C1/C3 retrained ablation results
|   |-- posthoc_ablations/               # C1 post-hoc ablation results
|   |-- std_ablation/                    # STD ablation results
|   |-- odor_mixtures_c2/               # C2 odor mixture results
|   |-- teacher_consistency_c5/          # C5 parameter consistency results
|   +-- task_complexity_c6/              # C6 task complexity results
|
+-- figures/                      # Generated figures (output of notebook)
    |-- 2_core.png                # Figure 2: Pairwise cosine similarity matrices
    |-- 2_core.pdf
    |-- 2d_kc_heatmap.png         # Figure S1: KC activity heatmap across odors
    |-- 2d_kc_heatmap.pdf
    |-- 3_concentration.png       # Figure 3: Concentration invariance
    +-- 3_concentration.pdf
```

## Training

### Canonical models

```bash
python run_training.py
```

Trains 5 models (seeds 42-46) for 300 epochs each. Results saved to
`results/all_connections_nonad_canonical/`.

### Revision experiments

The notebook (`notebooks/paper_figures.ipynb`) contains cells under
**Option A: Run revision experiments from scratch** that invoke each `run_*.py`
script via subprocess. Each cell skips conditions whose results already exist.

| Script | Experiment | Output |
|--------|-----------|--------|
| `run_training_energy_only.py` | C7: Energy constraint (9 conditions x 5 seeds) | `results/energy_only_c7/` |
| `run_teacher_consistency.py` | C5: Parameter consistency (post-hoc) | `results/teacher_consistency_c5/` |
| `run_ablation.py` | C1: Retrained ablations (3 conditions x 5 seeds) | `results/ablations_c7/` |
| `run_ablation.py --ln-quantile` | C3: LN threshold sensitivity (3 thresholds x 5 seeds) | `results/ablations_c7/` |
| `run_posthoc_ablation.py` | C1: Post-hoc ablations (2 conditions x 5 seeds) | `results/posthoc_ablations/` |
| `run_std_ablation.py` | STD: Short-term depression (2 conditions x 5 seeds) | `results/std_ablation/` |
| `run_odor_mixtures.py` | C2: Odor mixture coding (post-hoc, 5 seeds) | `results/odor_mixtures_c2/` |
| `run_honegger_metric.py` | C2: Honegger sub-additivity (post-hoc, 5 seeds) | `results/odor_mixtures_c2/` |
| `run_task_complexity.py` | C6: Task complexity (3 odor counts x 5 seeds) | `results/task_complexity_c6/` |

## Canonical Results (5-seed ensemble, from notebook Cell 25)

| Metric | Value | Biological comparison |
|--------|-------|-----------------------|
| Test accuracy | 71.3% ± 1.6% | Kreher RI: -0.40 to 0.74 |
| Centroid accuracy | 65.2% ± 2.3% | Decoder-free odor discrimination |
| KC sparsity | 12.6% ± 0.3% | Target: 5-10% (Lin et al., 2014) |
| AL decorrelation | ~5% | Modest, as expected |
| MB decorrelation | ~35% | Expansion coding (Bhandawat et al., 2007) |
| APL suppression ratio | 1.85 ± 0.03 (~46%) | ~50% (Mancini et al., 2023) |
| PN dynamic range | 1.25 ± 0.03x | Sublinear gain (Olsen et al., 2010) |
| KC dynamic range | ~1.14x | APL normalization |
| KC pattern similarity | 0.941 | Cross-model convergence |
| Parameter correlation | 0.966 | Connectome constrains solution |

## Emergent Computations (not trained for)

The model spontaneously reproduces the following biological properties, none
of which were part of the training objective:

- **MB-localized decorrelation**: ~35% in MB vs ~5% in AL
- **APL suppression**: KC activity reduced by ~46%, matching optogenetic data
- **Concentration invariance**: gain control compresses OR→PN→KC dynamic range
- **Sub-additive mixture coding**: 75.1% of KCs sub-additive (Honegger: 73%)

## Ablation Studies

Systematic removal of circuit components reveals their individual contributions:

| Ablation | Key finding |
|----------|-------------|
| Gap junctions removed | Dispensable when retrained; post-hoc removal collapses accuracy |
| APL disabled | Decorrelation shifts upstream (AL triples, MB drops) |
| STD removed | Gain control eliminated (PN 2.78x, KC 2.74x vs canonical 1.25x, 1.14x) |
| Shuffled connectome | Accuracy drops, showing specific wiring matters |
| No sparsity loss | 80% KC activity — optimizer overwhelms APL inhibition |
| All-neuron energy | Fails — optimizer silences AL instead of sparsifying KCs |

## Generated Figures

Running the notebook produces:

| Figure | File | Description |
|--------|------|-------------|
| Figure 2 | `figures/2_core.png` | Pairwise cosine similarity matrices (OR→PN→KC decorrelation) |
| Figure 3 | `figures/3_concentration.png` | Concentration invariance (gain control, accuracy, similarity) |
| Figure S1 | `figures/2d_kc_heatmap.png` | KC activity heatmap across odors |

All numerical values cited in the paper are printed by notebook Cell 25 (main paper)
and Cell 42 (revision experiments) for reproducibility.

## Notebook Output Reference

The notebook prints all values cited in the paper for reproducibility. Below are the
exact outputs from the most recent run.

<details>
<summary><b>Cell 25: Main Paper Values</b> (click to expand)</summary>

```
--- Methods: Training Protocol (Section 2.4) ---
  Teacher accuracy:          ~70%

--- Results: Classification & Sparsity (Section 3.1) ---
  Test accuracy:             71.3% +/- 1.6%
  Centroid accuracy:         65.2% +/- 2.3%
  Chance multiplier:         ~20x  (of 3.6%)
  KC sparsity:               12.6% +/- 0.3%
  Decoder-centroid gap:      6.1 pp
  KCs silent per stimulus:   ~87%

--- Results: Decorrelation (Section 3.2) ---
  AL decorrelation:          ~5%
  MB decorrelation:          ~35%
  Total decorrelation:       ~39%

--- Results: APL Inhibition (Section 3.3) ---
  Mancini ratio:             1.85 +/- 0.03
  KC suppression:            ~46%

--- Results: Concentration Invariance (Section 3.4) ---
  PN dynamic range:          1.25 +/- 0.03x
  OR input range:            1.75x
  KC dynamic range:          ~1.14x
  Accuracy range (c=0.3-5):  44--78%

--- Results: Consistency (Section 3.5) ---
  KC consistency:            0.941
  Parameter correlation (r): 0.966

--- Results: Biological Parameters (Section 3.6) ---
  g_soma:                    14.5 +/- 0.1 nS
  Gap junctions:             LN-LN 0.14, PN-PN 0.10, eLN-PN 0.08 nS
```

</details>

<details>
<summary><b>Cell 42: Revision Experiment Values</b> (click to expand)</summary>

```
--- C1: Ablation Studies ---
  RETRAINED:
  Canonical (baseline)             Acc= 69.7+/-2.5%  KC= 12.5+/-0.3%  AL=  4.8  MB= 35.1  Manc= 1.88
  No gap junctions                 Acc= 73.1+/-2.2%  KC= 12.5+/-0.2%  AL=  6.3  MB= 35.9  Manc= 1.83
  No APL                           Acc= 67.3+/-1.2%  KC= 12.2+/-0.2%  AL= 17.1  MB= 31.3  Manc=  N/A
  Shuffled connectome              Acc= 63.9+/-2.7%  KC= 11.7+/-0.3%  AL= 15.7  MB= 35.0  Manc= 2.39
  POST-HOC:
  No gap junctions (post-hoc)      Acc= 22.4+/-1.6%  KC= 14.8+/-0.2%  AL= -0.0  MB= 36.1  Manc= 1.87
  No APL (post-hoc)                Acc= 24.3+/-3.7%  KC= 33.0+/-0.5%  AL=  5.1  MB= 29.8  Manc=  N/A

--- STD Ablation ---
  No STD (retrained)               Acc= 69.6+/-2.0%  KC= 10.3%  AL= 2.4  MB= 40.8  KCx= 2.74
  No STD (post-hoc)                Acc= 53.4+/-2.4%  KC= 22.4%  AL= 7.0  MB= 41.0  KCx= 1.14

--- C2: Odor Mixtures ---
  Mix<->Component similarity             0.806 +/- 0.003
  Honegger sub-additive KCs (2-odor)     75.1% +/- 1.5%  (Honegger: 73%)
  Honegger sub-additive KCs (3-odor)     80.5% +/- 1.3%
  Mix vs individual SVM                   74.6% +/- 1.5%

--- C5: Teacher/Student Consistency ---
  Teacher-teacher weighted r:     0.781
  Student-student weighted r:     0.911  (delta +0.130)
  KC thresholds:                  0.532 -> 0.951
  Decoder weights:                0.787 -> 0.906
  All scalar CVs:                 < 1%

--- C6: Task Complexity ---
  n=7:   upper_bound= 11.7%    n=14:  upper_bound= 35.7%
  n=28:  upper_bound= 47.2%    n=56:  upper_bound= 56.5%

--- C7: Energy Constraint ---
  CE only (no sparsity):    KC= 80.0%  MB decorr= 0.2%  (APL alone fails)
  All-neuron energy:        Silences AL, destroys MB decorrelation
  KC-specific energy:       Preserves AL processing and biological sparsity
```

</details>

## Dependencies

Managed via `environment.yml`. Core requirements:

| Package | Version |
|---------|---------|
| Python | >= 3.10 |
| PyTorch | >= 2.0 |
| NumPy | >= 1.24 |
| SciPy | >= 1.10 |
| Pandas | >= 1.5 |
| Matplotlib | >= 3.7 |
| scikit-learn | >= 1.2 |
| JupyterLab | >= 4.0 |

## Data

Connectome data from Winding et al. (2023) and ORN responses from Kreher et al. (2008)
are bundled in `data/`:

- `kreher2008/` -- ORN response matrices (normalized and raw)
- `winding2023/` -- Collapsed connectivity matrices (31 `.pt` files)
- `winding2023_compartments/` -- Compartment-resolved (aa, ad, da, dd; 32 `.pt` files)
