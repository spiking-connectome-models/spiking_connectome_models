"""
run_odor_mixtures.py

C2: Odor Mixture / Superposition Experiments (Reviewer 2).
Presents combinations of 2-3 simultaneous odors to trained canonical models
and analyzes whether KC population codes produce unique, linearly separable
representations distinct from individual component codes.

Post-hoc analysis only — no retraining. Uses canonical models from
results/all_connections_nonad_canonical/.

Key questions:
  1. Do mixture KC codes differ from individual component codes?
  2. Are mixture representations linearly separable from each other?
  3. Do KC codes show non-linear mixture suppression (consistent with
     sparse coding / competitive inhibition via APL)?

Reference: Honegger et al. 2011 (Neuron) — KC ensemble codes for mixtures.

Results saved to: results/odor_mixtures_c2/
  mixture_results_seed{42-46}.json
  mixture_summary.json

Notebook section: Section B — C2 (odor mixture statistics and figure).
"""
import sys
from pathlib import Path

_pkg_parent = str(Path(__file__).parent.parent)
if _pkg_parent not in sys.path:
    sys.path.insert(0, _pkg_parent)
for _candidate in [
    Path(__file__).parent.parent / 'relative_crawling',
    Path.home() / 'Desktop' / 'relative_crawling',
]:
    if (_candidate / 'connectome_models').is_dir() and str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))
sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
sys.stderr.reconfigure(encoding='utf-8')

import json
import itertools
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score

from spiking_connectome_models.model import SpikingConnectomeConstrainedModel
from spiking_connectome_models.layers import SpikingParams

# ============================================================================
# CONFIG
# ============================================================================
DATA_DIR = None  # resolved at runtime
CANONICAL_DIR = Path(__file__).parent / 'results' / 'all_connections_nonad_canonical'
OUTPUT_DIR = Path(__file__).parent / 'results' / 'odor_mixtures_c2'

NOISE_STD = 0.3
N_TRIALS = 20       # noisy trials per stimulus for stable KC codes
N_STEPS = 30        # timesteps for evaluation (matching canonical eval)
SEEDS = [42, 43, 44, 45, 46]  # all 5 canonical seeds

# How to generate mixture OR responses:
# Element-wise sum (biologically: ORN activation is approximately additive
# for co-presented odorants at moderate concentrations)
# Clamped to [0, 1] after summing.

REALISTIC_PARAMS = SpikingParams(
    v_noise_std=1.0e-3, i_noise_std=15e-12, syn_noise_std=0.25,
    threshold_jitter_std=1.0e-3, orn_receptor_noise_std=0.10,
    circuit_noise_enabled=True,
)


# ============================================================================
# SPIKE ACCUMULATOR
# ============================================================================
class SpikeAccumulator:
    """Captures ORN and LN spikes via forward hooks."""
    def __init__(self):
        self.orn_spikes = None
        self.ln_spikes = None
        self._hooks = []

    def register(self, model):
        def orn_hook(module, input, output):
            spk = output[1]
            self.orn_spikes = spk if self.orn_spikes is None else self.orn_spikes + spk
        def ln_hook(module, input, output):
            spk = output[1]
            self.ln_spikes = spk if self.ln_spikes is None else self.ln_spikes + spk
        self._hooks.append(model.antennal_lobe.orn_neurons.register_forward_hook(orn_hook))
        self._hooks.append(model.antennal_lobe.ln_neurons.register_forward_hook(ln_hook))
        return self

    def reset(self):
        self.orn_spikes = None
        self.ln_spikes = None

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# ============================================================================
# HELPERS
# ============================================================================
def load_canonical_model(data_dir, seed, n_odors):
    """Load trained canonical model."""
    model = SpikingConnectomeConstrainedModel.from_data_dir(
        data_dir, n_odors=n_odors, n_or_types=21, target_sparsity=0.10,
        params=REALISTIC_PARAMS, include_nonad=True)
    model.n_steps_al = N_STEPS
    model.n_steps_kc = N_STEPS
    model_path = CANONICAL_DIR / f'model_seed{seed}.pt'
    state = torch.load(model_path, weights_only=False, map_location='cpu')
    model.load_state_dict(state)
    model.eval()
    return model


def generate_mixture_or(or_responses, odor_indices):
    """Generate mixture OR response by summing component OR responses.

    C2: Element-wise sum of component OR patterns, clamped to [0,1].
    This models the approximate additivity of ORN responses to co-presented
    odorants at moderate concentrations (Kreher et al. 2008).
    """
    mixture = torch.zeros_like(or_responses[0])
    for idx in odor_indices:
        mixture = mixture + or_responses[idx]
    mixture = mixture.clamp(0, 1)
    return mixture


def get_kc_code(model, or_input, n_trials, noise_std):
    """Get KC population code (spike rate vector) for a single stimulus.

    Returns mean KC spike rates across noisy trials, giving a stable
    population code for the stimulus.
    """
    n_or = or_input.shape[0]
    noisy = or_input.unsqueeze(0).expand(n_trials, -1)
    noise = torch.randn_like(noisy) * noise_std
    noisy = (noisy * (1.0 + noise)).clamp(min=0)

    model.eval()
    with torch.no_grad():
        _, info = model(noisy, return_all=True)
        kc_rates = info['kc_spikes'].float() / N_STEPS
    return kc_rates.mean(dim=0)  # mean across trials → (n_kc,)


def cosine_sim(a, b):
    """Cosine similarity between two vectors."""
    a_flat = a.flatten().float()
    b_flat = b.flatten().float()
    norm_a = torch.norm(a_flat)
    norm_b = torch.norm(b_flat)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(F.cosine_similarity(a_flat.unsqueeze(0), b_flat.unsqueeze(0)))


# ============================================================================
# MIXTURE ANALYSIS
# ============================================================================
def analyze_mixtures(model, or_responses, odor_names, seed):
    """Run full mixture analysis on a single model.

    C2 analysis:
    1. Get KC codes for all 28 individual odors
    2. Generate all 2-odor and sampled 3-odor mixtures
    3. Compare mixture codes to component codes
    4. Test linear separability of mixture vs individual codes
    5. Measure mixture suppression (sub-additivity from APL)
    """
    n_odors = or_responses.shape[0]
    print(f"\n  Getting KC codes for {n_odors} individual odors...")

    # 1. Individual odor KC codes
    individual_codes = {}
    for i in range(n_odors):
        code = get_kc_code(model, or_responses[i], N_TRIALS, NOISE_STD)
        individual_codes[i] = code

    # 2. Generate 2-odor mixtures (all C(28,2) = 378 pairs)
    pairs = list(itertools.combinations(range(n_odors), 2))
    print(f"  Generating {len(pairs)} 2-odor mixture codes...")

    pair_results = []
    for i, j in pairs:
        mix_or = generate_mixture_or(or_responses, [i, j])
        mix_code = get_kc_code(model, mix_or, N_TRIALS, NOISE_STD)

        # Similarity of mixture to each component
        sim_to_i = cosine_sim(mix_code, individual_codes[i])
        sim_to_j = cosine_sim(mix_code, individual_codes[j])

        # Linear prediction: average of component codes
        linear_pred = (individual_codes[i] + individual_codes[j]) / 2.0
        sim_to_linear = cosine_sim(mix_code, linear_pred)

        # Sparsity of mixture vs components
        mix_sparsity = (mix_code > 0).float().mean().item()
        comp_i_sp = (individual_codes[i] > 0).float().mean().item()
        comp_j_sp = (individual_codes[j] > 0).float().mean().item()

        # Suppression ratio: mixture sparsity vs mean component sparsity
        # <1 means mixture is sparser (sub-additive, APL suppression)
        mean_comp_sp = (comp_i_sp + comp_j_sp) / 2.0
        suppression = mix_sparsity / mean_comp_sp if mean_comp_sp > 0 else 1.0

        pair_results.append({
            'odors': [int(i), int(j)],
            'odor_names': [odor_names[i], odor_names[j]],
            'sim_to_comp1': sim_to_i,
            'sim_to_comp2': sim_to_j,
            'sim_to_linear_pred': sim_to_linear,
            'mix_sparsity': mix_sparsity,
            'comp_sparsities': [comp_i_sp, comp_j_sp],
            'suppression_ratio': suppression,
        })

    # 3. Sample 3-odor mixtures (50 random triplets for speed)
    rng = np.random.RandomState(seed + 3000)
    n_triplets = min(50, len(list(itertools.combinations(range(n_odors), 3))))
    all_triplets = list(itertools.combinations(range(n_odors), 3))
    triplet_indices = rng.choice(len(all_triplets), n_triplets, replace=False)
    triplets = [all_triplets[k] for k in triplet_indices]
    print(f"  Generating {len(triplets)} 3-odor mixture codes...")

    triplet_results = []
    for i, j, k in triplets:
        mix_or = generate_mixture_or(or_responses, [i, j, k])
        mix_code = get_kc_code(model, mix_or, N_TRIALS, NOISE_STD)

        linear_pred = (individual_codes[i] + individual_codes[j] + individual_codes[k]) / 3.0
        sim_to_linear = cosine_sim(mix_code, linear_pred)

        mix_sparsity = (mix_code > 0).float().mean().item()
        mean_comp_sp = np.mean([(individual_codes[x] > 0).float().mean().item()
                                for x in [i, j, k]])
        suppression = mix_sparsity / mean_comp_sp if mean_comp_sp > 0 else 1.0

        triplet_results.append({
            'odors': [int(i), int(j), int(k)],
            'sim_to_linear_pred': sim_to_linear,
            'mix_sparsity': mix_sparsity,
            'suppression_ratio': suppression,
        })

    # 4. Linear separability: can an SVM distinguish mixture from individual KC codes?
    print("  Testing linear separability...")

    # Collect multiple noisy trials for SVM
    n_svm_trials = 10
    X_individual = []
    y_individual = []
    for i in range(n_odors):
        noisy = or_responses[i].unsqueeze(0).expand(n_svm_trials, -1)
        noise = torch.randn_like(noisy) * NOISE_STD
        noisy = (noisy * (1.0 + noise)).clamp(min=0)
        with torch.no_grad():
            _, info = model(noisy, return_all=True)
            kc_rates = info['kc_spikes'].float() / N_STEPS
        X_individual.append(kc_rates.numpy())
        y_individual.extend([i] * n_svm_trials)

    X_individual = np.vstack(X_individual)
    y_individual = np.array(y_individual)

    # Individual odor classification accuracy (baseline)
    try:
        svm_indiv = LinearSVC(max_iter=5000, dual='auto')
        indiv_scores = cross_val_score(svm_indiv, X_individual, y_individual, cv=5)
        indiv_svm_acc = float(np.mean(indiv_scores))
    except Exception:
        indiv_svm_acc = float('nan')

    # Mixture vs individual: binary classification
    # Sample 50 random 2-odor mixtures for SVM
    rng2 = np.random.RandomState(seed + 4000)
    mix_sample = rng2.choice(len(pairs), min(50, len(pairs)), replace=False)
    X_mixture = []
    for idx in mix_sample:
        i, j = pairs[idx]
        mix_or = generate_mixture_or(or_responses, [i, j])
        noisy = mix_or.unsqueeze(0).expand(n_svm_trials, -1)
        noise = torch.randn_like(noisy) * NOISE_STD
        noisy = (noisy * (1.0 + noise)).clamp(min=0)
        with torch.no_grad():
            _, info = model(noisy, return_all=True)
            kc_rates = info['kc_spikes'].float() / N_STEPS
        X_mixture.append(kc_rates.numpy())

    X_mixture = np.vstack(X_mixture)

    # Binary: individual (0) vs mixture (1)
    X_binary = np.vstack([X_individual, X_mixture])
    y_binary = np.array([0] * len(X_individual) + [1] * len(X_mixture))
    try:
        svm_binary = LinearSVC(max_iter=5000, dual='auto')
        binary_scores = cross_val_score(svm_binary, X_binary, y_binary, cv=5)
        binary_svm_acc = float(np.mean(binary_scores))
    except Exception:
        binary_svm_acc = float('nan')

    # Inter-mixture discrimination: can we tell mixtures apart?
    X_mix_disc = []
    y_mix_disc = []
    for mix_idx, pair_idx in enumerate(mix_sample[:20]):  # 20 mixtures
        i, j = pairs[pair_idx]
        mix_or = generate_mixture_or(or_responses, [i, j])
        noisy = mix_or.unsqueeze(0).expand(n_svm_trials, -1)
        noise = torch.randn_like(noisy) * NOISE_STD
        noisy = (noisy * (1.0 + noise)).clamp(min=0)
        with torch.no_grad():
            _, info = model(noisy, return_all=True)
            kc_rates = info['kc_spikes'].float() / N_STEPS
        X_mix_disc.append(kc_rates.numpy())
        y_mix_disc.extend([mix_idx] * n_svm_trials)

    X_mix_disc = np.vstack(X_mix_disc)
    y_mix_disc = np.array(y_mix_disc)
    try:
        svm_mix = LinearSVC(max_iter=5000, dual='auto')
        mix_scores = cross_val_score(svm_mix, X_mix_disc, y_mix_disc, cv=5)
        mix_svm_acc = float(np.mean(mix_scores))
    except Exception:
        mix_svm_acc = float('nan')

    # 5. Summary statistics
    pair_sims_to_comp = [(r['sim_to_comp1'] + r['sim_to_comp2']) / 2 for r in pair_results]
    pair_sims_to_linear = [r['sim_to_linear_pred'] for r in pair_results]
    pair_suppressions = [r['suppression_ratio'] for r in pair_results]

    triplet_sims = [r['sim_to_linear_pred'] for r in triplet_results]
    triplet_suppressions = [r['suppression_ratio'] for r in triplet_results]

    summary = {
        'seed': seed,
        'n_individual_odors': n_odors,
        'n_2odor_mixtures': len(pairs),
        'n_3odor_mixtures': len(triplets),

        # Similarity: mixture code vs components
        'pair_sim_to_components_mean': float(np.mean(pair_sims_to_comp)),
        'pair_sim_to_components_std': float(np.std(pair_sims_to_comp)),
        'pair_sim_to_linear_pred_mean': float(np.mean(pair_sims_to_linear)),
        'pair_sim_to_linear_pred_std': float(np.std(pair_sims_to_linear)),

        # Suppression: mixture sparsity / component sparsity
        'pair_suppression_mean': float(np.mean(pair_suppressions)),
        'pair_suppression_std': float(np.std(pair_suppressions)),
        'triplet_suppression_mean': float(np.mean(triplet_suppressions)),
        'triplet_suppression_std': float(np.std(triplet_suppressions)),

        # Similarity: 3-odor mixtures
        'triplet_sim_to_linear_mean': float(np.mean(triplet_sims)),
        'triplet_sim_to_linear_std': float(np.std(triplet_sims)),

        # SVM classification
        'individual_svm_accuracy': indiv_svm_acc,
        'mixture_vs_individual_svm': binary_svm_acc,
        'inter_mixture_svm_accuracy': mix_svm_acc,
    }

    return summary, pair_results, triplet_results


# ============================================================================
# MAIN
# ============================================================================
def main():
    import argparse
    parser = argparse.ArgumentParser(description='C2: Odor mixture analysis')
    parser.add_argument('--seed', type=int, default=None,
                        help='Single seed (default: run all 5)')
    parser.add_argument('--n-seeds', type=int, default=None,
                        help='Number of seeds to use (default: all 5)')
    args = parser.parse_args()

    # Resolve data directory
    _parent = Path(__file__).resolve().parent.parent
    _data_candidates = [
        Path(__file__).resolve().parent / 'data',
        _parent / 'connectome_models' / 'data',
        _parent / 'relative_crawling' / 'connectome_models' / 'data',
    ]
    global DATA_DIR
    DATA_DIR = next((p for p in _data_candidates if (p / 'kreher2008').is_dir()), None)
    if DATA_DIR is None:
        raise FileNotFoundError('Cannot find connectome data.')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load OR responses and odor names
    kreher_dir = DATA_DIR / 'kreher2008'
    df = pd.read_csv(kreher_dir / 'orn_responses_normalized.csv', index_col=0)
    or_responses = torch.from_numpy(df.values).float()
    odor_names = df.index.tolist()
    n_odors = len(odor_names)
    print(f"Loaded {n_odors} odors x {or_responses.shape[1]} OR types")

    # Determine seeds
    if args.seed is not None:
        seeds = [args.seed]
    elif args.n_seeds is not None:
        seeds = SEEDS[:args.n_seeds]
    else:
        seeds = SEEDS

    all_summaries = []

    for seed in seeds:
        print(f"\n{'='*70}")
        print(f"SEED {seed}: MIXTURE ANALYSIS")
        print(f"{'='*70}")

        model = load_canonical_model(DATA_DIR, seed, n_odors)
        summary, pairs, triplets = analyze_mixtures(model, or_responses, odor_names, seed)
        del model

        all_summaries.append(summary)

        # Save per-seed results
        per_seed_path = OUTPUT_DIR / f'mixture_results_seed{seed}.json'
        with open(per_seed_path, 'w') as f:
            json.dump({
                'summary': summary,
                'pair_results': pairs,
                'triplet_results': triplets,
            }, f, indent=2)
        print(f"  Saved: {per_seed_path}")

        print(f"\n  --- Seed {seed} Summary ---")
        print(f"  2-odor mix sim to components:   {summary['pair_sim_to_components_mean']:.3f} ± {summary['pair_sim_to_components_std']:.3f}")
        print(f"  2-odor mix sim to linear pred:  {summary['pair_sim_to_linear_pred_mean']:.3f} ± {summary['pair_sim_to_linear_pred_std']:.3f}")
        print(f"  2-odor suppression ratio:       {summary['pair_suppression_mean']:.3f} ± {summary['pair_suppression_std']:.3f}")
        print(f"  3-odor suppression ratio:       {summary['triplet_suppression_mean']:.3f} ± {summary['triplet_suppression_std']:.3f}")
        print(f"  Individual SVM accuracy:        {summary['individual_svm_accuracy']:.1%}")
        print(f"  Mixture vs individual SVM:      {summary['mixture_vs_individual_svm']:.1%}")
        print(f"  Inter-mixture SVM:              {summary['inter_mixture_svm_accuracy']:.1%}")

    # ---- AGGREGATE SUMMARY ----
    if len(all_summaries) > 1:
        print(f"\n{'='*80}")
        print("C2 ODOR MIXTURE AGGREGATE SUMMARY")
        print(f"{'='*80}")

        metrics = [
            ('pair_sim_to_components_mean', 'Mix↔Component similarity'),
            ('pair_sim_to_linear_pred_mean', 'Mix↔Linear prediction sim'),
            ('pair_suppression_mean', '2-odor suppression ratio'),
            ('triplet_suppression_mean', '3-odor suppression ratio'),
            ('individual_svm_accuracy', 'Individual SVM accuracy'),
            ('mixture_vs_individual_svm', 'Mix vs Individual SVM'),
            ('inter_mixture_svm_accuracy', 'Inter-mixture SVM'),
        ]

        for key, label in metrics:
            vals = [s[key] for s in all_summaries]
            print(f"  {label:<35} {np.mean(vals):.3f} ± {np.std(vals):.3f}")

    # Save combined results
    combined_path = OUTPUT_DIR / 'mixture_summary.json'
    with open(combined_path, 'w') as f:
        json.dump({
            'n_seeds': len(all_summaries),
            'seeds': seeds,
            'per_seed': all_summaries,
            'aggregate': {
                key: {
                    'mean': float(np.mean([s[key] for s in all_summaries])),
                    'std': float(np.std([s[key] for s in all_summaries])),
                }
                for key, _ in [
                    ('pair_sim_to_components_mean', ''),
                    ('pair_sim_to_linear_pred_mean', ''),
                    ('pair_suppression_mean', ''),
                    ('triplet_suppression_mean', ''),
                    ('individual_svm_accuracy', ''),
                    ('mixture_vs_individual_svm', ''),
                    ('inter_mixture_svm_accuracy', ''),
                ]
            } if len(all_summaries) > 1 else {},
        }, f, indent=2)
    print(f"\nCombined summary saved: {combined_path}")


if __name__ == '__main__':
    main()
