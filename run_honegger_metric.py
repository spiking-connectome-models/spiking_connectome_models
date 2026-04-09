"""Compute Honegger-style per-KC sub-additivity for 2-odor and 3-odor mixtures.

Replicates the analysis from Honegger et al. (2011, J Neurosci): for each KC,
compare its mixture response to the sum of its individual component responses.
A KC is "sub-additive" if mixture_response < component1_response + component2_response.
Honegger reported 73% of KCs sub-additive for binary blends.

Runs all 5 seeds in parallel. Results saved to results/odor_mixtures_c2/honegger_metric.json.

Usage:
    python run_honegger_metric.py
"""
import sys
from pathlib import Path

# Add parent directory so the package can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import numpy as np
import json
from itertools import combinations

import ccn_s_connectome_revisions.model as model_module

# --- Configuration ---
BASE = Path(__file__).resolve().parent
MODEL_DIR = BASE / 'results' / 'all_connections_nonad_canonical'
DATA_DIR = BASE / 'data'
N_TRIALS = 20       # Noisy trials per stimulus (matches paper)
NOISE_STD = 0.3     # Multiplicative noise CV (matches training)
N_TRIPLETS = 50     # Random 3-odor triplets per seed


def run_seed(seed):
    """Run Honegger-style per-KC sub-additivity analysis for one seed.

    For each 2-odor pair and 3-odor triplet:
    1. Get mean KC firing rate vector for each component odor
    2. Get mean KC firing rate vector for the mixture
    3. Count KCs where mixture < sum(components) — these are sub-additive

    Returns dict with fraction sub-additive for pairs and triplets.
    """
    print(f"[Seed {seed}] Loading model...")
    mdl, or_responses, odor_names = model_module.load_spiking_model_and_data(DATA_DIR)
    state = torch.load(MODEL_DIR / f'model_seed{seed}.pt', map_location='cpu', weights_only=False)
    mdl.load_state_dict(state, strict=False)
    mdl.eval()

    n_odors = len(or_responses) - 1  # skip sfr
    odor_or = or_responses[1:]  # (27, 21)

    rng = np.random.default_rng(seed)

    def get_kc_rates(or_pattern):
        """Get mean KC firing rate vector (144-dim) for an OR pattern over N_TRIALS noisy trials."""
        kc_all = []
        for _ in range(N_TRIALS):
            noise = torch.from_numpy(
                rng.normal(0, NOISE_STD, or_pattern.shape).astype(np.float32))
            x = (or_pattern * (1.0 + noise)).clamp(0)
            with torch.no_grad():
                _, info = mdl(x.unsqueeze(0), return_all=True)
            kc_all.append(info['kc_spikes'].float().squeeze().numpy())
        return np.mean(kc_all, axis=0)  # (144,)

    # Get KC rates for each individual odor
    print(f"[Seed {seed}] Running {n_odors} individual odors...")
    individual_kc = {}
    for i in range(n_odors):
        individual_kc[i] = get_kc_rates(odor_or[i])

    # === 2-ODOR PAIRS ===
    pairs = list(combinations(range(n_odors), 2))
    print(f"[Seed {seed}] Running {len(pairs)} 2-odor pairs...")

    pair_sub = []   # Count of sub-additive KCs per pair
    pair_resp = []  # Count of responsive KCs per pair
    for idx, (i, j) in enumerate(pairs):
        # Mixture OR = element-wise sum of components, clamped to [0,1]
        mix_or = (odor_or[i] + odor_or[j]).clamp(0, 1)
        mix_kc = get_kc_rates(mix_or)
        # Linear prediction = sum of individual KC responses
        linear_pred = individual_kc[i] + individual_kc[j]
        # Only count KCs that responded to at least one condition
        responsive = (linear_pred > 0.01) | (mix_kc > 0.01)
        n_resp = responsive.sum()
        if n_resp > 0:
            # Honegger metric: fraction where mixture < sum(components)
            pair_sub.append((mix_kc[responsive] < linear_pred[responsive]).sum())
            pair_resp.append(n_resp)
        if (idx + 1) % 100 == 0:
            print(f"[Seed {seed}] pairs: {idx+1}/{len(pairs)}")

    pair_frac = np.sum(pair_sub) / np.sum(pair_resp)

    # === 3-ODOR TRIPLETS ===
    all_triplets = list(combinations(range(n_odors), 3))
    rng_trip = np.random.default_rng(seed + 1000)
    triplet_indices = rng_trip.choice(len(all_triplets), size=min(N_TRIPLETS, len(all_triplets)), replace=False)
    triplets = [all_triplets[idx] for idx in triplet_indices]
    print(f"[Seed {seed}] Running {len(triplets)} 3-odor triplets...")

    trip_sub = []
    trip_resp = []
    for idx, (i, j, k) in enumerate(triplets):
        mix_or = (odor_or[i] + odor_or[j] + odor_or[k]).clamp(0, 1)
        mix_kc = get_kc_rates(mix_or)
        linear_pred = individual_kc[i] + individual_kc[j] + individual_kc[k]
        responsive = (linear_pred > 0.01) | (mix_kc > 0.01)
        n_resp = responsive.sum()
        if n_resp > 0:
            trip_sub.append((mix_kc[responsive] < linear_pred[responsive]).sum())
            trip_resp.append(n_resp)

    trip_frac = np.sum(trip_sub) / np.sum(trip_resp)

    result = {
        'seed': seed,
        'pair_frac_sub_additive': float(pair_frac),
        'pair_total_sub': int(np.sum(pair_sub)),
        'pair_total_resp': int(np.sum(pair_resp)),
        'triplet_frac_sub_additive': float(trip_frac),
        'triplet_total_sub': int(np.sum(trip_sub)),
        'triplet_total_resp': int(np.sum(trip_resp)),
        'n_pairs': len(pairs),
        'n_triplets': len(triplets),
    }

    print(f"[Seed {seed}] DONE: pairs={pair_frac:.1%}, triplets={trip_frac:.1%}")
    return result


if __name__ == '__main__':
    import multiprocessing as mp

    seeds = [42, 43, 44, 45, 46]
    with mp.Pool(5) as pool:
        results = pool.map(run_seed, seeds)

    pair_fracs = [r['pair_frac_sub_additive'] for r in results]
    trip_fracs = [r['triplet_frac_sub_additive'] for r in results]

    print("\n" + "=" * 60)
    print("AGGREGATE (5 seeds)")
    print("=" * 60)
    print(f"2-odor: {np.mean(pair_fracs):.1%} ± {np.std(pair_fracs):.1%} sub-additive KCs")
    print(f"3-odor: {np.mean(trip_fracs):.1%} ± {np.std(trip_fracs):.1%} sub-additive KCs")
    print(f"Honegger (2011): 73% (2-odor only)")

    output = {
        'per_seed': results,
        'aggregate': {
            'pair_frac_mean': float(np.mean(pair_fracs)),
            'pair_frac_std': float(np.std(pair_fracs)),
            'triplet_frac_mean': float(np.mean(trip_fracs)),
            'triplet_frac_std': float(np.std(trip_fracs)),
        }
    }
    out_path = BASE / 'results' / 'odor_mixtures_c2' / 'honegger_metric.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")
