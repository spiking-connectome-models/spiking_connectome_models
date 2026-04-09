"""
run_posthoc_ablation.py

Post-hoc ablation on already-trained canonical models.
Loads trained model weights, zeros out specific connections, and re-evaluates.
NO retraining — tests whether the trained circuit relies on these connections.

C1 post-hoc: Reviewer 2 ablation studies
  (i)  Gap junctions removed at eval time
  (ii) APL inhibition disabled at eval time

Results saved to: results/posthoc_ablations/
  results_no_gap_posthoc.json
  results_no_apl_posthoc.json

Notebook section: Section B — C1 (post-hoc ablations table).

Usage:
    python run_posthoc_ablation.py
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
import torch
import torch.nn.functional as F
import numpy as np

from spiking_connectome_models.model import SpikingConnectomeConstrainedModel
from spiking_connectome_models.layers import SpikingParams
from spiking_connectome_models.analysis.compute import (
    compute_per_pair_decorrelation as _compute_per_pair_decorrelation,
    run_mancini_test as _run_mancini_test,
    centroid_accuracy as _centroid_accuracy,
    run_concentration_invariance as _run_concentration_invariance,
)
from connectome_models.model import ConnectomeConstrainedModel


# ============================================================================
# SPIKE ACCUMULATOR (forward hooks for ORN/LN spikes)
# ============================================================================
class SpikeAccumulator:
    def __init__(self, model):
        self.orn_spikes = None
        self.ln_spikes = None
        self._hooks = []
        def orn_hook(module, input, output):
            spk = output[1]
            self.orn_spikes = spk if self.orn_spikes is None else self.orn_spikes + spk
        def ln_hook(module, input, output):
            spk = output[1]
            self.ln_spikes = spk if self.ln_spikes is None else self.ln_spikes + spk
        self._hooks.append(model.antennal_lobe.orn_neurons.register_forward_hook(orn_hook))
        self._hooks.append(model.antennal_lobe.ln_neurons.register_forward_hook(ln_hook))

    def reset(self):
        self.orn_spikes = None
        self.ln_spikes = None

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# ============================================================================
# CONFIG
# ============================================================================
DATA_DIR = Path.home() / 'Desktop' / 'relative_crawling' / 'connectome_models' / 'data'
CANONICAL_DIR = Path(__file__).parent / 'results' / 'all_connections_nonad_canonical'
OUTPUT_DIR = Path(__file__).parent / 'results' / 'posthoc_ablations'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 43, 44, 45, 46]
N_STEPS = 30
NOISE_STD = 0.3

# Concentration invariance params
CONCENTRATIONS = [0.03, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0]
HILL_EC50 = 0.5
HILL_N = 1.5
N_CONC_TRIALS = 15

REALISTIC_PARAMS = SpikingParams(
    v_noise_std=1.0e-3, i_noise_std=15e-12, syn_noise_std=0.25,
    threshold_jitter_std=1.0e-3, orn_receptor_noise_std=0.10,
    circuit_noise_enabled=True,
)


# ============================================================================
# LOAD OR RESPONSES
# ============================================================================
def load_or_responses():
    """Load Kreher 2008 OR response data."""
    kreher_dir = DATA_DIR / 'kreher2008'
    pt_path = kreher_dir / 'orn_responses_normalized.pt'
    csv_path = kreher_dir / 'orn_responses_normalized.csv'

    if pt_path.exists():
        or_responses = torch.load(pt_path, weights_only=True)
    else:
        import pandas as pd
        df = pd.read_csv(csv_path, index_col=0)
        or_responses = torch.tensor(df.values, dtype=torch.float32)

    n_odors, n_or_types = or_responses.shape
    print(f"Kreher 2008 data: {n_odors} odors x {n_or_types} OR types")
    return or_responses, n_odors, n_or_types


# ============================================================================
# BUILD FRESH MODEL AND LOAD WEIGHTS
# ============================================================================
def load_canonical_model(seed, n_odors):
    """Build a fresh model and load canonical trained weights."""
    model = SpikingConnectomeConstrainedModel.from_data_dir(
        DATA_DIR, n_odors=n_odors, n_or_types=21, target_sparsity=0.10,
        params=REALISTIC_PARAMS, include_nonad=True)
    model.n_steps_al = N_STEPS
    model.n_steps_kc = N_STEPS

    # Load trained canonical weights
    model_path = CANONICAL_DIR / f'model_seed{seed}.pt'
    state = torch.load(model_path, weights_only=False, map_location='cpu')
    model.load_state_dict(state)
    model.eval()
    return model


# ============================================================================
# ABLATION FUNCTIONS (post-hoc, no retraining)
# ============================================================================
def ablate_gap_junctions(model):
    """Zero out all gap junction conductances.  [C1 post-hoc (i)]"""
    with torch.no_grad():
        if hasattr(model.antennal_lobe, 'log_g_gap_ln'):
            model.antennal_lobe.log_g_gap_ln.fill_(np.log(1e-30))
        if hasattr(model.antennal_lobe, 'log_g_gap_pn'):
            model.antennal_lobe.log_g_gap_pn.fill_(np.log(1e-30))
        if hasattr(model.antennal_lobe, 'log_g_gap_eln_pn'):
            model.antennal_lobe.log_g_gap_eln_pn.fill_(np.log(1e-30))
    print("  Post-hoc: gap junctions zeroed (all conductances -> ~0)")


def ablate_apl(model):
    """Disable APL by setting gain to -100 (softplus -> ~0).  [C1 post-hoc (ii)]"""
    with torch.no_grad():
        model.kc_layer.apl.apl_gain.fill_(-100.0)
    print(f"  Post-hoc: APL disabled (gain=-100, softplus={F.softplus(torch.tensor(-100.0)).item():.2e})")


# ============================================================================
# EVALUATION (same metrics as training scripts)
# ============================================================================
def evaluate_model(model, or_responses, seed, label, skip_mancini=False):
    """Run full evaluation suite on a model."""
    n_odors = or_responses.shape[0]
    # Generate test data
    noisy = or_responses.unsqueeze(1) + torch.randn(n_odors, 5, or_responses.shape[1]) * NOISE_STD
    noisy = noisy.clamp(min=0)
    test_X = noisy.reshape(-1, or_responses.shape[1])
    test_y = torch.arange(n_odors).unsqueeze(1).expand(-1, 5).reshape(-1)

    # Forward pass with spike accumulator
    accumulator = SpikeAccumulator(model)
    model.eval()
    with torch.no_grad():
        all_orn, all_ln, all_pn, all_kc = [], [], [], []
        tc, tt = 0, 0
        for i in range(0, len(test_X), 28):
            bx = test_X[i:i+28]
            by = test_y[i:i+28]
            accumulator.reset()
            logits, info = model(bx, return_all=True)
            tc += (logits.argmax(-1) == by).sum().item()
            tt += len(by)
            all_orn.append((accumulator.orn_spikes > 0).float().mean().item())
            all_ln.append((accumulator.ln_spikes > 0).float().mean().item())
            all_pn.append((info['pn_spikes'] > 0).float().mean().item())
            all_kc.append((info['kc_spikes'] > 0).float().mean().item())

    accumulator.remove()
    test_acc = tc / tt
    per_type = {
        'orn': float(np.mean(all_orn)), 'ln': float(np.mean(all_ln)),
        'pn': float(np.mean(all_pn)), 'kc': float(np.mean(all_kc)),
    }

    # Decorrelation
    pp = _compute_per_pair_decorrelation(model, or_responses, 10, NOISE_STD)

    # Mancini (skip for no-APL)
    if skip_mancini:
        manc = {'ratio': float('nan'), 'passes': False,
                'baseline_spikes': float('nan'), 'boosted_spikes': float('nan'),
                'skipped': True}
    else:
        manc = _run_mancini_test(model)

    # Centroid accuracy
    cent_acc = _centroid_accuracy(model, or_responses, 20, NOISE_STD)

    # Concentration invariance
    conc_results, conc_tests = _run_concentration_invariance(
        model, or_responses, seed, CONCENTRATIONS, HILL_EC50, HILL_N,
        N_CONC_TRIALS, NOISE_STD)

    g_soma = np.exp(model.kc_layer.kc_neurons.log_g_soma.item()) * 1e9
    apl_gain = F.softplus(torch.tensor(model.kc_layer.apl.apl_gain.item())).item()

    # Print summary
    manc_str = 'SKIP' if skip_mancini else f"{manc['ratio']:.2f}"
    print(f"  Acc:     linear={test_acc:.1%}, centroid={cent_acc:.1%}")
    print(f"  Sparse:  ORN={per_type['orn']:.1%}, LN={per_type['ln']:.1%}, "
          f"PN={per_type['pn']:.1%}, KC={per_type['kc']:.1%}")
    print(f"  Decorr:  AL={pp['al_decorr']:.1f}%, MB={pp['mb_decorr']:.1f}%")
    print(f"  Mancini: {manc_str}")
    print(f"  FlatKC:  {conc_tests['flat_kc_activity']}, "
          f"KC range: {conc_tests['kc_range']:.2f}x")

    return {
        'label': label, 'seed': seed,
        'accuracy': test_acc, 'centroid_accuracy': cent_acc,
        'per_type_sparsity': per_type,
        'decorrelation': {
            'al': pp['al_decorr'], 'mb': pp['mb_decorr'], 'total': pp['total_decorr'],
        },
        'mancini': {'ratio': manc['ratio'], 'passes': manc.get('passes', False),
                    'baseline': manc.get('baseline_spikes', 0),
                    'boosted': manc.get('boosted_spikes', 0)},
        'concentration_invariance': conc_tests,
        'g_soma_nS': g_soma, 'apl_gain': apl_gain,
    }


# ============================================================================
# MAIN
# ============================================================================
def main():
    or_responses, n_odors, n_or_types = load_or_responses()

    all_results = {'no_gap_posthoc': [], 'no_apl_posthoc': []}

    for seed in SEEDS:
        print(f"\n{'='*70}")
        print(f"SEED {seed}")
        print(f"{'='*70}")

        # (i) Gap junctions removed post-hoc
        print(f"\n--- Post-hoc: Remove gap junctions (seed {seed}) ---")
        model = load_canonical_model(seed, n_odors)
        ablate_gap_junctions(model)
        result = evaluate_model(model, or_responses, seed,
                                f'no_gap_posthoc_s{seed}')
        all_results['no_gap_posthoc'].append(result)
        del model

        # (ii) APL disabled post-hoc
        print(f"\n--- Post-hoc: Disable APL (seed {seed}) ---")
        model = load_canonical_model(seed, n_odors)
        ablate_apl(model)
        result = evaluate_model(model, or_responses, seed,
                                f'no_apl_posthoc_s{seed}', skip_mancini=True)
        all_results['no_apl_posthoc'].append(result)
        del model

    # Save results
    for condition, results in all_results.items():
        out_path = OUTPUT_DIR / f'results_{condition}.json'
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nSaved: {out_path}")

    # Summary table
    print(f"\n{'='*90}")
    print("POST-HOC ABLATION SUMMARY (5 seeds)")
    print(f"{'='*90}")
    print(f"{'Condition':<25} {'Accuracy':>12} {'Centroid':>12} {'KC%':>7} "
          f"{'AL dec':>8} {'MB dec':>8} {'Mancini':>8} {'FlatKC':>7}")
    print('-' * 90)

    for condition, label in [('no_gap_posthoc', 'No Gap (post-hoc)'),
                              ('no_apl_posthoc', 'No APL (post-hoc)')]:
        data = all_results[condition]
        accs = [r['accuracy'] for r in data]
        cents = [r['centroid_accuracy'] for r in data]
        kcs = [r['per_type_sparsity']['kc'] for r in data]
        als = [r['decorrelation']['al'] for r in data]
        mbs = [r['decorrelation']['mb'] for r in data]
        flat = sum(1 for r in data
                   if r['concentration_invariance'].get('flat_kc_activity',
                      r['concentration_invariance'].get('flat_kc', False)))

        manc_vals = [r['mancini']['ratio'] for r in data
                     if not np.isnan(r['mancini']['ratio'])]
        manc_str = f"{np.mean(manc_vals):.2f}" if manc_vals else "SKIP"

        print(f"{label:<25} {np.mean(accs):>5.1%}+/-{np.std(accs):.1%} "
              f"{np.mean(cents):>5.1%}+/-{np.std(cents):.1%} "
              f"{np.mean(kcs):>6.1%} {np.mean(als):>7.1f} {np.mean(mbs):>7.1f} "
              f"{manc_str:>8} {flat}/{len(data)}")


if __name__ == '__main__':
    main()
