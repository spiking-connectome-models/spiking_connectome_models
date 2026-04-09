"""
run_task_complexity.py

C6: Task Complexity / KC Threshold Scaling (Reviewer 3, Point 8).
Retrain models with different numbers of odors (7, 14, 28) and a larger
synthetic odor set (56) to test whether KC thresholds scale with task demands.

Key question: ~47% of KC thresholds hit the upper boundary (-30 mV) in the
canonical 28-odor model. Does this fraction scale with the number of odors?
If so, the connectome has capacity for more complex discrimination tasks.

Each condition: 2-phase training (teacher 300ep → student 300ep), 3 seeds (42-44).
After training, extract per-KC v_th distributions and compute boundary stats.

Results saved to: results/task_complexity_c6/
  results_c6_n{7,14,56}_s{42-44}.json
  canonical_thresholds.json   (n=28 baseline, seeds 42-44)
  task_complexity_summary.json

Notebook section: Section B — C6 (task complexity table and KC bound fraction figure).
"""
import sys
from pathlib import Path
import argparse

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
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from spiking_connectome_models.model import SpikingConnectomeConstrainedModel
from spiking_connectome_models.layers import SpikingParams
from spiking_connectome_models.analysis.compute import (
    compute_per_pair_decorrelation as _compute_per_pair_decorrelation,
    run_mancini_test as _run_mancini_test,
    centroid_accuracy as _centroid_accuracy,
)
from connectome_models.model import ConnectomeConstrainedModel

# ============================================================================
# CONFIG  (mirrors run_ablation.py — C6 uses same training pipeline)
# ============================================================================
DATA_DIR = None  # resolved at runtime
OUTPUT_DIR = Path(__file__).parent / 'results' / 'task_complexity_c6'
TEACHER_DIR = OUTPUT_DIR / 'teachers'

TEACHER_EPOCHS = 300
STUDENT_EPOCHS = 300
N_STEPS = 30
MAX_SP_WEIGHT = 15.0
ENERGY_RAMP_EPOCHS = 60
BASE_LR = 1e-3
GRAD_CLIP = 5.0

APL_BOOST = 4.0
LN_VTH_INIT = -0.0475
LN_PN_SCALE = 1.2
ORN_PN_SCALE = 0.7
NONAD_INIT = np.log(1e-13)

G_SOMA_MIN, G_SOMA_MAX_BIO = 1e-9, 20e-9
KCKC_LOG_MIN = np.log(1e-15)
LOG_STRENGTH_MAX = np.log(1e-8)

NOISE_STD = 0.3
NOISE_TYPE = 'multiplicative'


def _to_native(obj):
    """Recursively convert numpy/torch types to JSON-serializable Python builtins.

    C6 fix: json.dump(default=...) is bypassed by the C-extension for types
    like np.bool_ (post-NumPy 1.24). Recursively converting the whole dict
    before dump is the reliable alternative.
    """
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(v) for v in obj]
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if hasattr(obj, 'item'):  # torch scalars
        return obj.item()
    return obj

# Learning rate multipliers (same as run_ablation.py)
AL_LR = 0.2
NONAD_LR = 0.05
KC_LR = 4.0
KCKC_LR = 0.1
GSOMA_LR = 0.1
APL_TAU_LR = 0.05
KC_VTH_LR = 0.01
LN_VTH_LR = 0.01
ORN_VTH_LR = 0.01
PN_VTH_LR = 0.01

# V_TH bounds (from layers.py — used for threshold analysis)
V_TH_MIN = -0.055   # -55 mV
V_TH_MAX = -0.030   # -30 mV

REALISTIC_PARAMS = SpikingParams(
    v_noise_std=1.0e-3, i_noise_std=15e-12, syn_noise_std=0.25,
    threshold_jitter_std=1.0e-3, orn_receptor_noise_std=0.10,
    circuit_noise_enabled=True,
)

# C6 conditions: number of odors to train with
ODOR_COUNTS = [7, 14, 28, 56]
SEEDS = [42, 43, 44]


# ============================================================================
# SPIKE ACCUMULATOR (same as run_ablation.py)
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
# DATASET: Repeated odor presentations with noise
# ============================================================================
class SubsetOdorDataset(Dataset):
    """OR responses for a subset of odors, with noise augmentation.

    C6: allows training with fewer (7, 14) or more (synthetic 56) odors
    while keeping the same training infrastructure.
    """
    def __init__(self, or_responses, repeats_per_odor=10,
                 noise_std=0.3, noise_type='multiplicative'):
        self.or_responses = or_responses  # (n_odors, 21)
        self.n_odors = or_responses.shape[0]
        self.repeats = repeats_per_odor
        self.noise_std = noise_std
        self.noise_type = noise_type

    def __len__(self):
        return self.n_odors * self.repeats

    def __getitem__(self, idx):
        odor_idx = idx // self.repeats
        pattern = self.or_responses[odor_idx]
        noise = torch.randn_like(pattern) * self.noise_std
        if self.noise_type == 'multiplicative':
            noisy = pattern * (1.0 + noise)
        else:
            noisy = pattern + noise
        noisy = noisy.clamp(min=0)
        return noisy, odor_idx


# ============================================================================
# SYNTHETIC ODOR GENERATION
# ============================================================================
def generate_synthetic_odors(real_or_responses, n_synthetic, seed):
    """Generate synthetic OR response patterns for larger odor sets.

    C6: Creates plausible odor patterns by sampling from the empirical
    distribution of each OR type (per-receptor mean/std from Kreher 2008),
    then ensuring patterns are distinct from each other and from real odors.
    """
    rng = np.random.RandomState(seed + 5000)
    real_np = real_or_responses.numpy()
    n_real, n_or = real_np.shape

    # Per-OR-type statistics from Kreher 2008
    or_means = real_np.mean(axis=0)
    or_stds = real_np.std(axis=0)
    or_stds = np.maximum(or_stds, 0.05)  # floor to avoid degenerate types

    # Sample from per-receptor Gaussian, clamp to [0, 1]
    synthetic = np.zeros((n_synthetic, n_or))
    for i in range(n_synthetic):
        pattern = rng.normal(or_means, or_stds)
        # Add structured sparsity: each odor activates 3-8 OR types strongly
        n_active = rng.randint(3, 9)
        active_mask = np.zeros(n_or, dtype=bool)
        active_mask[rng.choice(n_or, n_active, replace=False)] = True
        # Suppress non-active receptors (reduce to background level)
        pattern[~active_mask] *= 0.2
        pattern = np.clip(pattern, 0, 1)
        synthetic[i] = pattern

    return torch.from_numpy(synthetic).float()


# ============================================================================
# HELPERS (mirrored from run_ablation.py)
# ============================================================================
def clamp_biological(model):
    model.clamp_to_biological_bounds()
    with torch.no_grad():
        model.kc_layer.kc_kc_aa.log_strength.clamp_(KCKC_LOG_MIN, LOG_STRENGTH_MAX)
        model.kc_layer.kc_neurons.log_g_soma.clamp_(np.log(G_SOMA_MIN), np.log(G_SOMA_MAX_BIO))
        if model.kc_layer.kc_kc_ad is not None:
            model.kc_layer.kc_kc_ad.log_strength.clamp_(KCKC_LOG_MIN, LOG_STRENGTH_MAX)
        if model.kc_layer.pn_kc_nonad is not None:
            model.kc_layer.pn_kc_nonad.log_strength.clamp_(KCKC_LOG_MIN, LOG_STRENGTH_MAX)


def get_param_groups(model):
    """Build optimizer param groups (same LR schedule as run_ablation.py)."""
    param_groups = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'log_tau_rec' in name or 'logit_U' in name:
            mult = GSOMA_LR
        elif 'nonad' in name:
            mult = NONAD_LR
        elif 'kc_kc_aa' in name or 'kc_kc_ad' in name:
            mult = KCKC_LR
        elif 'kc_kc_dd_gain' in name or 'kc_kc_da_gain' in name:
            mult = KCKC_LR
        elif 'log_g_soma' in name:
            mult = GSOMA_LR
        elif 'log_g_gap' in name:
            mult = AL_LR
        elif 'ln_pn_excit' in name:
            mult = AL_LR
        elif 'ln_ln' in name or 'pn_ln' in name:
            mult = AL_LR
        elif 'v_th' in name:
            if 'kc' in name: mult = KC_VTH_LR
            elif 'ln' in name: mult = LN_VTH_LR
            elif 'orn' in name: mult = ORN_VTH_LR
            elif 'pn' in name: mult = PN_VTH_LR
            else: mult = 0.01
        elif 'or_to_orn' in name or 'or_gains' in name:
            mult = 0.5
        elif 'apl_gain' in name:
            mult = KC_LR
        elif 'orn_neurons' in name or 'ln_neurons' in name or 'pn_neurons' in name or 'antennal_lobe' in name:
            mult = AL_LR
        elif 'log_tau_apl' in name:
            mult = APL_TAU_LR
        elif 'kc_layer' in name or 'kc_neurons' in name or 'apl' in name:
            mult = KC_LR
        else:
            mult = 1.0
        param_groups.append({'params': [param], 'lr': BASE_LR * mult})
    return param_groups


# ============================================================================
# KC THRESHOLD ANALYSIS  (C6-specific: the key measurement)
# ============================================================================
def analyze_kc_thresholds(model):
    """Extract KC threshold distribution and boundary statistics.

    C6: Measures how many KC thresholds hit V_TH_MAX (-30 mV), which
    indicates that gradient descent pushed them as excitable as possible.
    """
    v_th = model.kc_layer.kc_neurons.v_th.detach().cpu().numpy() * 1e3  # convert to mV
    n_kc = len(v_th)

    # Threshold at boundary = within 0.5 mV of V_TH_MAX
    at_upper = np.sum(v_th >= (V_TH_MAX * 1e3 - 0.5))  # -30.5 mV
    at_lower = np.sum(v_th <= (V_TH_MIN * 1e3 + 0.5))  # -54.5 mV

    return {
        'n_kc': int(n_kc),
        'v_th_mean_mV': float(np.mean(v_th)),
        'v_th_std_mV': float(np.std(v_th)),
        'v_th_min_mV': float(np.min(v_th)),
        'v_th_max_mV': float(np.max(v_th)),
        'v_th_median_mV': float(np.median(v_th)),
        'frac_at_upper_bound': float(at_upper / n_kc),
        'n_at_upper_bound': int(at_upper),
        'frac_at_lower_bound': float(at_lower / n_kc),
        'n_at_lower_bound': int(at_lower),
        'v_th_values_mV': v_th.tolist(),  # full distribution for plotting
    }


# ============================================================================
# TRAINING
# ============================================================================
def train_teacher(seed, data_dir, n_odors, train_loader, test_loader):
    """Train rate-based teacher model (Phase 1)."""
    TEACHER_DIR.mkdir(parents=True, exist_ok=True)
    teacher_path = TEACHER_DIR / f'teacher_n{n_odors}_seed{seed}.pt'

    if teacher_path.exists():
        print(f"  Teacher n_odors={n_odors} seed={seed} exists, loading...")
        return torch.load(teacher_path, weights_only=False)

    print(f"--- Training teacher (n_odors={n_odors}, seed={seed}, {TEACHER_EPOCHS} ep) ---")
    torch.manual_seed(seed)
    np.random.seed(seed)
    teacher = ConnectomeConstrainedModel.from_data_dir(
        data_dir, n_odors=n_odors, n_or_types=21, target_sparsity=0.10)
    opt = torch.optim.Adam(teacher.parameters(), lr=1e-2)
    for ep in range(TEACHER_EPOCHS):
        teacher.train()
        for bx, by in train_loader:
            opt.zero_grad()
            loss, _ = teacher.compute_loss(bx, by, sparsity_weight=2.0)
            loss.backward()
            opt.step()
        if (ep + 1) % 100 == 0:
            teacher.eval()
            c, t = 0, 0
            with torch.no_grad():
                for bx, by in test_loader:
                    c += (teacher(bx).argmax(-1) == by).sum().item()
                    t += len(by)
            print(f"  Teacher ep {ep+1}: {c/t:.1%}")

    teacher_state = {k: v.cpu().clone() for k, v in teacher.state_dict().items()}
    torch.save(teacher_state, teacher_path)
    print(f"  Teacher saved: {teacher_path}")
    return teacher_state


def train_student(seed, data_dir, n_odors, or_responses,
                  train_loader, test_loader, teacher_state):
    """Train spiking student model (Phase 2) and analyze KC thresholds.

    C6: If a trained model file already exists, skips training and runs
    evaluation only. This avoids re-running 300 epochs when the model was
    saved but the JSON write failed (e.g. the np.bool_ serialization bug).
    """
    label = f'c6_n{n_odors}_s{seed}'
    model_path = OUTPUT_DIR / f'model_{label}.pt'
    results_path = OUTPUT_DIR / f'results_{label}.json'

    print(f"\n{'='*70}")
    print(f"STUDENT TRAINING: {label}")
    print(f"{'='*70}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Build student model (always needed, even for eval-only path)
    student = SpikingConnectomeConstrainedModel.from_data_dir(
        data_dir, n_odors=n_odors, n_or_types=21, target_sparsity=0.10,
        params=REALISTIC_PARAMS, include_nonad=True)
    student.n_steps_al = N_STEPS
    student.n_steps_kc = N_STEPS

    if model_path.exists():
        # C6: trained model exists — load it and skip the 300-epoch training loop
        print(f"  Trained model found for {label}, loading (skipping training)...")
        state = torch.load(model_path, weights_only=False, map_location='cpu')
        student.load_state_dict(state)
    else:
        # Full training path: initialize from teacher then run training loop
        teacher = ConnectomeConstrainedModel.from_data_dir(
            data_dir, n_odors=n_odors, n_or_types=21, target_sparsity=0.10)
        teacher.load_state_dict(teacher_state)

        with torch.no_grad():
            for name, param in student.named_parameters():
                if 'v_th' in name:
                    param.fill_(LN_VTH_INIT if 'ln' in name else -0.0425)

            student.decoder.weight.copy_(teacher.decoder.weight)
            student.decoder.bias.copy_(teacher.decoder.bias)
            student.or_to_orn.or_gains.copy_(teacher.or_to_orn.or_gains)
            student.kc_layer.apl.apl_gain.data = teacher.kc_layer.apl.apl_gain.data.clone() * APL_BOOST

            student.kc_layer.kc_neurons.v_th.data += 0.005
            student.kc_layer.kc_kc_aa.log_strength.fill_(np.log(1e-11))
            if student.kc_layer.kc_kc_ad is not None:
                student.kc_layer.kc_kc_ad.log_strength.fill_(np.log(1e-13))
            student.kc_layer.kc_neurons.log_g_soma.fill_(np.log(10e-9))

            if student.antennal_lobe.ln_ln is not None:
                student.antennal_lobe.ln_ln.log_strength.fill_(np.log(1e-13))
            if student.antennal_lobe.pn_ln is not None:
                student.antennal_lobe.pn_ln.log_strength.fill_(np.log(1e-13))
            if student.antennal_lobe.ln_orn is not None:
                student.antennal_lobe.ln_orn.log_strength.fill_(np.log(1e-13))
            ln_pn_orig = student.antennal_lobe.ln_pn.log_strength.item()
            student.antennal_lobe.ln_pn.log_strength.fill_(ln_pn_orig + np.log(LN_PN_SCALE))
            ln_pn_inhib_strength = np.exp(student.antennal_lobe.ln_pn.log_strength.item())
            student.antennal_lobe.ln_pn_excit.log_strength.fill_(np.log(ln_pn_inhib_strength * 0.1))
            orn_pn_orig = student.antennal_lobe.orn_pn.log_strength.item()
            student.antennal_lobe.orn_pn.log_strength.fill_(orn_pn_orig + np.log(ORN_PN_SCALE))

            for attr in ['orn_ln_nonad', 'ln_pn_nonad', 'ln_pn_excit_nonad',
                          'ln_ln_nonad', 'pn_ln_nonad', 'ln_orn_nonad']:
                layer = getattr(student.antennal_lobe, attr, None)
                if layer is not None:
                    layer.log_strength.fill_(NONAD_INIT)
            if student.kc_layer.pn_kc_nonad is not None:
                student.kc_layer.pn_kc_nonad.log_strength.fill_(NONAD_INIT)
        del teacher

        accumulator = SpikeAccumulator().register(student)
        param_groups = get_param_groups(student)
        optimizer = torch.optim.Adam(param_groups)

        # ---- TRAINING LOOP ----
        for epoch in range(STUDENT_EPOCHS):
            progress = min(1.0, epoch / ENERGY_RAMP_EPOCHS)
            sp_w = progress * MAX_SP_WEIGHT

            student.train()
            for bx, by in train_loader:
                optimizer.zero_grad()
                accumulator.reset()
                logits, info = student(bx, return_all=True)
                ce_loss = F.cross_entropy(logits, by)

                # KC sparsity loss (same as canonical)
                kc_rates = info['kc_spikes'] / N_STEPS
                sp_loss = (torch.sigmoid((kc_rates - 0.02) * 50).mean() - 0.05) ** 2
                total_loss = ce_loss + sp_w * sp_loss

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), GRAD_CLIP)
                optimizer.step()
                clamp_biological(student)

            if (epoch + 1) % 50 == 0:
                student.eval()
                tc, tt = 0, 0
                all_kc = []
                with torch.no_grad():
                    for bx, by in test_loader:
                        accumulator.reset()
                        logits, info = student(bx, return_all=True)
                        tc += (logits.argmax(-1) == by).sum().item()
                        tt += len(by)
                        all_kc.append((info['kc_spikes'] > 0).float().mean().item())
                # C6: track threshold stats during training
                th = analyze_kc_thresholds(student)
                print(f"  Ep {epoch+1}: Acc={tc/tt:.1%}, KC={np.mean(all_kc):.1%}, "
                      f"vth_mean={th['v_th_mean_mV']:.1f}mV, "
                      f"at_upper={th['frac_at_upper_bound']:.0%}")

        accumulator.remove()
        # Save model after training completes (not needed in eval-only path)
        torch.save(student.state_dict(), model_path)

    # ---- FINAL EVALUATION ---- (runs for both trained and loaded models)
    student.eval()

    # KC threshold analysis (C6 key measurement)
    threshold_stats = analyze_kc_thresholds(student)

    # Classification accuracy
    tc, tt = 0, 0
    all_kc = []
    with torch.no_grad():
        for bx, by in test_loader:
            logits, info = student(bx, return_all=True)
            tc += (logits.argmax(-1) == by).sum().item()
            tt += len(by)
            all_kc.append((info['kc_spikes'] > 0).float().mean().item())
    test_acc = tc / tt
    kc_sparsity = float(np.mean(all_kc))

    # Decorrelation
    pp = _compute_per_pair_decorrelation(student, or_responses, 10, NOISE_STD)

    # Centroid accuracy
    cent_acc = _centroid_accuracy(student, or_responses, 20, NOISE_STD)

    # Mancini test
    manc = _run_mancini_test(student)

    print(f"\n  --- FINAL: {label} ---")
    print(f"  Acc:  linear={test_acc:.1%}, centroid={cent_acc:.1%}")
    print(f"  KC:   sparsity={kc_sparsity:.1%}")
    print(f"  Dec:  AL={pp['al_decorr']:.1f}%, MB={pp['mb_decorr']:.1f}%")
    print(f"  Manc: {manc['ratio']:.2f}")
    print(f"  v_th: mean={threshold_stats['v_th_mean_mV']:.1f}mV, "
          f"at_upper={threshold_stats['frac_at_upper_bound']:.0%} "
          f"({threshold_stats['n_at_upper_bound']}/{threshold_stats['n_kc']})")

    results = {
        'label': label,
        'n_odors': n_odors,
        'seed': seed,
        'accuracy': test_acc,
        'centroid_accuracy': cent_acc,
        'kc_sparsity': kc_sparsity,
        'decorrelation': {
            'al': pp['al_decorr'], 'mb': pp['mb_decorr'], 'total': pp['total_decorr'],
        },
        'mancini': {'ratio': float(manc['ratio']), 'passes': bool(manc['passes'])},
        'threshold_stats': threshold_stats,
    }

    # C6 fix: use _to_native() to pre-convert the whole dict before json.dump.
    # json's C-extension bypasses default= for np.bool_ (post-NumPy 1.24),
    # so recursive pre-conversion is the only reliable approach.
    with open(results_path, 'w') as f:
        json.dump(_to_native(results), f, indent=2)
    print(f"  Saved: {results_path}")
    return results


# ============================================================================
# CANONICAL THRESHOLD EXTRACTION (for the 28-odor baseline)
# ============================================================================
def extract_canonical_thresholds(data_dir, seeds):
    """Extract KC thresholds from already-trained canonical models."""
    canonical_dir = Path(__file__).parent / 'results' / 'all_connections_nonad_canonical'
    results = []
    for seed in seeds:
        model_path = canonical_dir / f'model_seed{seed}.pt'
        if not model_path.exists():
            print(f"  Canonical model seed {seed} not found, skipping")
            continue
        model = SpikingConnectomeConstrainedModel.from_data_dir(
            data_dir, n_odors=28, n_or_types=21, target_sparsity=0.10,
            params=REALISTIC_PARAMS, include_nonad=True)
        state = torch.load(model_path, weights_only=False, map_location='cpu')
        model.load_state_dict(state)
        model.eval()
        th = analyze_kc_thresholds(model)
        th['seed'] = seed
        results.append(th)
        del model
        print(f"  Canonical seed {seed}: vth_mean={th['v_th_mean_mV']:.1f}mV, "
              f"at_upper={th['frac_at_upper_bound']:.0%} "
              f"({th['n_at_upper_bound']}/{th['n_kc']})")
    return results


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description='C6: Task complexity / KC threshold scaling')
    parser.add_argument('--n-odors', type=int, default=None,
                        help='Train a single odor count (7, 14, 28, or 56)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Single seed (default: run all 3)')
    parser.add_argument('--canonical-only', action='store_true',
                        help='Only extract thresholds from canonical 28-odor models')
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

    # Load Kreher 2008 OR responses (all 28 odors)
    kreher_dir = DATA_DIR / 'kreher2008'
    pt_path = kreher_dir / 'orn_responses_normalized.pt'
    if pt_path.exists():
        all_or_responses = torch.load(pt_path, weights_only=True)
    else:
        df = pd.read_csv(kreher_dir / 'orn_responses_normalized.csv', index_col=0)
        all_or_responses = torch.from_numpy(df.values).float()

    # Extract canonical model thresholds as baseline
    print("="*70)
    print("EXTRACTING CANONICAL (28-odor) KC THRESHOLDS")
    print("="*70)
    canonical_th = extract_canonical_thresholds(DATA_DIR, SEEDS)
    if canonical_th:
        canon_path = OUTPUT_DIR / 'canonical_thresholds.json'
        with open(canon_path, 'w') as f:
            json.dump(canonical_th, f, indent=2)
        avg_upper = np.mean([t['frac_at_upper_bound'] for t in canonical_th])
        print(f"\n  Canonical avg at upper bound: {avg_upper:.1%}")

    if args.canonical_only:
        return

    # Determine which conditions to run
    odor_counts = [args.n_odors] if args.n_odors else ODOR_COUNTS
    seeds = [args.seed] if args.seed else SEEDS

    all_results = []

    for n_odors in odor_counts:
        print(f"\n{'='*70}")
        print(f"CONDITION: {n_odors} ODORS")
        print(f"{'='*70}")

        # Prepare OR responses for this condition
        if n_odors <= 28:
            # C6: Subset of real Kreher odors (deterministic selection per seed)
            for seed in seeds:
                rng = np.random.RandomState(seed + 2000)
                if n_odors < 28:
                    indices = sorted(rng.choice(28, n_odors, replace=False))
                else:
                    indices = list(range(28))
                or_subset = all_or_responses[indices]
                print(f"\n  Seed {seed}: using odor indices {indices}")

                # Create dataloaders for this subset
                train_ds = SubsetOdorDataset(or_subset, repeats_per_odor=10,
                                             noise_std=NOISE_STD, noise_type=NOISE_TYPE)
                test_ds = SubsetOdorDataset(or_subset, repeats_per_odor=5,
                                            noise_std=NOISE_STD, noise_type=NOISE_TYPE)
                train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
                test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

                # Train teacher
                teacher_state = train_teacher(seed, DATA_DIR, n_odors, train_loader, test_loader)

                # Train student and analyze thresholds
                result = train_student(seed, DATA_DIR, n_odors, or_subset,
                                       train_loader, test_loader, teacher_state)
                all_results.append(result)

        else:
            # C6: Synthetic odor set (larger than 28)
            for seed in seeds:
                n_synthetic = n_odors - 28
                synthetic = generate_synthetic_odors(all_or_responses, n_synthetic, seed)
                or_combined = torch.cat([all_or_responses, synthetic], dim=0)
                print(f"\n  Seed {seed}: 28 real + {n_synthetic} synthetic = {n_odors} odors")

                train_ds = SubsetOdorDataset(or_combined, repeats_per_odor=10,
                                             noise_std=NOISE_STD, noise_type=NOISE_TYPE)
                test_ds = SubsetOdorDataset(or_combined, repeats_per_odor=5,
                                            noise_std=NOISE_STD, noise_type=NOISE_TYPE)
                train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
                test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

                teacher_state = train_teacher(seed, DATA_DIR, n_odors, train_loader, test_loader)
                result = train_student(seed, DATA_DIR, n_odors, or_combined,
                                       train_loader, test_loader, teacher_state)
                all_results.append(result)

    # ---- SUMMARY ----
    print(f"\n{'='*80}")
    print("C6 TASK COMPLEXITY SUMMARY: KC THRESHOLD SCALING")
    print(f"{'='*80}")
    print(f"\n  {'n_odors':>8} {'seed':>5} {'Acc':>6} {'Cent':>6} {'KC%':>6} "
          f"{'MB dec':>7} {'vth_mean':>9} {'at_upper':>9}")

    for r in all_results:
        th = r['threshold_stats']
        print(f"  {r['n_odors']:>8} {r['seed']:>5} {r['accuracy']:>5.1%} "
              f"{r['centroid_accuracy']:>5.1%} {r['kc_sparsity']:>5.1%} "
              f"{r['decorrelation']['mb']:>+6.1f}% {th['v_th_mean_mV']:>8.1f}mV "
              f"{th['frac_at_upper_bound']:>8.0%}")

    # Aggregate by n_odors
    print(f"\n  --- Averages ---")
    for n in sorted(set(r['n_odors'] for r in all_results)):
        subset = [r for r in all_results if r['n_odors'] == n]
        avg_acc = np.mean([r['centroid_accuracy'] for r in subset])
        avg_kc = np.mean([r['kc_sparsity'] for r in subset])
        avg_mb = np.mean([r['decorrelation']['mb'] for r in subset])
        avg_upper = np.mean([r['threshold_stats']['frac_at_upper_bound'] for r in subset])
        avg_vth = np.mean([r['threshold_stats']['v_th_mean_mV'] for r in subset])
        print(f"  n={n:>3}: centroid={avg_acc:.1%}, KC={avg_kc:.1%}, "
              f"MB_dec={avg_mb:+.1f}%, vth={avg_vth:.1f}mV, at_upper={avg_upper:.0%}")

    # Save combined results (without v_th_values_mV for compactness)
    summary_results = []
    for r in all_results:
        r_copy = json.loads(json.dumps(r))
        r_copy['threshold_stats'].pop('v_th_values_mV', None)
        summary_results.append(r_copy)
    summary_path = OUTPUT_DIR / 'task_complexity_summary.json'
    with open(summary_path, 'w') as f:
        json.dump({'results': summary_results, 'canonical_thresholds': canonical_th}, f, indent=2)
    print(f"\n  Summary saved: {summary_path}")


if __name__ == '__main__':
    main()
