"""
run_training_energy_only.py

C7: General Energy Constraint (Reviewer 3, Point 10).
Tests 9 variants of energy/sparsity loss across 5 seeds (42-46), producing
45 result JSONs used in the C7 and C4 notebook sections.

Each variant is one subprocess call. The 9 conditions are:
  canonical       --kc-sparsity              (CE + KC sparsity, canonical baseline)
  ce_only                                    (CE loss only, no sparsity/energy)
  energy_1        --energy-weight 1          (CE + all-type energy, weight=1)
  energy_conserv  --energy-weight 3          (CE + all-type energy, weight=3)
  energy_aggress  --energy-weight 15         (CE + all-type energy, weight=15)
  energy_50       --energy-weight 50         (CE + all-type energy, weight=50)
  kc_energy_conserv --kc-energy-only --energy-weight 3   (CE + KC-only energy)
  kc_energy_aggress --kc-energy-only --energy-weight 15  (CE + KC-only energy)
  kc_energy_50    --kc-energy-only --energy-weight 50    (CE + KC-only energy)

Energy loss = L1 penalty on mean firing rate. All-type: averaged equally across
ORN/LN/PN/KC. KC-only: applied only to KC firing rates.

Results saved to: results/energy_only_c7/
  results_{label}.json       (seed 42)
  results_{label}_seed{s}.json  (seeds 43-46)

Notebook sections: Section B — C7 (energy constraint table/figure),
                   Section B — C4 (ce_only variant for sparsity ablation comparison).

Usage (single run):
    python run_training_energy_only.py --train-teacher --seed 42
    python run_training_energy_only.py --label ce_only --seed 42
    python run_training_energy_only.py --kc-sparsity --label canonical --seed 42
    python run_training_energy_only.py --energy-weight 15 --label energy_aggress --seed 43
    python run_training_energy_only.py --kc-energy-only --energy-weight 3 --label kc_energy_conserv --seed 42
"""
import sys
from pathlib import Path
import argparse

# Path setup (identical to canonical run_training.py)
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
import pandas as pd

from spiking_connectome_models.model import SpikingConnectomeConstrainedModel
from spiking_connectome_models.layers import SpikingParams
from spiking_connectome_models.analysis.compute import (
    compute_per_pair_decorrelation as _compute_per_pair_decorrelation,
    compute_mean_sim_decorrelation, run_mancini_test as _run_mancini_test,
    run_concentration_invariance as _run_concentration_invariance,
    centroid_accuracy as _centroid_accuracy,
)
from connectome_models.model import ConnectomeConstrainedModel
from spiking_connectome_models.dataset import load_kreher2008_all_odors, create_dataloaders

# ============================================================================
# SPIKE ACCUMULATOR (forward hooks for ORN/LN spike capture)
# ============================================================================
class SpikeAccumulator:
    """Hook-based accumulator for ORN and LN spike counts."""
    def __init__(self):
        self.orn_spikes = None
        self.ln_spikes = None
        self._hooks = []

    def register(self, model):
        def orn_hook(module, input, output):
            spk = output[1]
            if self.orn_spikes is None:
                self.orn_spikes = spk
            else:
                self.orn_spikes = self.orn_spikes + spk

        def ln_hook(module, input, output):
            spk = output[1]
            if self.ln_spikes is None:
                self.ln_spikes = spk
            else:
                self.ln_spikes = self.ln_spikes + spk

        self._hooks.append(
            model.antennal_lobe.orn_neurons.register_forward_hook(orn_hook))
        self._hooks.append(
            model.antennal_lobe.ln_neurons.register_forward_hook(ln_hook))
        return self

    def reset(self):
        self.orn_spikes = None
        self.ln_spikes = None

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# ============================================================================
# CONFIGURATION
# ============================================================================
G_SOMA_MIN, G_SOMA_MAX_BIO = 1e-9, 20e-9
KCKC_LOG_MIN = np.log(1e-15)
LOG_STRENGTH_MAX = np.log(1e-8)

APL_BOOST = 4.0
TEACHER_EPOCHS = 300
STUDENT_EPOCHS = 300
BASE_LR = 1e-3
KC_VTH_LR = 0.01
LN_VTH_LR = 0.01
ORN_VTH_LR = 0.01
PN_VTH_LR = 0.01
GRAD_CLIP = 5.0

LN_VTH_INIT = -0.0475
LN_PN_SCALE = 1.2
ORN_PN_SCALE = 0.7
N_STEPS = 30

AL_LR = 0.2
NONAD_LR = 0.05
KC_LR = 4.0
KCKC_LR = 0.1
GSOMA_LR = 0.1
APL_TAU_LR = 0.05

NOISE_TYPE = 'multiplicative'
NOISE_STD = 0.3
NONAD_INIT = np.log(1e-13)

REALISTIC_PARAMS = SpikingParams(
    v_noise_std=1.0e-3,
    i_noise_std=15e-12,
    syn_noise_std=0.25,
    threshold_jitter_std=1.0e-3,
    orn_receptor_noise_std=0.10,
    circuit_noise_enabled=True,
)

CONCENTRATIONS = [0.03, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0]
HILL_EC50 = 1.0
HILL_N = 1
N_CONC_TRIALS = 10
ENERGY_RAMP_EPOCHS = 60
SEED = 42
DEVICE = torch.device('cpu')

_pkg_root = Path(__file__).resolve().parent
OUTPUT_DIR = _pkg_root / 'results' / 'energy_only_c7'
TEACHER_PATH = OUTPUT_DIR / 'teacher_seed42.pt'


# ============================================================================
# HELPERS
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
            if 'kc' in name:
                mult = KC_VTH_LR
            elif 'ln' in name:
                mult = LN_VTH_LR
            elif 'orn' in name:
                mult = ORN_VTH_LR
            elif 'pn' in name:
                mult = PN_VTH_LR
            else:
                mult = 0.01
        elif 'or_to_orn' in name or 'or_gains' in name:
            mult = 0.5
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


def run_analysis(model, or_responses, seed):
    """Run all analysis metrics. Model must be on CPU."""
    pp = _compute_per_pair_decorrelation(model, or_responses, 10, NOISE_STD)
    manc = _run_mancini_test(model)
    cent = _centroid_accuracy(model, or_responses, 20, NOISE_STD)
    conc_results, conc_tests = _run_concentration_invariance(
        model, or_responses, seed, CONCENTRATIONS, HILL_EC50, HILL_N,
        N_CONC_TRIALS, NOISE_STD)
    return pp, manc, cent, conc_results, conc_tests


# ============================================================================
# STUDENT INITIALIZATION
# ============================================================================
def initialize_student(data_dir, n_odors, teacher_state):
    """Create and initialize a student model from teacher state."""
    student = SpikingConnectomeConstrainedModel.from_data_dir(
        data_dir, n_odors=n_odors, n_or_types=21, target_sparsity=0.10,
        params=REALISTIC_PARAMS, include_nonad=True)
    student.n_steps_al = N_STEPS
    student.n_steps_kc = N_STEPS

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
        student.antennal_lobe.ln_pn_excit.log_strength.fill_(
            np.log(ln_pn_inhib_strength * 0.1))
        orn_pn_orig = student.antennal_lobe.orn_pn.log_strength.item()
        student.antennal_lobe.orn_pn.log_strength.fill_(orn_pn_orig + np.log(ORN_PN_SCALE))

        for attr in ['orn_ln_nonad', 'ln_pn_nonad', 'ln_pn_excit_nonad',
                      'ln_ln_nonad', 'pn_ln_nonad', 'ln_orn_nonad']:
            layer = getattr(student.antennal_lobe, attr, None)
            if layer is not None:
                layer.log_strength.fill_(NONAD_INIT)
        if student.kc_layer.pn_kc_nonad is not None:
            student.kc_layer.pn_kc_nonad.log_strength.fill_(NONAD_INIT)

    return student.to(DEVICE)


# ============================================================================
# TRAIN TEACHER (run once, save to disk)
# ============================================================================
def train_teacher(data_dir, n_odors, train_loader, test_loader):
    """Train rate-based teacher and save state dict."""
    print(f"--- Training teacher (seed {SEED}, {TEACHER_EPOCHS} epochs) ---")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
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
            print(f"  Teacher epoch {ep+1}: {c/t:.1%}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    teacher_state = {k: v.cpu().clone() for k, v in teacher.state_dict().items()}
    torch.save(teacher_state, TEACHER_PATH)
    print(f"  Teacher saved to {TEACHER_PATH}")
    return teacher_state


# ============================================================================
# TRAIN STUDENT (CE + optional energy, NO KC sparsity)
# ============================================================================
def train_student(energy_weight, label, data_dir, train_loader, test_loader,
                  n_odors, or_responses, teacher_state, kc_sparsity=False,
                  kc_energy_only=False):
    """Train student with CE + optional energy + optional KC sparsity."""
    loss_desc = "CE"
    if kc_sparsity:
        loss_desc += " + KC sparsity"
    if energy_weight > 0:
        energy_type = "KC-only energy" if kc_energy_only else "all-type energy"
        loss_desc += f" + {energy_weight} * {energy_type}"
    print(f"\n{'='*70}")
    print(f"C7: {label} (energy_weight={energy_weight}, seed {SEED})")
    print(f"Loss = {loss_desc}")
    print(f"{'='*70}")

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    student = initialize_student(data_dir, n_odors, teacher_state)
    accumulator = SpikeAccumulator().register(student)

    param_groups = get_param_groups(student)
    optimizer = torch.optim.Adam(param_groups)

    for epoch in range(STUDENT_EPOCHS):
        # Ramp weights over first 60 epochs (same schedule as canonical)
        progress = min(1.0, epoch / ENERGY_RAMP_EPOCHS)
        sp_w = progress * 15.0 if kc_sparsity else 0.0  # KC sparsity weight
        e_w = progress * energy_weight

        student.train()
        train_correct, train_total = 0, 0

        for bx, by in train_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            accumulator.reset()

            logits, info = student(bx, return_all=True)

            ce_loss = F.cross_entropy(logits, by)
            total_loss = ce_loss

            # KC sparsity loss (canonical, if enabled)
            kc_rates = info['kc_spikes'] / N_STEPS
            if kc_sparsity:
                sp_loss = (torch.sigmoid((kc_rates - 0.02) * 50).mean() - 0.05) ** 2
                total_loss = total_loss + sp_w * sp_loss

            # Energy constraint (if energy_weight > 0)
            if energy_weight > 0:
                if kc_energy_only:
                    # KC-only energy: penalize only KC mean firing rate
                    energy_loss = kc_rates.mean()
                else:
                    # All-type energy: penalize mean rate across all neuron types equally
                    orn_rate = accumulator.orn_spikes.mean() / N_STEPS
                    ln_rate = accumulator.ln_spikes.mean() / N_STEPS
                    pn_rate = info['pn_spikes'].mean() / N_STEPS
                    kc_rate = kc_rates.mean()
                    energy_loss = (orn_rate + ln_rate + pn_rate + kc_rate) / 4.0
                total_loss = total_loss + e_w * energy_loss

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), GRAD_CLIP)
            optimizer.step()
            clamp_biological(student)

            train_correct += (logits.argmax(-1) == by).sum().item()
            train_total += len(by)

        # Periodic evaluation (every 50 epochs)
        if (epoch + 1) % 50 == 0:
            student.eval()
            test_correct, test_total = 0, 0
            all_orn, all_ln, all_pn, all_kc = [], [], [], []
            with torch.no_grad():
                for bx, by in test_loader:
                    bx, by = bx.to(DEVICE), by.to(DEVICE)
                    accumulator.reset()
                    logits, info = student(bx, return_all=True)
                    test_correct += (logits.argmax(-1) == by).sum().item()
                    test_total += len(by)
                    all_orn.append((accumulator.orn_spikes > 0).float().mean().item())
                    all_ln.append((accumulator.ln_spikes > 0).float().mean().item())
                    all_pn.append((info['pn_spikes'] > 0).float().mean().item())
                    all_kc.append((info['kc_spikes'] > 0).float().mean().item())

            student.cpu()
            decorr = compute_mean_sim_decorrelation(student, or_responses)
            manc = _run_mancini_test(student)
            student.to(DEVICE)

            print(f"  Ep {epoch+1}: Test={test_correct/test_total:.1%}, "
                  f"KC={np.mean(all_kc):.1%}, ORN={np.mean(all_orn):.1%}, "
                  f"LN={np.mean(all_ln):.1%}, PN={np.mean(all_pn):.1%}, "
                  f"AL={decorr['al_decorr']:.1f}%, MB={decorr['mb_decorr']:.1f}%, "
                  f"Manc={manc['ratio']:.2f}")

    # ---- FINAL EVALUATION ----
    student.eval()
    print(f"\n  --- FINAL EVALUATION: {label} ---")

    # Per-type sparsity
    test_correct, test_total = 0, 0
    all_orn, all_ln, all_pn, all_kc = [], [], [], []
    with torch.no_grad():
        for bx, by in test_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            accumulator.reset()
            logits, info = student(bx, return_all=True)
            test_correct += (logits.argmax(-1) == by).sum().item()
            test_total += len(by)
            all_orn.append((accumulator.orn_spikes > 0).float().mean().item())
            all_ln.append((accumulator.ln_spikes > 0).float().mean().item())
            all_pn.append((info['pn_spikes'] > 0).float().mean().item())
            all_kc.append((info['kc_spikes'] > 0).float().mean().item())

    accumulator.remove()
    test_acc = test_correct / test_total
    per_type_sp = {
        'orn': float(np.mean(all_orn)), 'ln': float(np.mean(all_ln)),
        'pn': float(np.mean(all_pn)), 'kc': float(np.mean(all_kc)),
    }

    # Full analysis on CPU
    student.cpu()
    pp, manc, cent_acc, conc_results, conc_tests = run_analysis(
        student, or_responses, SEED)

    g_soma = np.exp(student.kc_layer.kc_neurons.log_g_soma.item()) * 1e9
    apl_gain = F.softplus(torch.tensor(student.kc_layer.apl.apl_gain.item())).item()

    print(f"  Accuracy:  linear={test_acc:.1%}, centroid={cent_acc:.1%}")
    print(f"  Sparsity:  ORN={per_type_sp['orn']:.1%}, LN={per_type_sp['ln']:.1%}, "
          f"PN={per_type_sp['pn']:.1%}, KC={per_type_sp['kc']:.1%}")
    print(f"  Decorr:    AL={pp['al_decorr']:.1f}%, MB={pp['mb_decorr']:.1f}%")
    print(f"  Mancini:   {manc['ratio']:.2f} ({'PASS' if manc['passes'] else 'FAIL'})")
    print(f"  Gain:      OR={conc_tests['or_range']:.2f}x, "
          f"PN={conc_tests['pn_range']:.2f}x, KC={conc_tests['kc_range']:.2f}x")
    print(f"  Flat KC:   {conc_tests['flat_kc_activity']}")
    print(f"  SubPN:     {conc_tests['sublinear_pn_gain']}")
    print(f"  RobClass:  {conc_tests['robust_classification']}")
    print(f"  OdorID:    {conc_tests['odor_identity_preservation']}")

    # Save model + results
    model_path = OUTPUT_DIR / f'model_{label}_seed{SEED}.pt'
    torch.save(student.state_dict(), model_path)

    results = {
        'label': label, 'energy_weight': energy_weight, 'seed': SEED,
        'loss_formulation': f'CE + {energy_weight} * energy (no KC sparsity)',
        'accuracy': test_acc, 'centroid_accuracy': cent_acc,
        'per_type_sparsity': per_type_sp,
        'decorrelation': {
            'al': pp['al_decorr'], 'mb': pp['mb_decorr'],
            'total': pp['total_decorr'],
        },
        'mancini': {'ratio': manc['ratio'], 'passes': manc['passes'],
                    'baseline': manc['baseline_spikes'], 'boosted': manc['boosted_spikes']},
        'concentration_invariance': {
            'or_range': conc_tests['or_range'], 'pn_range': conc_tests['pn_range'],
            'kc_range': conc_tests['kc_range'],
            'sublinear_pn': conc_tests['sublinear_pn_gain'],
            'flat_kc': conc_tests['flat_kc_activity'],
            'robust_class': conc_tests['robust_classification'],
            'odor_identity': conc_tests['odor_identity_preservation'],
        },
        'g_soma_nS': g_soma, 'apl_gain': apl_gain,
    }

    results_path = OUTPUT_DIR / f'results_{label}_seed{SEED}.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n  Model saved: {model_path}")
    print(f"  Results saved: {results_path}")
    print(f"  DONE: {label}")


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description='C7: CE + energy (no KC sparsity)')
    parser.add_argument('--train-teacher', action='store_true',
                        help='Train and save teacher only')
    parser.add_argument('--energy-weight', type=float, default=0.0,
                        help='Energy constraint weight (0 = CE only)')
    parser.add_argument('--kc-sparsity', action='store_true',
                        help='Include canonical KC sparsity loss (CE + KC sp)')
    parser.add_argument('--kc-energy-only', action='store_true',
                        help='Energy penalty on KC firing rate only (not all types)')
    parser.add_argument('--label', type=str, default='ce_only',
                        help='Label for this run')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    args = parser.parse_args()

    global SEED, TEACHER_PATH
    SEED = args.seed
    TEACHER_PATH = OUTPUT_DIR / f'teacher_seed{SEED}.pt'

    # Find data
    _parent = Path(__file__).resolve().parent.parent
    _data_candidates = [
        Path(__file__).resolve().parent / 'data',
        _parent / 'connectome_models' / 'data',
        _parent / 'relative_crawling' / 'connectome_models' / 'data',
    ]
    data_dir = next((p for p in _data_candidates if (p / 'kreher2008').is_dir()), None)
    if data_dir is None:
        raise FileNotFoundError('Cannot find connectome data (kreher2008/).')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    train_ds, test_ds, odor_names = load_kreher2008_all_odors(
        data_dir, train_repeats=10, test_repeats=5,
        noise_std=NOISE_STD, noise_type=NOISE_TYPE)
    train_loader, test_loader = create_dataloaders(
        train_ds, test_ds, batch_size=16)
    n_odors = len(odor_names)
    df = pd.read_csv(data_dir / "kreher2008/orn_responses_normalized.csv", index_col=0)
    or_responses = torch.from_numpy(df.values).float()

    if args.train_teacher:
        train_teacher(data_dir, n_odors, train_loader, test_loader)
        return

    # Load teacher
    if not TEACHER_PATH.exists():
        print("Teacher not found, training...")
        teacher_state = train_teacher(data_dir, n_odors, train_loader, test_loader)
    else:
        print(f"Loading teacher from {TEACHER_PATH}")
        teacher_state = torch.load(TEACHER_PATH, weights_only=False)

    train_student(args.energy_weight, args.label, data_dir,
                  train_loader, test_loader, n_odors, or_responses,
                  teacher_state, kc_sparsity=args.kc_sparsity,
                  kc_energy_only=args.kc_energy_only)


if __name__ == "__main__":
    main()
