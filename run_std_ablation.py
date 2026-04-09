"""
STD (Short-Term Depression) Ablation Study — seeds 42-46.

Two conditions:
  1. no_std_train   — Train from scratch with STD disabled at every timestep.
                      Tests whether STD is needed during learning.
  2. posthoc_std_off — Load canonical trained models (with STD) and evaluate
                      with STD disabled at test time only.
                      Tests whether STD matters for inference/read-out.

Results saved to:  results/std_ablation/
  no_std_train/    model_seed{42-46}.pt  +  results.json
  posthoc_std_off/ results.json          (no new .pt files — uses canonical weights)

Metrics match the canonical training script:
  accuracy, centroid_accuracy, sparsity,
  per-pair AL/MB/total decorrelation, Mancini, concentration invariance,
  g_soma, gap conductances, LN→PN split.

Notebook section: Section B — STD Ablation (comparison table and bar chart).

Usage:
    python -m spiking_connectome_models.run_std_ablation
    python -m spiking_connectome_models.run_std_ablation --condition no_std_train
    python -m spiking_connectome_models.run_std_ablation --condition posthoc_std_off
"""

import sys, argparse
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
sys.stdout.reconfigure(line_buffering=True)

import json
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')

from spiking_connectome_models.model import SpikingConnectomeConstrainedModel
from spiking_connectome_models.layers import SpikingParams
from spiking_connectome_models.analysis.compute import (
    compute_per_pair_decorrelation as _compute_per_pair_decorrelation,
    compute_mean_sim_decorrelation,
    run_mancini_test as _run_mancini_test,
    run_concentration_invariance as _run_concentration_invariance,
    centroid_accuracy as _centroid_accuracy,
)
from connectome_models.model import ConnectomeConstrainedModel
from spiking_connectome_models.dataset import load_kreher2008_all_odors, create_dataloaders

# ============================================================================
# CONFIGURATION — identical to canonical training script
# ============================================================================
G_SOMA_MIN, G_SOMA_MAX_BIO = 1e-9, 20e-9
KCKC_LOG_MIN = np.log(1e-15)
LOG_STRENGTH_MAX = np.log(1e-8)

APL_BOOST = 4.0
TEACHER_EPOCHS = 300
STUDENT_EPOCHS = 300
MAX_SP_WEIGHT = 15.0
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

REALISTIC_PARAMS = SpikingParams(
    v_noise_std=1.0e-3,
    i_noise_std=15e-12,
    syn_noise_std=0.25,
    threshold_jitter_std=1.0e-3,
    orn_receptor_noise_std=0.10,
    circuit_noise_enabled=True,
)

SEEDS = [42, 43, 44, 45, 46]
NONAD_INIT = np.log(1e-13)
CONCENTRATIONS = [0.03, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0]
HILL_EC50 = 1.0
HILL_N = 1
N_CONC_TRIALS = 10

STD_ABLATION_DIR = Path(__file__).resolve().parent / 'results' / 'std_ablation'
CANONICAL_DIR = Path(__file__).resolve().parent / 'results' / 'all_connections_nonad_canonical'


# ============================================================================
# HELPERS (mirror canonical script)
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


def compute_per_pair_decorrelation(model, or_responses, n_trials=10, disable_std=False):
    """Wrapper around canonical decorrelation, with optional STD disable."""
    if disable_std:
        _patch_model_disable_std(model)
    r = _compute_per_pair_decorrelation(model, or_responses, n_trials, NOISE_STD)
    if disable_std:
        _unpatch_model(model)
    return {
        'kc_or': r['kc_or_ratio'], 'kc_pn': r['kc_pn_ratio'], 'pn_or': r['pn_or_ratio'],
        'total_decorr_pct': r['total_decorr'], 'mb_decorr_pct': r['mb_decorr'],
        'al_decorr_pct': r['al_decorr'],
    }


def run_mancini(model, disable_std=False, carbachol=1e-10, apl_inject=0.7):
    """Mancini 2023 test. STD disable applies to baseline-activity pass; the
    Mancini protocol itself uses disable_apl / apl_inject_current flags that
    are layered on top."""
    if disable_std:
        _patch_model_disable_std(model)
    result = _run_mancini_test(model, carbachol, apl_inject)
    if disable_std:
        _unpatch_model(model)
    return result['ratio']


def run_concentration_invariance(model, or_responses, seed, disable_std=False):
    if disable_std:
        _patch_model_disable_std(model)
    r = _run_concentration_invariance(
        model, or_responses, seed, CONCENTRATIONS, HILL_EC50, HILL_N, N_CONC_TRIALS, NOISE_STD)
    if disable_std:
        _unpatch_model(model)
    return r


def centroid_accuracy(model, or_responses, n_trials=20, disable_std=False):
    if disable_std:
        _patch_model_disable_std(model)
    r = _centroid_accuracy(model, or_responses, n_trials, NOISE_STD)
    if disable_std:
        _unpatch_model(model)
    return r


def evaluate_model(model, test_loader, disable_std=False):
    model.eval()
    correct, total, sparsities = 0, 0, []
    with torch.no_grad():
        for bx, by in test_loader:
            logits, info = model(bx, return_all=True, disable_std=disable_std)
            correct += (logits.argmax(-1) == by).sum().item()
            total += len(by)
            sparsities.append(info['sparsity'])
    return correct / total, np.mean(sparsities)


def compute_mean_sim_decorr(model, or_responses, disable_std=False):
    if disable_std:
        _patch_model_disable_std(model)
    r = compute_mean_sim_decorrelation(model, or_responses)
    if disable_std:
        _unpatch_model(model)
    return r


# ============================================================================
# MODEL PATCHING HELPERS
# Temporarily wrap model.forward so all downstream analysis calls (decorr,
# Mancini, conc-invariance) automatically use disable_std=True.
# ============================================================================

def _patch_model_disable_std(model):
    """Monkeypatch model.forward to always pass disable_std=True."""
    if hasattr(model, '_orig_forward_std'):
        return  # already patched
    orig = model.forward

    def _patched(*args, **kwargs):
        kwargs.setdefault('disable_std', True)
        return orig(*args, **kwargs)

    model._orig_forward_std = orig
    model.forward = _patched


def _unpatch_model(model):
    """Restore original forward after patching."""
    if hasattr(model, '_orig_forward_std'):
        model.forward = model._orig_forward_std
        del model._orig_forward_std


# ============================================================================
# COLLECT METRICS
# ============================================================================

def collect_metrics(seed, student, test_loader, or_responses, disable_std):
    """Run full evaluation suite and return results dict."""
    student.eval()
    ds = disable_std

    test_acc, sparsity = evaluate_model(student, test_loader, disable_std=ds)
    cent_acc = centroid_accuracy(student, or_responses, disable_std=ds)
    pp_decorr = compute_per_pair_decorrelation(student, or_responses, disable_std=ds)
    mancini = run_mancini(student, disable_std=ds)
    mancini_pass = 1.5 <= mancini <= 2.5
    conc_results, conc_tests = run_concentration_invariance(student, or_responses, seed, disable_std=ds)

    g_gap_ln = float(np.exp(student.antennal_lobe.log_g_gap_ln.item())) \
        if student.antennal_lobe.log_g_gap_ln is not None else None
    g_gap_pn = float(np.exp(student.antennal_lobe.log_g_gap_pn.item()))
    g_gap_eln = float(np.exp(student.antennal_lobe.log_g_gap_eln_pn.item()))
    ln_pn_inhib_str = float(np.exp(student.antennal_lobe.ln_pn.log_strength.item()))
    ln_pn_excit_str = float(np.exp(student.antennal_lobe.ln_pn_excit.log_strength.item()))
    g_soma = np.exp(student.kc_layer.kc_neurons.log_g_soma.item()) * 1e9
    apl_gain = F.softplus(torch.tensor(student.kc_layer.apl.apl_gain.item())).item()
    kc_apl_strength = float(np.exp(student.kc_layer.apl.kc_apl_log_strength.item()))

    nonad_strengths = {}
    for name, layer in [
        ('orn_ln_nonad', student.antennal_lobe.orn_ln_nonad),
        ('ln_pn_nonad', student.antennal_lobe.ln_pn_nonad),
        ('ln_pn_excit_nonad', student.antennal_lobe.ln_pn_excit_nonad),
        ('ln_ln_nonad', student.antennal_lobe.ln_ln_nonad),
        ('pn_ln_nonad', student.antennal_lobe.pn_ln_nonad),
        ('ln_orn_nonad', student.antennal_lobe.ln_orn_nonad),
        ('pn_kc_nonad', student.kc_layer.pn_kc_nonad),
    ]:
        if layer is not None:
            nonad_strengths[name] = float(np.exp(layer.log_strength.item()))

    print(f"    Acc: {test_acc:.1%}, Centroid: {cent_acc:.1%}, Sp: {sparsity:.1%}")
    print(f"    AL: {pp_decorr['al_decorr_pct']:.1f}%, MB: {pp_decorr['mb_decorr_pct']:.1f}%, "
          f"Total: {pp_decorr['total_decorr_pct']:.1f}%")
    print(f"    Mancini: {mancini:.2f} ({'PASS' if mancini_pass else 'FAIL'})")
    print(f"    Conc: SubPN={'P' if conc_tests['sublinear_pn_gain'] else 'F'}, "
          f"FlatKC={'P' if conc_tests['flat_kc_activity'] else 'F'}, "
          f"RobClass={'P' if conc_tests['robust_classification'] else 'F'}, "
          f"Identity={conc_tests['odor_identity_preservation']}")
    print(f"    g_soma: {g_soma:.1f} nS, APL gain: {apl_gain:.2f}, KC->APL: {kc_apl_strength:.2e}")

    return {
        'seed': seed,
        'accuracy': test_acc,
        'centroid_accuracy': cent_acc,
        'sparsity': sparsity,
        'per_pair_decorrelation': pp_decorr,
        'mancini': mancini,
        'mancini_pass': mancini_pass,
        'concentration_invariance': {
            'per_concentration': conc_results,
            'predictions': {k: v for k, v in conc_tests.items()
                            if isinstance(v, (bool, float, int, str))},
        },
        'g_soma_nS': g_soma,
        'apl_gain_effective': apl_gain,
        'kc_apl_strength': kc_apl_strength,
        'gap_junction_conductances': {'ln_ln': g_gap_ln, 'pn_pn': g_gap_pn, 'eln_pn': g_gap_eln},
        'ln_pn_split': {'inhibitory_strength': ln_pn_inhib_str, 'excitatory_strength': ln_pn_excit_str},
        'nonad_strengths': nonad_strengths,
    }


# ============================================================================
# CONDITION 1: NO-STD TRAINING
# ============================================================================

def run_no_std_training(data_dir, train_loader, test_loader, n_odors, or_responses, output_dir):
    """Train from scratch with STD disabled throughout."""
    output_dir.mkdir(parents=True, exist_ok=True)
    all_results = []

    for seed in SEEDS:
        print(f"\n{'='*70}")
        print(f"NO-STD TRAINING (seed {seed})")
        print(f"{'='*70}")

        torch.manual_seed(seed)
        np.random.seed(seed)

        history = {'train_acc': [], 'test_acc': [], 'sparsity': [],
                   'al_decorr': [], 'mb_decorr': [], 'mancini': [], 'g_soma': []}

        # --- Teacher (rate-based, unchanged) ---
        print(f"\n  Training teacher ({TEACHER_EPOCHS} epochs)...")
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
                print(f"    Teacher epoch {ep+1}: {c/t:.1%}")

        # --- Student (STD disabled for all forward calls) ---
        print(f"\n  Setting up student (STD DISABLED, all connections + non-AD)...")
        student = SpikingConnectomeConstrainedModel.from_data_dir(
            data_dir, n_odors=n_odors, n_or_types=21, target_sparsity=0.10,
            params=REALISTIC_PARAMS, include_nonad=True)
        student.n_steps_al = N_STEPS
        student.n_steps_kc = N_STEPS

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
            for name, attr in [
                ('orn_ln_nonad', student.antennal_lobe.orn_ln_nonad),
                ('ln_pn_nonad', student.antennal_lobe.ln_pn_nonad),
                ('ln_pn_excit_nonad', student.antennal_lobe.ln_pn_excit_nonad),
                ('ln_ln_nonad', student.antennal_lobe.ln_ln_nonad),
                ('pn_ln_nonad', student.antennal_lobe.pn_ln_nonad),
                ('ln_orn_nonad', student.antennal_lobe.ln_orn_nonad),
                ('pn_kc_nonad', student.kc_layer.pn_kc_nonad),
            ]:
                if attr is not None:
                    attr.log_strength.fill_(NONAD_INIT)

        print(f"\n  Training student ({STUDENT_EPOCHS} epochs, STD DISABLED)...")
        param_groups = get_param_groups(student)
        optimizer = torch.optim.Adam(param_groups)
        ep300_state = None

        for epoch in range(STUDENT_EPOCHS):
            progress = min(1.0, epoch / 60)
            sp_w = progress * MAX_SP_WEIGHT

            student.train()
            train_correct, train_total = 0, 0
            for bx, by in train_loader:
                optimizer.zero_grad()
                # *** STD DISABLED during training ***
                logits, info = student(bx, return_all=True, disable_std=True)
                ce_loss = F.cross_entropy(logits, by)
                kc_rates = info['kc_spikes'] / N_STEPS
                sp_loss = (torch.sigmoid((kc_rates - 0.02) * 50).mean() - 0.05) ** 2
                (ce_loss + sp_w * sp_loss).backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), GRAD_CLIP)
                optimizer.step()
                clamp_biological(student)
                train_correct += (logits.argmax(-1) == by).sum().item()
                train_total += len(by)

            if (epoch + 1) % 50 == 0:
                student.eval()
                test_acc, sparsity = evaluate_model(student, test_loader, disable_std=True)
                decorr = compute_mean_sim_decorr(student, or_responses, disable_std=True)
                mancini = run_mancini(student, disable_std=True)
                g_soma = np.exp(student.kc_layer.kc_neurons.log_g_soma.item()) * 1e9

                history['train_acc'].append(train_correct / train_total)
                history['test_acc'].append(test_acc)
                history['sparsity'].append(sparsity)
                history['al_decorr'].append(decorr['al_decorr'])
                history['mb_decorr'].append(decorr['mb_decorr'])
                history['mancini'].append(mancini)
                history['g_soma'].append(g_soma)

                print(f"  Ep {epoch+1}: Train={train_correct/train_total:.1%}, Test={test_acc:.1%}, "
                      f"Sp={sparsity:.1%}, AL={decorr['al_decorr']:.1f}%, MB={decorr['mb_decorr']:.1f}%, "
                      f"Manc={mancini:.2f}, g={g_soma:.1f}nS")

            if (epoch + 1) == 300:
                ep300_state = {k: v.clone() for k, v in student.state_dict().items()}

        # Evaluate at epoch 300 with STD disabled
        student.load_state_dict(ep300_state)
        student.eval()
        print(f"\n  --- Seed {seed} Epoch 300 Evaluation (no_std_train) ---")
        results = collect_metrics(seed, student, test_loader, or_responses, disable_std=True)
        results['history'] = history

        torch.save(student.state_dict(), output_dir / f'model_seed{seed}.pt')
        all_results.append(results)

    _save_and_print_summary('no_std_train', all_results, output_dir)
    return all_results


# ============================================================================
# CONDITION 2: POST-HOC STD REMOVAL
# ============================================================================

def run_posthoc_std_off(data_dir, test_loader, n_odors, or_responses, output_dir):
    """Load canonical models; evaluate with STD disabled."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if not CANONICAL_DIR.exists():
        raise FileNotFoundError(
            f"Canonical model directory not found: {CANONICAL_DIR}\n"
            "Run run_training.py first to generate canonical models.")

    all_results = []

    for seed in SEEDS:
        print(f"\n{'='*70}")
        print(f"POST-HOC STD OFF — loading canonical seed {seed}")
        print(f"{'='*70}")

        model = SpikingConnectomeConstrainedModel.from_data_dir(
            data_dir, n_odors=n_odors, n_or_types=21, target_sparsity=0.10,
            params=REALISTIC_PARAMS, include_nonad=True)
        model.load_state_dict(torch.load(
            CANONICAL_DIR / f'model_seed{seed}.pt', weights_only=True))
        model.n_steps_al = N_STEPS
        model.n_steps_kc = N_STEPS
        model.eval()

        print(f"  Loaded. Evaluating with STD DISABLED...")
        results = collect_metrics(seed, model, test_loader, or_responses, disable_std=True)
        all_results.append(results)

    _save_and_print_summary('posthoc_std_off', all_results, output_dir)
    return all_results


# ============================================================================
# SUMMARY + SAVE
# ============================================================================

def _save_and_print_summary(condition, all_results, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / 'results.json'
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved → {out_path}")

    accs   = [r['accuracy'] for r in all_results]
    cents  = [r['centroid_accuracy'] for r in all_results]
    sps    = [r['sparsity'] for r in all_results]
    al_ds  = [r['per_pair_decorrelation']['al_decorr_pct'] for r in all_results]
    mb_ds  = [r['per_pair_decorrelation']['mb_decorr_pct'] for r in all_results]
    tot_ds = [r['per_pair_decorrelation']['total_decorr_pct'] for r in all_results]
    mancs  = [r['mancini'] for r in all_results]
    gsomas = [r['g_soma_nS'] for r in all_results]

    print(f"\n{'='*80}")
    print(f"STD ABLATION — {condition.upper()} — SUMMARY ({len(SEEDS)} seeds)")
    print(f"{'='*80}")
    print(f"{'Seed':<6} {'Acc':<8} {'Cent':<8} {'Sp':<8} {'AL%':<8} {'MB%':<8} {'Tot%':<8} {'Manc':<8} {'g_soma':<8}")
    print("-" * 72)
    for r in all_results:
        pp = r['per_pair_decorrelation']
        m_ok = "P" if r['mancini_pass'] else "F"
        print(f"{r['seed']:<6} {r['accuracy']:<8.1%} {r['centroid_accuracy']:<8.1%} "
              f"{r['sparsity']:<8.1%} "
              f"{pp['al_decorr_pct']:<8.1f} {pp['mb_decorr_pct']:<8.1f} {pp['total_decorr_pct']:<8.1f} "
              f"{r['mancini']:.2f}{m_ok:<3} {r['g_soma_nS']:<8.1f}")
    print("-" * 72)
    print(f"{'Mean':<6} {np.mean(accs):<8.1%} {np.mean(cents):<8.1%} {np.mean(sps):<8.1%} "
          f"{np.mean(al_ds):<8.1f} {np.mean(mb_ds):<8.1f} {np.mean(tot_ds):<8.1f} "
          f"{np.mean(mancs):<8.2f} {np.mean(gsomas):<8.1f}")
    print(f"{'Std':<6} {np.std(accs)*100:<8.1f} {np.std(cents)*100:<8.1f} {np.std(sps)*100:<8.1f} "
          f"{np.std(al_ds):<8.1f} {np.std(mb_ds):<8.1f} {np.std(tot_ds):<8.1f} "
          f"{np.std(mancs):<8.2f} {np.std(gsomas):<8.1f}")
    n_pass = sum(1 for r in all_results if r['mancini_pass'])
    print(f"\nMancini: {n_pass}/{len(SEEDS)} pass")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='STD Ablation Study')
    parser.add_argument('--condition', choices=['no_std_train', 'posthoc_std_off', 'both'],
                        default='both',
                        help='Which condition to run (default: both)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Run only this seed (default: all 5 seeds). '
                             'Useful for parallelising across terminal windows.')
    args = parser.parse_args()

    _pkg_root = Path(__file__).resolve().parent
    _parent = _pkg_root.parent
    _data_candidates = [
        _pkg_root / 'data',
        _parent / 'connectome_models' / 'data',
        _parent / 'relative_crawling' / 'connectome_models' / 'data',
    ]
    data_dir = next((p for p in _data_candidates if (p / 'kreher2008').is_dir()), None)
    if data_dir is None:
        raise FileNotFoundError('Cannot find connectome data (kreher2008/).')

    STD_ABLATION_DIR.mkdir(parents=True, exist_ok=True)

    # Allow single-seed override (useful when running seeds in parallel terminals
    # or when a long run was interrupted and you want to resume from a specific seed).
    if args.seed is not None:
        global SEEDS
        SEEDS = [args.seed]

    print("=" * 80)
    print("STD ABLATION STUDY")
    print("=" * 80)
    print(f"\nCondition : {args.condition}")
    print(f"Seeds     : {SEEDS}")
    print(f"Output    : {STD_ABLATION_DIR}")
    print(f"\nSTD model : Tsodyks-Markram (tau_rec ~200ms, U ~0.3)")
    print(f"  no_std_train   — train from scratch, disable_std=True always")
    print(f"  posthoc_std_off — load canonical weights, evaluate disable_std=True")

    train_dataset, test_dataset, odor_names = load_kreher2008_all_odors(
        data_dir, train_repeats=10, test_repeats=5,
        noise_std=NOISE_STD, noise_type=NOISE_TYPE)
    train_loader, test_loader = create_dataloaders(train_dataset, test_dataset, batch_size=16)
    n_odors = len(odor_names)

    df = pd.read_csv(data_dir / "kreher2008/orn_responses_normalized.csv", index_col=0)
    or_responses = torch.from_numpy(df.values).float()

    if args.condition in ('posthoc_std_off', 'both'):
        print(f"\n{'#'*70}")
        print("CONDITION: POST-HOC STD REMOVAL (canonical weights, STD disabled at eval)")
        print(f"{'#'*70}")
        run_posthoc_std_off(
            data_dir, test_loader, n_odors, or_responses,
            STD_ABLATION_DIR / 'posthoc_std_off')

    if args.condition in ('no_std_train', 'both'):
        print(f"\n{'#'*70}")
        print("CONDITION: NO-STD TRAINING (train from scratch, STD disabled throughout)")
        print(f"{'#'*70}")
        run_no_std_training(
            data_dir, train_loader, test_loader, n_odors, or_responses,
            STD_ABLATION_DIR / 'no_std_train')

    print(f"\n{'='*80}")
    print("STD ABLATION COMPLETE")
    print(f"Results: {STD_ABLATION_DIR}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
