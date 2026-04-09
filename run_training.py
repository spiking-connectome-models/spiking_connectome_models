"""
Canonical training script: All-Connections Non-AD (5 seeds x 300 epochs).

This is the CANONICAL training script for the paper. It trains 5 spiking models,
each from its own independently-trained rate-based teacher, using every synapse
in the Winding et al. 2023 connectome (AD + non-AD + gap junctions + realistic noise).

Non-AD connections (initialized weak at 1e-13 A, LR 0.05x):
  ORN→LN (182), LN→PN (582), LN→LN (605), PN→LN (518), PN→KC (3), LN→ORN (310)

Reports per seed at epoch 300: accuracy (linear + centroid), sparsity,
  per-pair decorrelation (AL + MB), Mancini, concentration invariance (4 tests).
Aggregates mean +/- std across 5 seeds.

Results saved to: results/all_connections_nonad_canonical/

Notebook section: Section A — Original Paper Figures (primary results source).

Usage:
    python -m spiking_connectome_models.run_training
"""
import sys
from pathlib import Path
_pkg_parent = str(Path(__file__).parent.parent)
if _pkg_parent not in sys.path:
    sys.path.insert(0, _pkg_parent)
# connectome_models (rate-based teacher) lives in relative_crawling/
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
import matplotlib.pyplot as plt

from spiking_connectome_models.model import SpikingConnectomeConstrainedModel
from spiking_connectome_models.layers import SpikingParams
from spiking_connectome_models.analysis.compute import (
    compute_per_pair_decorrelation as _compute_per_pair_decorrelation,
    compute_mean_sim_decorrelation, run_mancini_test as _run_mancini_test,
    run_concentration_invariance as _run_concentration_invariance,
)
from connectome_models.model import ConnectomeConstrainedModel
from spiking_connectome_models.dataset import load_kreher2008_all_odors, create_dataloaders

# ============================================================================
# CONFIGURATION (identical to realistic_noise_canonical + non-AD)
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
NONAD_LR = 0.05  # Non-AD connections: weak LR to avoid disrupting canonical
KC_LR = 4.0
KCKC_LR = 0.1
GSOMA_LR = 0.1
APL_TAU_LR = 0.05

NOISE_TYPE = 'multiplicative'
NOISE_STD = 0.3  # 30% CV (Kreher 2008 inter-fly variability)

# Boosted circuit noise (biologically realistic)
REALISTIC_PARAMS = SpikingParams(
    v_noise_std=1.0e-3,           # 1.0 mV
    i_noise_std=15e-12,           # 15 pA
    syn_noise_std=0.25,           # 25% CV
    threshold_jitter_std=1.0e-3,  # 1.0 mV
    orn_receptor_noise_std=0.10,  # 10% CV
    circuit_noise_enabled=True,
)

SEEDS = [42, 43, 44, 45, 46]
OUTPUT_DIR = Path(__file__).resolve().parent / 'results' / 'all_connections_nonad_canonical'

# Non-AD initialization
NONAD_INIT = np.log(1e-13)  # Very weak: ~0.1 fA

# Concentration invariance
CONCENTRATIONS = [0.03, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0]
HILL_EC50 = 1.0
HILL_N = 1
N_CONC_TRIALS = 10


# ============================================================================
# HELPER FUNCTIONS
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
            mult = NONAD_LR  # Non-AD connections: weak LR
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


# Delegate to analysis subpackage (canonical implementations)
def compute_per_pair_decorrelation(model, or_responses, n_trials=10):
    r = _compute_per_pair_decorrelation(model, or_responses, n_trials, NOISE_STD)
    return {
        'kc_or': r['kc_or_ratio'], 'kc_pn': r['kc_pn_ratio'], 'pn_or': r['pn_or_ratio'],
        'total_decorr_pct': r['total_decorr'], 'mb_decorr_pct': r['mb_decorr'],
        'al_decorr_pct': r['al_decorr'],
    }


def run_mancini(model, carbachol=1e-10, apl_inject=0.7):
    result = _run_mancini_test(model, carbachol, apl_inject)
    return result['ratio']


def evaluate_model(model, test_loader):
    model.eval()
    correct, total, sparsities = 0, 0, []
    with torch.no_grad():
        for bx, by in test_loader:
            logits, info = model(bx, return_all=True)
            correct += (logits.argmax(-1) == by).sum().item()
            total += len(by)
            sparsities.append(info['sparsity'])
    return correct / total, np.mean(sparsities)


def centroid_accuracy(model, or_responses, n_trials=20):
    from spiking_connectome_models.analysis.compute import centroid_accuracy as _centroid_accuracy
    return _centroid_accuracy(model, or_responses, n_trials, NOISE_STD)


def run_concentration_invariance(model, or_responses, seed):
    return _run_concentration_invariance(
        model, or_responses, seed, CONCENTRATIONS, HILL_EC50, HILL_N, N_CONC_TRIALS, NOISE_STD)


# ============================================================================
# TRAIN SINGLE MODEL
# ============================================================================
def train_single_model(seed, data_dir, train_loader, test_loader, n_odors, or_responses, output_dir):
    print(f"\n{'='*70}")
    print(f"TRAINING MODEL (seed {seed})")
    print(f"{'='*70}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    history = {
        'train_acc': [], 'test_acc': [], 'sparsity': [],
        'al_decorr': [], 'mb_decorr': [], 'mancini': [], 'g_soma': [],
    }

    # ---- TEACHER ----
    print(f"\nTraining teacher ({TEACHER_EPOCHS} epochs)...")
    teacher = ConnectomeConstrainedModel.from_data_dir(data_dir, n_odors=n_odors, n_or_types=21, target_sparsity=0.10)
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

    # ---- STUDENT (all connections + non-AD) ----
    print(f"\nSetting up student (all connections + non-AD, realistic noise, APL {APL_BOOST}x)...")
    student = SpikingConnectomeConstrainedModel.from_data_dir(
        data_dir, n_odors=n_odors, n_or_types=21, target_sparsity=0.10,
        params=REALISTIC_PARAMS, include_nonad=True)
    student.n_steps_al = N_STEPS
    student.n_steps_kc = N_STEPS

    with torch.no_grad():
        for name, param in student.named_parameters():
            if 'v_th' in name:
                if 'ln' in name:
                    param.fill_(LN_VTH_INIT)
                else:
                    param.fill_(-0.0425)
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

        # Initialize non-AD connections very weak
        if student.antennal_lobe.orn_ln_nonad is not None:
            student.antennal_lobe.orn_ln_nonad.log_strength.fill_(NONAD_INIT)
        if student.antennal_lobe.ln_pn_nonad is not None:
            student.antennal_lobe.ln_pn_nonad.log_strength.fill_(NONAD_INIT)
        if student.antennal_lobe.ln_pn_excit_nonad is not None:
            student.antennal_lobe.ln_pn_excit_nonad.log_strength.fill_(NONAD_INIT)
        if student.antennal_lobe.ln_ln_nonad is not None:
            student.antennal_lobe.ln_ln_nonad.log_strength.fill_(NONAD_INIT)
        if student.antennal_lobe.pn_ln_nonad is not None:
            student.antennal_lobe.pn_ln_nonad.log_strength.fill_(NONAD_INIT)
        if student.antennal_lobe.ln_orn_nonad is not None:
            student.antennal_lobe.ln_orn_nonad.log_strength.fill_(NONAD_INIT)
        if student.kc_layer.pn_kc_nonad is not None:
            student.kc_layer.pn_kc_nonad.log_strength.fill_(NONAD_INIT)

    # ---- TRAIN ----
    print(f"\nTraining student ({STUDENT_EPOCHS} epochs)...")
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
            logits, info = student(bx, return_all=True)
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
            test_acc, sparsity = evaluate_model(student, test_loader)
            decorr = compute_mean_sim_decorrelation(student, or_responses)
            mancini = run_mancini(student)
            g_soma = np.exp(student.kc_layer.kc_neurons.log_g_soma.item()) * 1e9

            history['train_acc'].append(train_correct / train_total)
            history['test_acc'].append(test_acc)
            history['sparsity'].append(sparsity)
            history['al_decorr'].append(decorr['al_decorr'])
            history['mb_decorr'].append(decorr['mb_decorr'])
            history['mancini'].append(mancini)
            history['g_soma'].append(g_soma)

            g_gap_ln = np.exp(student.antennal_lobe.log_g_gap_ln.item()) if student.antennal_lobe.log_g_gap_ln is not None else 0
            g_gap_pn = np.exp(student.antennal_lobe.log_g_gap_pn.item())
            g_gap_eln = np.exp(student.antennal_lobe.log_g_gap_eln_pn.item())

            print(f"  Ep {epoch+1}: Train={train_correct/train_total:.1%}, Test={test_acc:.1%}, "
                  f"Sp={sparsity:.1%}, AL={decorr['al_decorr']:.1f}%, MB={decorr['mb_decorr']:.1f}%, "
                  f"Manc={mancini:.2f}, g={g_soma:.1f}nS, "
                  f"gapLN={g_gap_ln:.2e}, gapPN={g_gap_pn:.2e}, gapeLN={g_gap_eln:.2e}")

        if (epoch + 1) == 300:
            ep300_state = {k: v.clone() for k, v in student.state_dict().items()}

    # ---- EVALUATE AT EPOCH 300 ----
    student.load_state_dict(ep300_state)
    student.eval()
    print(f"\n  --- Seed {seed} Epoch 300 Evaluation ---")

    test_acc, sparsity = evaluate_model(student, test_loader)
    cent_acc = centroid_accuracy(student, or_responses)
    pp_decorr = compute_per_pair_decorrelation(student, or_responses)
    mancini = run_mancini(student)
    mancini_pass = 1.5 <= mancini <= 2.5
    conc_results, conc_tests = run_concentration_invariance(student, or_responses, seed)

    # Gap junction conductances
    g_gap_ln = float(np.exp(student.antennal_lobe.log_g_gap_ln.item())) if student.antennal_lobe.log_g_gap_ln is not None else None
    g_gap_pn = float(np.exp(student.antennal_lobe.log_g_gap_pn.item()))
    g_gap_eln = float(np.exp(student.antennal_lobe.log_g_gap_eln_pn.item()))
    ln_pn_inhib_str = float(np.exp(student.antennal_lobe.ln_pn.log_strength.item()))
    ln_pn_excit_str = float(np.exp(student.antennal_lobe.ln_pn_excit.log_strength.item()))
    g_soma = np.exp(student.kc_layer.kc_neurons.log_g_soma.item()) * 1e9
    apl_gain = F.softplus(torch.tensor(student.kc_layer.apl.apl_gain.item())).item()
    kc_apl_strength = float(np.exp(student.kc_layer.apl.kc_apl_log_strength.item()))

    # Non-AD strengths
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

    print(f"  Acc: {test_acc:.1%}, Centroid: {cent_acc:.1%}, Sp: {sparsity:.1%}")
    print(f"  AL: {pp_decorr['al_decorr_pct']:.1f}%, MB: {pp_decorr['mb_decorr_pct']:.1f}%, Total: {pp_decorr['total_decorr_pct']:.1f}%")
    print(f"  Mancini: {mancini:.2f} ({'PASS' if mancini_pass else 'FAIL'})")
    print(f"  Conc: SubPN={'P' if conc_tests['sublinear_pn_gain'] else 'F'}, "
          f"FlatKC={'P' if conc_tests['flat_kc_activity'] else 'F'}, "
          f"RobClass={'P' if conc_tests['robust_classification'] else 'F'}, "
          f"Identity={conc_tests['odor_identity_preservation']}")
    print(f"  Gap: LN={g_gap_ln:.2e}, PN={g_gap_pn:.2e}, eLN={g_gap_eln:.2e}")
    print(f"  LN->PN: inhib={ln_pn_inhib_str:.2e}, excit={ln_pn_excit_str:.2e}")
    print(f"  g_soma: {g_soma:.1f} nS, APL gain: {apl_gain:.2f}, KC->APL: {kc_apl_strength:.2e}")
    print(f"  Non-AD: {', '.join(f'{k}={v:.2e}' for k, v in nonad_strengths.items())}")

    results = {
        'seed': seed, 'eval_epoch': 300,
        'accuracy': test_acc, 'centroid_accuracy': cent_acc, 'sparsity': sparsity,
        'per_pair_decorrelation': pp_decorr,
        'mancini': mancini, 'mancini_pass': mancini_pass,
        'concentration_invariance': {
            'per_concentration': conc_results,
            'predictions': {k: v for k, v in conc_tests.items() if isinstance(v, (bool, float, int, str))},
        },
        'g_soma_nS': g_soma, 'apl_gain_effective': apl_gain, 'kc_apl_strength': kc_apl_strength,
        'gap_junction_conductances': {'ln_ln': g_gap_ln, 'pn_pn': g_gap_pn, 'eln_pn': g_gap_eln},
        'ln_pn_split': {'inhibitory_strength': ln_pn_inhib_str, 'excitatory_strength': ln_pn_excit_str},
        'nonad_strengths': nonad_strengths,
        'history': history,
    }

    torch.save(student.state_dict(), output_dir / f'model_seed{seed}.pt')
    return results, student


# ============================================================================
# MAIN
# ============================================================================
def main():
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
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("ALL-CONNECTIONS NON-AD CANONICAL (5 models)")
    print("=" * 80)
    print(f"\nAll connectome connections modeled:")
    print(f"  AD (canonical): ORN->PN, ORN->LN, LN->PN, LN->LN, PN->LN, LN->ORN, PN->KC, KC->APL, APL->KC, KC-KC")
    print(f"  Non-AD (weak init, low LR): ORN->LN (182), LN->PN (582), LN->LN (605), PN->LN (518), LN->ORN (310), PN->KC (3)")
    print(f"  Gap junctions: LN-LN, PN-PN (sister), eLN-PN (bidirectional)")
    print(f"  Excitatory LN subtypes: Picky LNs (low fan-out, glutamatergic)")
    print(f"\nRealistic noise (identical to canonical):")
    print(f"  OR: {NOISE_STD*100:.0f}% CV, membrane: {REALISTIC_PARAMS.v_noise_std*1e3:.1f} mV, "
          f"background: {REALISTIC_PARAMS.i_noise_std*1e12:.0f} pA")
    print(f"  synaptic: {REALISTIC_PARAMS.syn_noise_std*100:.0f}% CV, threshold: "
          f"{REALISTIC_PARAMS.threshold_jitter_std*1e3:.1f} mV, receptor: {REALISTIC_PARAMS.orn_receptor_noise_std*100:.0f}%")
    print(f"\nNon-AD config: init={np.exp(NONAD_INIT):.1e}, LR={NONAD_LR}")
    print(f"\nSeeds: {SEEDS}")
    print(f"Epochs: {STUDENT_EPOCHS}")
    print(f"Output: {OUTPUT_DIR}")

    train_dataset, test_dataset, odor_names = load_kreher2008_all_odors(
        data_dir, train_repeats=10, test_repeats=5,
        noise_std=NOISE_STD, noise_type=NOISE_TYPE,
    )
    train_loader, test_loader = create_dataloaders(train_dataset, test_dataset, batch_size=16)
    n_odors = len(odor_names)

    df = pd.read_csv(data_dir / "kreher2008/orn_responses_normalized.csv", index_col=0)
    or_responses = torch.from_numpy(df.values).float()

    all_results = []
    for seed in SEEDS:
        results, model = train_single_model(
            seed, data_dir, train_loader, test_loader, n_odors, or_responses, OUTPUT_DIR
        )
        all_results.append(results)

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print(f"\n{'='*80}")
    print("ALL-CONNECTIONS NON-AD RESULTS SUMMARY (5 models, epoch 300)")
    print(f"{'='*80}")

    accs = [r['accuracy'] for r in all_results]
    cents = [r['centroid_accuracy'] for r in all_results]
    sps = [r['sparsity'] for r in all_results]
    al_ds = [r['per_pair_decorrelation']['al_decorr_pct'] for r in all_results]
    mb_ds = [r['per_pair_decorrelation']['mb_decorr_pct'] for r in all_results]
    total_ds = [r['per_pair_decorrelation']['total_decorr_pct'] for r in all_results]
    mancs = [r['mancini'] for r in all_results]
    gsomas = [r['g_soma_nS'] for r in all_results]

    # Per-seed table
    print(f"\n{'Seed':<6} {'Acc':<8} {'Cent':<8} {'Sp':<8} {'AL%':<8} {'MB%':<8} {'Tot%':<8} {'Manc':<8} {'g_soma':<8}")
    print("-" * 72)
    for r in all_results:
        pp = r['per_pair_decorrelation']
        m_ok = "P" if r['mancini_pass'] else "F"
        print(f"{r['seed']:<6} {r['accuracy']:<8.1%} {r['centroid_accuracy']:<8.1%} {r['sparsity']:<8.1%} "
              f"{pp['al_decorr_pct']:<8.1f} {pp['mb_decorr_pct']:<8.1f} {pp['total_decorr_pct']:<8.1f} "
              f"{r['mancini']:.2f}{m_ok:<3} {r['g_soma_nS']:<8.1f}")
    print("-" * 72)
    print(f"{'Mean':<6} {np.mean(accs):<8.1%} {np.mean(cents):<8.1%} {np.mean(sps):<8.1%} "
          f"{np.mean(al_ds):<8.1f} {np.mean(mb_ds):<8.1f} {np.mean(total_ds):<8.1f} "
          f"{np.mean(mancs):<8.2f} {np.mean(gsomas):<8.1f}")
    print(f"{'Std':<6} {np.std(accs)*100:<8.1f} {np.std(cents)*100:<8.1f} {np.std(sps)*100:<8.1f} "
          f"{np.std(al_ds):<8.1f} {np.std(mb_ds):<8.1f} {np.std(total_ds):<8.1f} "
          f"{np.std(mancs):<8.2f} {np.std(gsomas):<8.1f}")

    # Mancini pass rate
    n_pass = sum(1 for r in all_results if r['mancini_pass'])
    print(f"\nMancini: {n_pass}/{len(SEEDS)} pass")

    # Concentration invariance summary
    print(f"\nConcentration Invariance:")
    for test_name in ['sublinear_pn_gain', 'flat_kc_activity', 'robust_classification', 'odor_identity_preservation']:
        vals = [r['concentration_invariance']['predictions'][test_name] for r in all_results]
        if isinstance(vals[0], bool):
            n_pass_t = sum(vals)
            print(f"  {test_name}: {n_pass_t}/{len(SEEDS)} pass")
        else:
            print(f"  {test_name}: {', '.join(str(v) for v in vals)}")

    # Gap junction summary
    print(f"\nGap Junction Conductances (mean +/- std):")
    gj_ln = [r['gap_junction_conductances']['ln_ln'] for r in all_results if r['gap_junction_conductances']['ln_ln'] is not None]
    gj_pn = [r['gap_junction_conductances']['pn_pn'] for r in all_results]
    gj_eln = [r['gap_junction_conductances']['eln_pn'] for r in all_results]
    if gj_ln:
        print(f"  LN-LN: {np.mean(gj_ln):.2e} +/- {np.std(gj_ln):.2e}")
    print(f"  PN-PN: {np.mean(gj_pn):.2e} +/- {np.std(gj_pn):.2e}")
    print(f"  eLN-PN: {np.mean(gj_eln):.2e} +/- {np.std(gj_eln):.2e}")

    ln_inh = [r['ln_pn_split']['inhibitory_strength'] for r in all_results]
    ln_exc = [r['ln_pn_split']['excitatory_strength'] for r in all_results]
    print(f"\nLN->PN Split:")
    print(f"  Inhibitory: {np.mean(ln_inh):.2e} +/- {np.std(ln_inh):.2e}")
    print(f"  Excitatory: {np.mean(ln_exc):.2e} +/- {np.std(ln_exc):.2e}")

    # Non-AD strength summary
    print(f"\nNon-AD Connection Strengths (mean +/- std):")
    nonad_keys = list(all_results[0]['nonad_strengths'].keys())
    for key in nonad_keys:
        vals = [r['nonad_strengths'].get(key, 0) for r in all_results]
        print(f"  {key}: {np.mean(vals):.2e} +/- {np.std(vals):.2e}")

    # ========================================================================
    # SAVE
    # ========================================================================
    results_json = {
        'config': {
            'apl_boost': APL_BOOST, 'teacher_epochs': TEACHER_EPOCHS, 'student_epochs': STUDENT_EPOCHS,
            'g_soma_min_nS': G_SOMA_MIN * 1e9, 'g_soma_max_nS': G_SOMA_MAX_BIO * 1e9,
            'noise_type': NOISE_TYPE, 'noise_std': NOISE_STD, 'seeds': SEEDS,
            'v_noise_std': REALISTIC_PARAMS.v_noise_std,
            'i_noise_std': REALISTIC_PARAMS.i_noise_std,
            'syn_noise_std': REALISTIC_PARAMS.syn_noise_std,
            'threshold_jitter_std': REALISTIC_PARAMS.threshold_jitter_std,
            'orn_receptor_noise_std': REALISTIC_PARAMS.orn_receptor_noise_std,
            'nonad_init': float(np.exp(NONAD_INIT)),
            'nonad_lr': NONAD_LR,
        },
        'summary': {
            'accuracy_mean': float(np.mean(accs)), 'accuracy_std': float(np.std(accs)),
            'centroid_mean': float(np.mean(cents)), 'centroid_std': float(np.std(cents)),
            'sparsity_mean': float(np.mean(sps)), 'sparsity_std': float(np.std(sps)),
            'al_decorr_mean': float(np.mean(al_ds)), 'al_decorr_std': float(np.std(al_ds)),
            'mb_decorr_mean': float(np.mean(mb_ds)), 'mb_decorr_std': float(np.std(mb_ds)),
            'total_decorr_mean': float(np.mean(total_ds)), 'total_decorr_std': float(np.std(total_ds)),
            'mancini_mean': float(np.mean(mancs)), 'mancini_std': float(np.std(mancs)),
            'mancini_pass_rate': f"{n_pass}/{len(SEEDS)}",
            'g_soma_mean': float(np.mean(gsomas)), 'g_soma_std': float(np.std(gsomas)),
        },
        'per_seed': [],
    }
    for r in all_results:
        seed_data = {k: v for k, v in r.items() if k != 'history'}
        seed_data['history'] = {
            'epochs': list(range(50, STUDENT_EPOCHS + 1, 50)),
            **{k: v for k, v in r['history'].items()},
        }
        results_json['per_seed'].append(seed_data)

    with open(OUTPUT_DIR / 'results.json', 'w') as f:
        json.dump(results_json, f, indent=2, default=str)

    # ========================================================================
    # PLOTS
    # ========================================================================
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    epochs = list(range(50, STUDENT_EPOCHS + 1, 50))

    ax = axes[0, 0]
    for r in all_results:
        ax.plot(epochs, [a*100 for a in r['history']['test_acc']], 'o-', alpha=0.7, label=f"Seed {r['seed']}")
    ax.set_xlabel('Epoch'); ax.set_ylabel('Test Accuracy (%)'); ax.set_title('Test Accuracy')
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    for r in all_results:
        ax.plot(epochs, [s*100 for s in r['history']['sparsity']], 'o-', alpha=0.7, label=f"Seed {r['seed']}")
    ax.axhline(y=10, color='r', linestyle='--', alpha=0.5, label='Target')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Sparsity (%)'); ax.set_title('KC Sparsity')
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    for r in all_results:
        ax.plot(epochs, r['history']['al_decorr'], 's--', alpha=0.5, label=f"AL s{r['seed']}")
        ax.plot(epochs, r['history']['mb_decorr'], 'o-', alpha=0.7, label=f"MB s{r['seed']}")
    ax.set_xlabel('Epoch'); ax.set_ylabel('Decorrelation (%)'); ax.set_title('AL vs MB Decorrelation')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    for r in all_results:
        ax.plot(epochs, r['history']['mancini'], 'o-', alpha=0.7, label=f"Seed {r['seed']}")
    ax.axhline(y=2.0, color='g', linestyle='--', alpha=0.5)
    ax.axhline(y=1.5, color='r', linestyle='--', alpha=0.3)
    ax.axhline(y=2.5, color='r', linestyle='--', alpha=0.3)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Mancini Ratio'); ax.set_title('APL Inhibition')
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    for r in all_results:
        ax.plot(epochs, r['history']['g_soma'], 'o-', alpha=0.7, label=f"Seed {r['seed']}")
    ax.axhline(y=G_SOMA_MAX_BIO*1e9, color='r', linestyle='--', alpha=0.3)
    ax.axhline(y=G_SOMA_MIN*1e9, color='r', linestyle='--', alpha=0.3)
    ax.set_xlabel('Epoch'); ax.set_ylabel('g_soma (nS)'); ax.set_title('KC Soma Conductance')
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    x = np.arange(len(SEEDS)); width = 0.15
    ax.bar(x - 2*width, [r['accuracy']*100 for r in all_results], width, label='Acc', color='blue', alpha=0.7)
    ax.bar(x - width, [r['centroid_accuracy']*100 for r in all_results], width, label='Cent', color='cyan', alpha=0.7)
    ax.bar(x, [r['per_pair_decorrelation']['al_decorr_pct'] for r in all_results], width, label='AL%', color='orange', alpha=0.7)
    ax.bar(x + width, [r['per_pair_decorrelation']['mb_decorr_pct'] for r in all_results], width, label='MB%', color='red', alpha=0.7)
    ax.bar(x + 2*width, [r['mancini']*20 for r in all_results], width, label='Manc*20', color='purple', alpha=0.7)
    ax.set_xlabel('Seed'); ax.set_ylabel('Value'); ax.set_title('Final Metrics by Seed')
    ax.set_xticks(x); ax.set_xticklabels(SEEDS); ax.legend(loc='upper right'); ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'training_curves.png', dpi=150)
    plt.close()

    print(f"\nResults saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
