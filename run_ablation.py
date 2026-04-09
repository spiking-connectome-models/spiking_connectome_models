"""
run_ablation.py

Unified ablation training script for CCN 2026 revisions.
Handles C1 (component ablations), C3 (LN threshold), C4 (sparsity ablations),
and APL-boosted energy experiments — all via command-line flags.

All conditions share the same training infrastructure. Teacher is trained once
per seed and cached to disk.

Examples:
    # Train teacher for a seed
    python run_ablation.py --train-teacher --seed 42

    # C1(i): No gap junctions
    python run_ablation.py --no-gap --kc-sparsity --seed 42 --label c1i_no_gap_s42

    # C1(ii) / C4(ii): No APL + KC sparsity
    python run_ablation.py --no-apl --kc-sparsity --seed 42 --label c1ii_no_apl_s42

    # C1(iii): Shuffled connectome
    python run_ablation.py --shuffle --kc-sparsity --seed 42 --label c1iii_shuffle_s42

    # C3: Different LN fan-out quantile
    python run_ablation.py --kc-sparsity --ln-quantile 0.50 --seed 42 --label c3_q050_s42

    # APL-boosted energy
    python run_ablation.py --energy-weight 15 --apl-boost 8 --seed 42 --label e15_apl8_s42

Results saved to: results/ablations_c7/

Notebook sections: Section B — C1 (retrained ablations),
                   Section B — C3 (LN threshold variants),
                   Section B — C4 (c4i_no_sp sparsity ablation).
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
# SPIKE ACCUMULATOR
# ============================================================================
class SpikeAccumulator:
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
# CONFIGURATION
# ============================================================================
G_SOMA_MIN, G_SOMA_MAX_BIO = 1e-9, 20e-9
KCKC_LOG_MIN = np.log(1e-15)
LOG_STRENGTH_MAX = np.log(1e-8)

APL_BOOST_DEFAULT = 4.0
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
NONAD_INIT = np.log(1e-13)

REALISTIC_PARAMS = SpikingParams(
    v_noise_std=1.0e-3, i_noise_std=15e-12, syn_noise_std=0.25,
    threshold_jitter_std=1.0e-3, orn_receptor_noise_std=0.10,
    circuit_noise_enabled=True,
)

CONCENTRATIONS = [0.03, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0]
HILL_EC50 = 1.0
HILL_N = 1
N_CONC_TRIALS = 10
ENERGY_RAMP_EPOCHS = 60
DEVICE = torch.device('cpu')

_pkg_root = Path(__file__).resolve().parent
OUTPUT_DIR = _pkg_root / 'results' / 'ablations_c7'
TEACHER_DIR = OUTPUT_DIR / 'teachers'

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


def get_param_groups(model, apl_lr_mult=1.0):
    """Build optimizer param groups. apl_lr_mult scales APL gain learning rate."""
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
            mult = KC_LR * apl_lr_mult
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
# CONNECTOME SHUFFLING (C1iii: preserve degree distribution per neuron)
# ============================================================================
def shuffle_connectome(model, seed):
    """Shuffle all connectome weight matrices while preserving per-neuron degree.

    C1(iii): Reviewer 2 — tests whether specific wiring matters or just
    degree distribution. For each connectivity matrix, independently permute
    rows (presynaptic) and columns (postsynaptic), preserving marginal sums.
    """
    rng = np.random.RandomState(seed + 1000)  # Offset to avoid seed collision

    def shuffle_buffer(layer, attr='weight_matrix'):
        if not hasattr(layer, attr):
            return
        W = getattr(layer, attr)
        if W is None:
            return
        W_np = W.cpu().numpy().copy()
        # Permute rows and columns independently
        row_perm = rng.permutation(W_np.shape[0])
        col_perm = rng.permutation(W_np.shape[1])
        W_shuffled = W_np[row_perm][:, col_perm]
        W.copy_(torch.from_numpy(W_shuffled).to(W.dtype))

    # AL pathways
    al = model.antennal_lobe
    for layer_name in ['orn_pn', 'orn_ln', 'ln_pn', 'ln_pn_excit',
                       'orn_ln_nonad', 'ln_pn_nonad', 'ln_pn_excit_nonad',
                       'ln_ln_nonad', 'pn_ln_nonad', 'ln_orn_nonad']:
        layer = getattr(al, layer_name, None)
        if layer is not None and hasattr(layer, 'weight_matrix'):
            shuffle_buffer(layer)

    # KC pathways
    kc = model.kc_layer
    for layer_name in ['pn_kc', 'pn_kc_nonad', 'kc_kc_aa', 'kc_kc_ad']:
        layer = getattr(kc, layer_name, None)
        if layer is not None and hasattr(layer, 'weight_matrix'):
            shuffle_buffer(layer)

    # APL pathways
    apl = kc.apl
    for buf_name in ['kc_apl_weights', 'apl_kc_weights', 'kc_apl_da_weights']:
        if hasattr(apl, buf_name):
            buf = getattr(apl, buf_name)
            if buf is not None:
                buf_np = buf.cpu().numpy().copy()
                row_perm = rng.permutation(buf_np.shape[0])
                col_perm = rng.permutation(buf_np.shape[1])
                buf.copy_(torch.from_numpy(buf_np[row_perm][:, col_perm]).to(buf.dtype))

    # Gap junction masks
    for mask_name in ['gap_ln_ln_mask', 'gap_pn_pn_mask', 'gap_eln_pn_mask']:
        if hasattr(al, mask_name):
            mask = getattr(al, mask_name)
            if mask is not None:
                m_np = mask.cpu().numpy().copy()
                perm = rng.permutation(m_np.shape[0])
                if m_np.shape[0] == m_np.shape[1]:
                    # Square symmetric matrix: use same perm for rows and cols
                    m_shuffled = m_np[perm][:, perm]
                else:
                    col_perm = rng.permutation(m_np.shape[1])
                    m_shuffled = m_np[perm][:, col_perm]
                mask.copy_(torch.from_numpy(m_shuffled).to(mask.dtype))

    print("  Connectome shuffled (degree-preserving permutation)")


# ============================================================================
# LN THRESHOLD OVERRIDE (C3: vary Picky vs Broad/Choosy classification)
# ============================================================================
def override_ln_threshold(model, data_dir, quantile):
    """Reclassify LN subtypes with a different fan-out quantile.

    C3: Reviewer 3, Point 2 — sensitivity to LN classification threshold.
    Default is 0.33 (33rd percentile). Berck et al. 2016 proportional split
    would be ~0.417 (5 picky / 12 connected = 5:7 split).

    BUG FIX: Must rebuild ln_pn and ln_pn_excit SpikingConnectomeLinear layers,
    not just update the is_excitatory_ln buffer. The old code only changed the
    buffer, which had no effect because the weight matrices were already baked
    into separate inhibitory/excitatory layers at init time.
    """
    from spiking_connectome_models.layers import SpikingConnectomeLinear

    winding_dir = data_dir / 'winding2023'
    ln_to_pn = torch.load(winding_dir / 'ln_to_pn.pt', weights_only=True)

    fan_out = (ln_to_pn > 0).sum(dim=1)
    has_pn = fan_out > 0
    n_connected = has_pn.sum().item()

    if n_connected > 0:
        connected_fan_out = fan_out[has_pn].float()
        threshold = torch.quantile(connected_fan_out, quantile)
        is_excitatory_ln = has_pn & (fan_out <= threshold)
    else:
        is_excitatory_ln = torch.zeros(ln_to_pn.shape[0], dtype=torch.bool)

    n_excit = is_excitatory_ln.sum().item()
    n_inhib = (has_pn & ~is_excitatory_ln).sum().item()
    n_silent = (~has_pn).sum().item()

    # Update buffer
    model.antennal_lobe.is_excitatory_ln.copy_(is_excitatory_ln)

    # Rebuild AD LN→PN layers with new classification  [C3 FIX]
    al = model.antennal_lobe
    ln_pn_inhib = ln_to_pn.clone()
    ln_pn_inhib[is_excitatory_ln] = 0
    ln_pn_excit = ln_to_pn.clone()
    ln_pn_excit[~is_excitatory_ln] = 0

    # Save log_strength values before replacing layers — initialize_student sets
    # these before calling override_ln_threshold, and new layers reset them.
    old_ln_pn_log_strength = al.ln_pn.log_strength.item()
    old_ln_pn_excit_log_strength = al.ln_pn_excit.log_strength.item()

    al.ln_pn = SpikingConnectomeLinear(ln_pn_inhib, al.ln_pn.params, sign="inhibitory")
    al.ln_pn_excit = SpikingConnectomeLinear(ln_pn_excit, al.ln_pn_excit.params, sign="excitatory")

    # Restore log_strength values that were set by initialize_student
    with torch.no_grad():
        al.ln_pn.log_strength.fill_(old_ln_pn_log_strength)
        al.ln_pn_excit.log_strength.fill_(old_ln_pn_excit_log_strength)
    print(f"  log_strength restored: ln_pn={old_ln_pn_log_strength:.4f}, "
          f"ln_pn_excit={old_ln_pn_excit_log_strength:.4f}")

    # Rebuild non-AD LN→PN layers if they exist
    ln_to_pn_nonad = None
    nonad_path = winding_dir / 'ln_to_pn_nonad.pt'
    if nonad_path.exists():
        ln_to_pn_nonad = torch.load(nonad_path, weights_only=True)
    if ln_to_pn_nonad is not None and al.ln_pn_nonad is not None:
        ln_pn_nonad_inhib = ln_to_pn_nonad.clone()
        ln_pn_nonad_inhib[is_excitatory_ln] = 0
        ln_pn_nonad_excit = ln_to_pn_nonad.clone()
        ln_pn_nonad_excit[~is_excitatory_ln] = 0
        NONAD_INIT = 1e-12
        al.ln_pn_nonad = SpikingConnectomeLinear(
            ln_pn_nonad_inhib, al.ln_pn_nonad.params,
            init_strength=NONAD_INIT, sign="inhibitory")
        al.ln_pn_excit_nonad = SpikingConnectomeLinear(
            ln_pn_nonad_excit, al.ln_pn_excit_nonad.params,
            init_strength=NONAD_INIT, sign="excitatory")

    # Rebuild eLN-PN gap junction mask
    gap_eln_pn = ln_to_pn.clone().float()
    gap_eln_pn[~is_excitatory_ln] = 0
    gap_eln_pn = (gap_eln_pn > 0).float()
    al.gap_eln_pn_mask.copy_(gap_eln_pn)

    print(f"  LN reclassified (quantile={quantile:.2f}): "
          f"{n_excit} excit (Picky), {n_inhib} inhib (Broad/Choosy), {n_silent} silent")
    print(f"  Rebuilt ln_pn ({int((ln_pn_inhib > 0).sum())} inhib), "
          f"ln_pn_excit ({int((ln_pn_excit > 0).sum())} excit), "
          f"eLN-PN gap ({int(gap_eln_pn.sum())} conns)")


# ============================================================================
# STUDENT INITIALIZATION WITH ABLATIONS
# ============================================================================
def initialize_student(data_dir, n_odors, teacher_state, args):
    """Create student model with optional ablations applied."""
    apl_boost = args.apl_boost

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
        student.kc_layer.apl.apl_gain.data = teacher.kc_layer.apl.apl_gain.data.clone() * apl_boost

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

    # ---- APPLY ABLATIONS ----

    # C1(i): Disable gap junctions
    if args.no_gap:
        with torch.no_grad():
            if hasattr(student.antennal_lobe, 'log_g_gap_ln'):
                student.antennal_lobe.log_g_gap_ln.fill_(np.log(1e-30))
                student.antennal_lobe.log_g_gap_ln.requires_grad = False
            if hasattr(student.antennal_lobe, 'log_g_gap_pn'):
                student.antennal_lobe.log_g_gap_pn.fill_(np.log(1e-30))
                student.antennal_lobe.log_g_gap_pn.requires_grad = False
            if hasattr(student.antennal_lobe, 'log_g_gap_eln_pn'):
                student.antennal_lobe.log_g_gap_eln_pn.fill_(np.log(1e-30))
                student.antennal_lobe.log_g_gap_eln_pn.requires_grad = False
        print("  Gap junctions DISABLED (all conductances frozen at ~0)")

    # C1(ii) / C4(ii): Disable APL
    # BUG FIX: apl_gain=0 gives softplus(0)=0.693, so APL is still ~53% active.
    # Use -100 so softplus(-100)≈0, truly zeroing APL output.  [Reviewer 2, Pt 7]
    if args.no_apl:
        with torch.no_grad():
            student.kc_layer.apl.apl_gain.fill_(-100.0)
        for p in student.kc_layer.apl.parameters():
            p.requires_grad = False
        print("  APL DISABLED (gain=-100 → softplus≈0, all APL params frozen)")

    # C1(iii): Shuffle connectome
    if args.shuffle:
        shuffle_connectome(student, args.seed)

    # C3: Override LN threshold
    if args.ln_quantile is not None:
        override_ln_threshold(student, data_dir, args.ln_quantile)

    return student.to(DEVICE)


# ============================================================================
# TRAIN TEACHER
# ============================================================================
def train_teacher(seed, data_dir, n_odors, train_loader, test_loader):
    TEACHER_DIR.mkdir(parents=True, exist_ok=True)
    teacher_path = TEACHER_DIR / f'teacher_seed{seed}.pt'

    if teacher_path.exists():
        print(f"  Teacher for seed {seed} already exists, loading...")
        return torch.load(teacher_path, weights_only=False)

    print(f"--- Training teacher (seed {seed}, {TEACHER_EPOCHS} epochs) ---")
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
            print(f"  Teacher epoch {ep+1}: {c/t:.1%}")

    teacher_state = {k: v.cpu().clone() for k, v in teacher.state_dict().items()}
    torch.save(teacher_state, teacher_path)
    print(f"  Teacher saved to {teacher_path}")
    return teacher_state


# ============================================================================
# TRAIN STUDENT
# ============================================================================
def train_student(args, data_dir, train_loader, test_loader,
                  n_odors, or_responses, teacher_state):
    energy_weight = args.energy_weight
    kc_sparsity = args.kc_sparsity
    kc_energy_only = args.kc_energy_only
    label = args.label
    seed = args.seed

    loss_parts = ["CE"]
    if kc_sparsity: loss_parts.append("KC_sp")
    if energy_weight > 0:
        etype = "KC_energy" if kc_energy_only else "all_energy"
        loss_parts.append(f"{energy_weight}*{etype}")
    ablations = []
    if args.no_gap: ablations.append("no_gap")
    if args.no_apl: ablations.append("no_apl")
    if args.shuffle: ablations.append("shuffle")
    if args.ln_quantile is not None: ablations.append(f"ln_q{args.ln_quantile:.2f}")
    if args.apl_boost != APL_BOOST_DEFAULT: ablations.append(f"apl_boost={args.apl_boost}")

    loss_desc = " + ".join(loss_parts)
    if ablations: loss_desc += f" [{', '.join(ablations)}]"

    print(f"\n{'='*70}")
    print(f"TRAINING: {label} (seed {seed})")
    print(f"Loss = {loss_desc}")
    print(f"{'='*70}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    student = initialize_student(data_dir, n_odors, teacher_state, args)
    accumulator = SpikeAccumulator().register(student)

    param_groups = get_param_groups(student, apl_lr_mult=args.apl_lr_mult)
    optimizer = torch.optim.Adam(param_groups)

    for epoch in range(STUDENT_EPOCHS):
        progress = min(1.0, epoch / ENERGY_RAMP_EPOCHS)
        sp_w = progress * MAX_SP_WEIGHT if kc_sparsity else 0.0
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

            kc_rates = info['kc_spikes'] / N_STEPS
            if kc_sparsity:
                sp_loss = (torch.sigmoid((kc_rates - 0.02) * 50).mean() - 0.05) ** 2
                total_loss = total_loss + sp_w * sp_loss

            if energy_weight > 0:
                if kc_energy_only:
                    energy_loss = kc_rates.mean()
                else:
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

        if (epoch + 1) % 50 == 0:
            student.eval()
            tc, tt = 0, 0
            all_orn, all_ln, all_pn, all_kc = [], [], [], []
            with torch.no_grad():
                for bx, by in test_loader:
                    bx, by = bx.to(DEVICE), by.to(DEVICE)
                    accumulator.reset()
                    logits, info = student(bx, return_all=True)
                    tc += (logits.argmax(-1) == by).sum().item()
                    tt += len(by)
                    all_orn.append((accumulator.orn_spikes > 0).float().mean().item())
                    all_ln.append((accumulator.ln_spikes > 0).float().mean().item())
                    all_pn.append((info['pn_spikes'] > 0).float().mean().item())
                    all_kc.append((info['kc_spikes'] > 0).float().mean().item())

            student.cpu()
            decorr = compute_mean_sim_decorrelation(student, or_responses)
            manc = _run_mancini_test(student)
            student.to(DEVICE)

            print(f"  Ep {epoch+1}: Test={tc/tt:.1%}, "
                  f"KC={np.mean(all_kc):.1%}, ORN={np.mean(all_orn):.1%}, "
                  f"LN={np.mean(all_ln):.1%}, PN={np.mean(all_pn):.1%}, "
                  f"AL={decorr['al_decorr']:.1f}%, MB={decorr['mb_decorr']:.1f}%, "
                  f"Manc={manc['ratio']:.2f}")

    # ---- FINAL EVALUATION ----
    student.eval()
    print(f"\n  --- FINAL EVALUATION: {label} ---")

    tc, tt = 0, 0
    all_orn, all_ln, all_pn, all_kc = [], [], [], []
    with torch.no_grad():
        for bx, by in test_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            accumulator.reset()
            logits, info = student(bx, return_all=True)
            tc += (logits.argmax(-1) == by).sum().item()
            tt += len(by)
            all_orn.append((accumulator.orn_spikes > 0).float().mean().item())
            all_ln.append((accumulator.ln_spikes > 0).float().mean().item())
            all_pn.append((info['pn_spikes'] > 0).float().mean().item())
            all_kc.append((info['kc_spikes'] > 0).float().mean().item())

    accumulator.remove()
    test_acc = tc / tt
    per_type_sp = {
        'orn': float(np.mean(all_orn)), 'ln': float(np.mean(all_ln)),
        'pn': float(np.mean(all_pn)), 'kc': float(np.mean(all_kc)),
    }

    student.cpu()
    pp = _compute_per_pair_decorrelation(student, or_responses, 10, NOISE_STD)
    # Skip Mancini test when APL is disabled — the test is meaningless without APL
    if args.no_apl:
        manc = {'ratio': float('nan'), 'passes': False,
                'baseline_spikes': float('nan'), 'boosted_spikes': float('nan'),
                'skipped': True}
        print("  Mancini: SKIPPED (APL disabled)")
    else:
        manc = _run_mancini_test(student)
    cent_acc = _centroid_accuracy(student, or_responses, 20, NOISE_STD)
    conc_results, conc_tests = _run_concentration_invariance(
        student, or_responses, seed, CONCENTRATIONS, HILL_EC50, HILL_N,
        N_CONC_TRIALS, NOISE_STD)

    g_soma = np.exp(student.kc_layer.kc_neurons.log_g_soma.item()) * 1e9
    apl_gain = F.softplus(torch.tensor(student.kc_layer.apl.apl_gain.item())).item()

    # Gap junction conductances
    gap_conductances = {}
    al = student.antennal_lobe
    for gname in ['log_g_gap_ln', 'log_g_gap_pn', 'log_g_gap_eln_pn']:
        if hasattr(al, gname):
            gap_conductances[gname.replace('log_g_gap_', '') + '_nS'] = float(
                np.exp(getattr(al, gname).item()) * 1e9)

    print(f"  Acc:     linear={test_acc:.1%}, centroid={cent_acc:.1%}")
    print(f"  Sparse:  ORN={per_type_sp['orn']:.1%}, LN={per_type_sp['ln']:.1%}, "
          f"PN={per_type_sp['pn']:.1%}, KC={per_type_sp['kc']:.1%}")
    print(f"  Decorr:  AL={pp['al_decorr']:.1f}%, MB={pp['mb_decorr']:.1f}%")
    print(f"  Mancini: {manc['ratio']:.2f} ({'PASS' if manc['passes'] else 'FAIL'})")
    print(f"  Gain:    OR={conc_tests['or_range']:.2f}x, "
          f"PN={conc_tests['pn_range']:.2f}x, KC={conc_tests['kc_range']:.2f}x")
    print(f"  FlatKC:  {conc_tests['flat_kc_activity']}, SubPN: {conc_tests['sublinear_pn_gain']}")

    model_path = OUTPUT_DIR / f'model_{label}.pt'
    torch.save(student.state_dict(), model_path)

    results = {
        'label': label, 'seed': seed, 'loss': loss_desc,
        'energy_weight': energy_weight, 'apl_boost': args.apl_boost,
        'accuracy': test_acc, 'centroid_accuracy': cent_acc,
        'per_type_sparsity': per_type_sp,
        'decorrelation': {
            'al': pp['al_decorr'], 'mb': pp['mb_decorr'], 'total': pp['total_decorr'],
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
        'gap_conductances_nS': gap_conductances,
    }

    results_path = OUTPUT_DIR / f'results_{label}.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"  Saved: {results_path}")
    print(f"  DONE: {label}")


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description='Ablation training for CCN revisions')
    parser.add_argument('--train-teacher', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--label', type=str, required=False)
    # Loss components
    parser.add_argument('--energy-weight', type=float, default=0.0)
    parser.add_argument('--kc-sparsity', action='store_true')
    parser.add_argument('--kc-energy-only', action='store_true')
    # Ablations
    parser.add_argument('--no-gap', action='store_true', help='C1(i): disable gap junctions')
    parser.add_argument('--no-apl', action='store_true', help='C1(ii)/C4(ii): disable APL')
    parser.add_argument('--shuffle', action='store_true', help='C1(iii): shuffle connectome')
    # Sensitivity
    parser.add_argument('--ln-quantile', type=float, default=None,
                        help='C3: LN fan-out quantile (default 0.33)')
    parser.add_argument('--apl-boost', type=float, default=APL_BOOST_DEFAULT,
                        help='APL gain boost factor (default 4.0)')
    parser.add_argument('--apl-lr-mult', type=float, default=1.0,
                        help='Multiplier on APL gain learning rate')
    args = parser.parse_args()

    _parent = Path(__file__).resolve().parent.parent
    _data_candidates = [
        Path(__file__).resolve().parent / 'data',
        _parent / 'connectome_models' / 'data',
        _parent / 'relative_crawling' / 'connectome_models' / 'data',
    ]
    data_dir = next((p for p in _data_candidates if (p / 'kreher2008').is_dir()), None)
    if data_dir is None:
        raise FileNotFoundError('Cannot find connectome data.')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_ds, test_ds, odor_names = load_kreher2008_all_odors(
        data_dir, train_repeats=10, test_repeats=5,
        noise_std=NOISE_STD, noise_type=NOISE_TYPE)
    train_loader, test_loader = create_dataloaders(train_ds, test_ds, batch_size=16)
    n_odors = len(odor_names)
    df = pd.read_csv(data_dir / "kreher2008/orn_responses_normalized.csv", index_col=0)
    or_responses = torch.from_numpy(df.values).float()

    if args.train_teacher:
        train_teacher(args.seed, data_dir, n_odors, train_loader, test_loader)
        return

    teacher_state = train_teacher(args.seed, data_dir, n_odors, train_loader, test_loader)
    train_student(args, data_dir, train_loader, test_loader,
                  n_odors, or_responses, teacher_state)


if __name__ == "__main__":
    main()
