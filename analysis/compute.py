"""Computation functions for connectome model analysis.

All non-plotting analysis: model loading, evaluation, decorrelation,
Mancini test, per-odor metrics, consistency, centroid accuracy,
gap junctions, non-AD strengths, concentration invariance, parameter CV.
"""

import numpy as np
import torch
from pathlib import Path
from scipy.stats import pearsonr
from itertools import combinations

from spiking_connectome_models.model import SpikingConnectomeConstrainedModel
from spiking_connectome_models.layers import SpikingParams

from .utils import (
    cosine_sim_matrix, cosine_sim_pair, noisy_forward_pass,
    per_pair_similarity_ratios, centroid_classify,
    hill_effective_concentration,
)

# Default noise parameters (biologically realistic)
REALISTIC_PARAMS = SpikingParams(
    v_noise_std=1.0e-3,           # 1.0 mV
    i_noise_std=15e-12,           # 15 pA
    syn_noise_std=0.25,           # 25% CV
    threshold_jitter_std=1.0e-3,  # 1.0 mV
    orn_receptor_noise_std=0.10,  # 10% CV
    circuit_noise_enabled=True,
)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_models(data_dir, model_dir, seeds, n_steps=30, include_nonad=True):
    """Load all trained models.

    Args:
        data_dir: Path to connectome_models/data
        model_dir: Path to results directory with model_seed{seed}.pt files
        seeds: list of random seeds
        n_steps: simulation timesteps (MUST be 30 for canonical models)
        include_nonad: whether to include non-AD connections
    """
    models = []
    for seed in seeds:
        model = SpikingConnectomeConstrainedModel.from_data_dir(
            data_dir, n_odors=28, n_or_types=21, target_sparsity=0.10,
            params=REALISTIC_PARAMS, include_nonad=include_nonad,
        )
        model.load_state_dict(torch.load(model_dir / f'model_seed{seed}.pt', weights_only=True))
        model.n_steps_al = n_steps
        model.n_steps_kc = n_steps
        model.eval()
        models.append(model)
    return models


# ---------------------------------------------------------------------------
# Basic evaluation
# ---------------------------------------------------------------------------

def evaluate_model(model, test_loader):
    """Evaluate model accuracy and sparsity on test data.

    Returns dict with accuracy, sparsity, or_sim, pn_sim, kc_sim.
    """
    model.eval()
    correct, total, sparsities = 0, 0, []
    or_all, pn_all, kc_all = [], [], []
    with torch.no_grad():
        for bx, by in test_loader:
            logits, info = model(bx, return_all=True)
            correct += (logits.argmax(-1) == by).sum().item()
            total += len(by)
            sparsities.append(info['sparsity'])
            or_all.append(bx)
            pn_all.append(info['pn_spikes'].float())
            kc_all.append(info['kc_spikes'].float())
    return {
        'accuracy': correct / total,
        'sparsity': np.mean(sparsities),
        'or_sim': cosine_sim_matrix(torch.cat(or_all)),
        'pn_sim': cosine_sim_matrix(torch.cat(pn_all)),
        'kc_sim': cosine_sim_matrix(torch.cat(kc_all)),
    }


# ---------------------------------------------------------------------------
# Decorrelation
# ---------------------------------------------------------------------------

def compute_per_pair_decorrelation(model, or_responses, n_trials=10, noise_std=0.3):
    """Compute per-pair ratio decorrelation (CANONICAL method).

    For each odor pair, compute ratio of downstream to upstream cosine similarity.
    Returns dict with kc_or_ratio, pn_or_ratio, kc_pn_ratio, and decorrelation %.
    """
    data = noisy_forward_pass(model, or_responses, n_trials, noise_std, seed=99999)
    or_pats = np.array([p.mean(axis=0) for p in data['or_patterns']])
    pn_pats = np.array([p.mean(axis=0) for p in data['pn_patterns']])
    kc_pats = np.array([p.mean(axis=0) for p in data['kc_patterns']])

    ratios = per_pair_similarity_ratios(or_pats, pn_pats, kc_pats)
    return {
        'kc_or_ratio': np.mean(ratios['kc_or_ratios']),
        'pn_or_ratio': np.mean(ratios['pn_or_ratios']),
        'kc_pn_ratio': np.mean(ratios['kc_pn_ratios']),
        'total_decorr': (1 - np.mean(ratios['kc_or_ratios'])) * 100,
        'al_decorr': (1 - np.mean(ratios['pn_or_ratios'])) * 100,
        'mb_decorr': (1 - np.mean(ratios['kc_pn_ratios'])) * 100,
    }


def compute_mean_sim_decorrelation(model, or_responses, n_trials=10, noise_std=0.3):
    """Compute mean-similarity decorrelation (used for training monitoring).

    Less conservative than per-pair method. Uses mean pairwise cosine similarity
    across all patterns, then computes ratios.
    """
    data = noisy_forward_pass(model, or_responses, n_trials, noise_std, seed=99999)
    or_pats = torch.from_numpy(np.array([p.mean(axis=0) for p in data['or_patterns']]))
    pn_pats = torch.from_numpy(np.array([p.mean(axis=0) for p in data['pn_patterns']]))
    kc_pats = torch.from_numpy(np.array([p.mean(axis=0) for p in data['kc_patterns']]))
    or_sim = cosine_sim_matrix(or_pats)
    pn_sim = cosine_sim_matrix(pn_pats)
    kc_sim = cosine_sim_matrix(kc_pats)
    return {
        'or_sim': or_sim, 'pn_sim': pn_sim, 'kc_sim': kc_sim,
        'al_decorr': (1 - pn_sim / max(or_sim, 1e-8)) * 100,
        'mb_decorr': (1 - kc_sim / max(pn_sim, 1e-8)) * 100,
        'total_decorr': (1 - kc_sim / max(or_sim, 1e-8)) * 100,
    }


# ---------------------------------------------------------------------------
# Mancini APL inhibition test
# ---------------------------------------------------------------------------

def run_mancini_test(model, carbachol=1e-10, apl_inject=0.7, n_trials=20):
    """Mancini APL inhibition test. Expected ratio ~2.0.

    Compares KC spiking with vs without APL current injection.
    Passes if ratio is in [1.5, 2.5].
    """
    model.eval()
    with torch.no_grad():
        baseline_spikes = []
        for _ in range(n_trials):
            _, info = model(torch.zeros(1, 21), return_all=True,
                           kc_inject_current=carbachol, apl_inject_current=0.0)
            baseline_spikes.append(info['kc_spikes'].sum().item())
        boosted_spikes = []
        for _ in range(n_trials):
            _, info = model(torch.zeros(1, 21), return_all=True,
                           kc_inject_current=carbachol, apl_inject_current=apl_inject)
            boosted_spikes.append(info['kc_spikes'].sum().item())
    baseline_mean = np.mean(baseline_spikes)
    boosted_mean = np.mean(boosted_spikes)
    ratio = baseline_mean / max(boosted_mean, 0.1)
    return {
        'ratio': ratio,
        'baseline_spikes': baseline_mean,
        'boosted_spikes': boosted_mean,
        'passes': 1.5 <= ratio <= 2.5,
    }


# ---------------------------------------------------------------------------
# Per-odor evaluation
# ---------------------------------------------------------------------------

def evaluate_per_odor(model, or_responses, odor_names, n_trials=20,
                      noise_std=0.3, seed=42):
    """Evaluate per-odor accuracy and sparsity.

    Returns (per_odor_acc, per_odor_sparsity, per_odor_kc) lists.
    """
    data = noisy_forward_pass(model, or_responses, n_trials, noise_std, seed)
    n_odors = len(odor_names)
    per_odor_acc, per_odor_sparsity, per_odor_kc = [], [], []
    for odor_idx in range(n_odors):
        logits = data['logits'][odor_idx]
        correct = sum(1 for l in logits if np.argmax(l) == odor_idx)
        per_odor_acc.append(correct / n_trials)
        per_odor_sparsity.append(np.mean(data['sparsities'][odor_idx]))
        kc_mean = torch.from_numpy(data['kc_patterns'][odor_idx]).float().mean(dim=0)
        per_odor_kc.append(kc_mean)
    return per_odor_acc, per_odor_sparsity, per_odor_kc


def evaluate_per_odor_all_models(models, or_responses, odor_names,
                                 n_trials=20, noise_std=0.3):
    """Evaluate per-odor metrics across ALL models.

    Returns (per_odor_acc, per_odor_sparsity, per_odor_kc) where acc/sparsity
    are lists of (mean, std) tuples and per_odor_kc is from the best model.
    """
    n_odors = len(odor_names)
    all_accs, all_sps = [], []
    for m_idx, model in enumerate(models):
        acc, sp, _ = evaluate_per_odor(model, or_responses, odor_names,
                                       n_trials, noise_std, seed=42 + m_idx)
        all_accs.append(acc)
        all_sps.append(sp)
    all_accs = np.array(all_accs)
    all_sps = np.array(all_sps)
    per_odor_acc = [(float(np.mean(all_accs[:, i])), float(np.std(all_accs[:, i])))
                    for i in range(n_odors)]
    per_odor_sparsity = [(float(np.mean(all_sps[:, i])), float(np.std(all_sps[:, i])))
                         for i in range(n_odors)]
    best_idx = np.argmax([np.mean(accs) for accs in all_accs])
    _, _, per_odor_kc = evaluate_per_odor(models[best_idx], or_responses, odor_names,
                                          n_trials, noise_std, seed=42 + best_idx)
    return per_odor_acc, per_odor_sparsity, per_odor_kc


# ---------------------------------------------------------------------------
# Cross-model consistency
# ---------------------------------------------------------------------------

def compute_cross_model_consistency(models, or_responses, odor_names,
                                    n_trials=10, noise_std=0.3):
    """Compute prediction consistency across models.

    Uses shared RNG across all models to ensure identical noise per trial.
    Returns (mean_consistency, per_odor_consistency_list).
    """
    rng = np.random.default_rng(12345)
    n_odors = len(odor_names)
    n_models = len(models)
    all_preds = np.zeros((n_odors, n_trials, n_models), dtype=int)
    for m_idx, model in enumerate(models):
        model.eval()
        with torch.no_grad():
            for odor_idx in range(n_odors):
                for trial in range(n_trials):
                    x = or_responses[odor_idx].clone()
                    noise = torch.from_numpy(
                        rng.normal(0, noise_std, x.shape).astype(np.float32))
                    x = torch.clamp(x * (1.0 + noise), min=0).unsqueeze(0)
                    logits = model(x)
                    all_preds[odor_idx, trial, m_idx] = logits.argmax(-1).item()
    consistency_per_odor = []
    for odor_idx in range(n_odors):
        agreements, total_pairs = 0, 0
        for trial in range(n_trials):
            preds = all_preds[odor_idx, trial, :]
            for i in range(n_models):
                for j in range(i + 1, n_models):
                    if preds[i] == preds[j]:
                        agreements += 1
                    total_pairs += 1
        consistency_per_odor.append(agreements / total_pairs if total_pairs > 0 else 0)
    return np.mean(consistency_per_odor), consistency_per_odor


def compute_kc_consistency_per_odor(models, or_responses, odor_names,
                                    n_trials=10, noise_std=0.3):
    """Compute KC pattern consistency per odor across models.

    Uses shared RNG. Returns (mean_consistency, list of (mean, std) tuples).
    """
    rng = np.random.default_rng(54321)
    n_odors = len(odor_names)
    n_models = len(models)
    consistency_per_odor = []
    for odor_idx in range(n_odors):
        all_kc = []
        for model in models:
            model.eval()
            kc_trials = []
            with torch.no_grad():
                for _ in range(n_trials):
                    x = or_responses[odor_idx].clone()
                    noise = torch.from_numpy(
                        rng.normal(0, noise_std, x.shape).astype(np.float32))
                    x = torch.clamp(x * (1.0 + noise), min=0).unsqueeze(0)
                    _, info = model(x, return_all=True)
                    kc_trials.append(info['kc_spikes'].squeeze(0))
            all_kc.append(torch.stack(kc_trials).mean(dim=0))
        kc_stack = torch.stack(all_kc)
        norms = kc_stack.norm(dim=1, keepdim=True).clamp(min=1e-8)
        normalized = kc_stack / norms
        corr_matrix = normalized @ normalized.T
        mask = torch.triu(torch.ones(n_models, n_models), diagonal=1).bool()
        pairwise = corr_matrix[mask].cpu().numpy()
        consistency_per_odor.append((float(np.mean(pairwise)), float(np.std(pairwise))))
    return np.mean([c[0] for c in consistency_per_odor]), consistency_per_odor


# ---------------------------------------------------------------------------
# Centroid accuracy
# ---------------------------------------------------------------------------

def centroid_accuracy(model, or_responses, n_trials=20, noise_std=0.3):
    """Compute centroid-based classification accuracy.

    Builds centroids from first half of trials, classifies second half.
    """
    data = noisy_forward_pass(model, or_responses, n_trials, noise_std, seed=12345)
    n_odors = len(or_responses)
    half = n_trials // 2
    centroids = np.array([data['kc_patterns'][i][:half].mean(axis=0)
                          for i in range(n_odors)])
    correct, total = 0, 0
    for odor_idx in range(n_odors):
        for trial in data['kc_patterns'][odor_idx][half:]:
            pred = centroid_classify(centroids, trial)
            if pred == -1:
                continue
            if pred == odor_idx:
                correct += 1
            total += 1
    return correct / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Circuit property extraction
# ---------------------------------------------------------------------------

def extract_gap_junction_info(model):
    """Extract gap junction conductances and LN->PN split.

    Returns (gap_info, ln_pn_split) dicts.
    """
    al = model.antennal_lobe
    gap_info = {}
    for attr, key in [('log_g_gap_ln', 'ln_ln'), ('log_g_gap_pn', 'pn_pn'),
                      ('log_g_gap_eln_pn', 'eln_pn')]:
        val = getattr(al, attr, None)
        gap_info[key] = float(torch.exp(val).item()) if val is not None else None

    ln_pn_split = {}
    if hasattr(al, 'ln_pn'):
        ln_pn_split['inhibitory_strength'] = float(torch.exp(al.ln_pn.log_strength).item())
    if hasattr(al, 'ln_pn_excit'):
        ln_pn_split['excitatory_strength'] = float(torch.exp(al.ln_pn_excit.log_strength).item())
    if hasattr(al, 'is_excitatory_ln'):
        ln_pn_split['n_excitatory_ln'] = int(al.is_excitatory_ln.sum().item())
        ln_pn_split['n_total_ln'] = len(al.is_excitatory_ln)

    return gap_info, ln_pn_split


def extract_nonad_strengths(model):
    """Extract non-AD connection strengths."""
    nonad = {}
    for name, layer in [
        ('orn_ln_nonad', getattr(model.antennal_lobe, 'orn_ln_nonad', None)),
        ('ln_pn_nonad', getattr(model.antennal_lobe, 'ln_pn_nonad', None)),
        ('ln_pn_excit_nonad', getattr(model.antennal_lobe, 'ln_pn_excit_nonad', None)),
        ('ln_ln_nonad', getattr(model.antennal_lobe, 'ln_ln_nonad', None)),
        ('pn_ln_nonad', getattr(model.antennal_lobe, 'pn_ln_nonad', None)),
        ('ln_orn_nonad', getattr(model.antennal_lobe, 'ln_orn_nonad', None)),
        ('pn_kc_nonad', getattr(model.kc_layer, 'pn_kc_nonad', None)),
    ]:
        if layer is not None:
            nonad[name] = float(np.exp(layer.log_strength.item()))
    return nonad


# ---------------------------------------------------------------------------
# Concentration invariance
# ---------------------------------------------------------------------------

def run_concentration_invariance(model, or_responses, seed, concentrations=None,
                                 hill_ec50=1.0, hill_n=1, n_trials=10, noise_std=0.3):
    """Run concentration invariance test for a single model.

    Tests: sublinear PN gain, flat KC activity, robust classification,
    odor identity preservation across 7 concentrations.

    Returns (conc_results, tests) dicts.
    """
    if concentrations is None:
        concentrations = [0.03, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0]

    sfr = or_responses[0]
    rng = np.random.default_rng(seed)
    model.eval()

    def get_patterns(scaled_resp):
        or_pats, pn_pats, kc_pats, kc_trials_all = [], [], [], []
        decoder_correct, decoder_total = 0, 0
        with torch.no_grad():
            for odor_idx in range(len(scaled_resp)):
                or_t, pn_t, kc_t = [], [], []
                true_label = odor_idx + 1
                for _ in range(n_trials):
                    base = scaled_resp[odor_idx]
                    noise = torch.from_numpy(
                        rng.normal(0, noise_std, base.shape).astype(np.float32))
                    x = (base * (1.0 + noise)).clamp(0)
                    logits, info = model(x.unsqueeze(0), return_all=True)
                    or_t.append(x.numpy())
                    pn_t.append(info['pn_spikes'].float().squeeze().numpy())
                    kc_t.append(info['kc_spikes'].float().squeeze().numpy())
                    if logits.argmax(-1).item() == true_label:
                        decoder_correct += 1
                    decoder_total += 1
                or_pats.append(np.mean(or_t, 0))
                pn_pats.append(np.mean(pn_t, 0))
                kc_pats.append(np.mean(kc_t, 0))
                kc_trials_all.append(np.array(kc_t))
        return (np.array(or_pats), np.array(pn_pats), np.array(kc_pats),
                kc_trials_all, decoder_correct / max(decoder_total, 1))

    baseline_eff = hill_effective_concentration(1.0, hill_ec50, hill_n)
    baseline_resp = sfr.unsqueeze(0) + (or_responses[1:] - sfr.unsqueeze(0)) * baseline_eff
    baseline_or, baseline_pn, baseline_kc, _, _ = get_patterns(baseline_resp)
    kc_centroids = baseline_kc
    sfr_np = sfr.numpy()

    def representation_similarity(pats_test, pats_base):
        return float(np.mean([cosine_sim_pair(pats_test[i], pats_base[i])
                              for i in range(len(pats_test))]))

    conc_results = {}
    for c in concentrations:
        eff = hill_effective_concentration(c, hill_ec50, hill_n)
        scaled = (sfr.unsqueeze(0) + (or_responses[1:] - sfr.unsqueeze(0)) * eff).clamp(0)
        or_p, pn_p, kc_p, kc_trials, dec_acc = get_patterns(scaled)

        # Centroid classification
        centroid_correct, centroid_total = 0, 0
        for odor_idx, trials in enumerate(kc_trials):
            for trial in trials:
                pred = centroid_classify(kc_centroids, trial)
                if pred == -1:
                    continue
                if pred == odor_idx:
                    centroid_correct += 1
                centroid_total += 1

        # Similarity to baseline
        or_test_sub = or_p - sfr_np
        or_base_sub = baseline_or - sfr_np
        conc_results[str(c)] = {
            'effective_c': eff, 'decoder_acc': dec_acc,
            'mean_or': float(np.mean([np.sum(p) for p in or_p])),
            'mean_pn': float(np.mean([np.sum(p) for p in pn_p])),
            'mean_kc': float(np.mean([np.sum(p) for p in kc_p])),
            'kc_centroid_acc': centroid_correct / max(centroid_total, 1),
            'or_sim': representation_similarity(or_test_sub, or_base_sub),
            'pn_sim': representation_similarity(pn_p, baseline_pn),
            'kc_sim': representation_similarity(kc_p, baseline_kc),
        }

    c_vals = [float(c) for c in concentrations]
    or_range = conc_results[str(c_vals[-1])]['mean_or'] / max(conc_results[str(c_vals[0])]['mean_or'], 1e-8)
    pn_range = conc_results[str(c_vals[-1])]['mean_pn'] / max(conc_results[str(c_vals[0])]['mean_pn'], 1e-8)
    kc_range = conc_results[str(c_vals[-1])]['mean_kc'] / max(conc_results[str(c_vals[0])]['mean_kc'], 1e-8)
    moderate_concs = [c for c in c_vals if 0.3 <= c <= 5.0]
    mean_moderate_acc = np.mean([conc_results[str(c)]['decoder_acc'] for c in moderate_concs])
    non_baseline_accs = [conc_results[str(c)]['kc_centroid_acc'] for c in c_vals if c != 1.0]
    mean_cross_conc_acc = np.mean(non_baseline_accs)

    tests = {
        'sublinear_pn_gain': bool(pn_range < or_range),
        'flat_kc_activity': bool(kc_range < 1.2),
        'robust_classification': bool(mean_moderate_acc > 0.5),
        'odor_identity_preservation': ('PASS' if mean_cross_conc_acc > 0.6
                                       else 'PARTIAL' if mean_cross_conc_acc > 0.3
                                       else 'FAIL'),
        'or_range': float(or_range), 'pn_range': float(pn_range), 'kc_range': float(kc_range),
        'mean_moderate_acc': float(mean_moderate_acc),
        'mean_cross_conc_centroid_acc': float(mean_cross_conc_acc),
    }
    return conc_results, tests




def compute_few_param_cv(all_params, min_params=10):
    """Compute coefficient of variation for few-parameter groups.

    Groups with >= min_params use Pearson correlation instead (see
    compute_pairwise_correlations). This handles the remaining small groups.
    """
    all_categories = set()
    for mp in all_params:
        all_categories.update(mp.keys())

    cv_results = {}
    for cat in sorted(all_categories):
        if cat in ('Total', 'Overall'):
            continue
        cat_params = [mp[cat].cpu().numpy() for mp in all_params if cat in mp]
        if len(cat_params) < 2:
            continue
        n_params = len(cat_params[0])
        if n_params >= min_params:
            continue
        stacked = np.array(cat_params)
        means = np.mean(stacked, axis=0)
        stds = np.std(stacked, axis=0)
        valid = np.abs(means) > 1e-10
        if valid.any():
            cvs = stds[valid] / np.abs(means[valid])
            cv_results[cat] = {
                'mean_cv': float(np.mean(cvs)),
                'per_element_cv': [float(c) for c in cvs],
                'n_params': n_params,
                'element_means': [float(m) for m in means],
                'element_stds': [float(s) for s in stds],
            }
    return cv_results
