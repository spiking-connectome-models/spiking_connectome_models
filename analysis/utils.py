"""Shared primitive functions for connectome model analysis.

Provides reusable building blocks used by compute.py and plotting.py:
  - Cosine similarity (pairwise, matrix)
  - Noisy forward pass (unified noise loop for all evaluations)
  - Per-pair similarity ratios (canonical decorrelation method)
  - Centroid classification
  - Hill dose-response
  - Parameter extraction and cross-model consistency metrics
"""

import numpy as np
import torch
from itertools import combinations
from scipy.stats import pearsonr


# ---------------------------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------------------------

def cosine_sim_pair(a, b):
    """Cosine similarity between two numpy vectors."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def cosine_sim_matrix(patterns):
    """Mean pairwise cosine similarity between pattern rows.

    Works on torch tensors. Replaces compute_cosine_similarity() and
    compute_mean_correlation() from earlier scripts.

    Args:
        patterns: (n_patterns, n_features) tensor

    Returns:
        Mean absolute cosine similarity across all off-diagonal pairs.
    """
    if patterns.dim() == 1:
        patterns = patterns.unsqueeze(0)
    if patterns.dim() > 2:
        patterns = patterns.reshape(patterns.shape[0], -1)
    n = patterns.shape[0]
    if n < 2:
        return 1.0
    norms = patterns.norm(dim=1, keepdim=True).clamp(min=1e-8)
    normalized = patterns / norms
    sim_matrix = normalized @ normalized.T
    mask = ~torch.eye(n, dtype=torch.bool, device=sim_matrix.device)
    off_diag = sim_matrix[mask]
    return off_diag.abs().mean().item()


# ---------------------------------------------------------------------------
# Noisy forward pass
# ---------------------------------------------------------------------------

def noisy_forward_pass(model, or_responses, n_trials, noise_std, seed):
    """Run model on all odors with multiplicative noise, collecting per-trial results.

    This is the unified noise loop used by compute_per_pair_decorrelation,
    evaluate_per_odor, and centroid_accuracy. Each uses its own seed for
    reproducibility.

    Args:
        model: SpikingConnectomeConstrainedModel (set to eval mode internally)
        or_responses: (n_odors, n_or) tensor of baseline OR responses
        n_trials: number of noisy trials per odor
        noise_std: multiplicative noise CV (e.g. 0.3 for 30%)
        seed: random seed for reproducibility

    Returns:
        dict with per-odor lists:
            or_patterns: list of (n_trials, n_or) numpy arrays
            pn_patterns: list of (n_trials, n_pn) numpy arrays
            kc_patterns: list of (n_trials, n_kc) numpy arrays
            logits: list of (n_trials, n_classes) numpy arrays
            sparsities: list of n_trials-length lists
    """
    rng = np.random.default_rng(seed)
    model.eval()
    n_odors = len(or_responses)
    all_or, all_pn, all_kc, all_logits, all_sp = [], [], [], [], []
    with torch.no_grad():
        for odor_idx in range(n_odors):
            or_t, pn_t, kc_t, logit_t, sp_t = [], [], [], [], []
            for _ in range(n_trials):
                base = or_responses[odor_idx]
                noise = torch.from_numpy(
                    rng.normal(0, noise_std, base.shape).astype(np.float32)
                )
                x = (base * (1.0 + noise)).clamp(0)
                logits, info = model(x.unsqueeze(0), return_all=True)
                or_t.append(x.numpy())
                pn_t.append(info['pn_spikes'].float().squeeze().numpy())
                kc_t.append(info['kc_spikes'].float().squeeze().numpy())
                logit_t.append(logits.squeeze().numpy())
                sp_t.append(info['sparsity'])
            all_or.append(np.array(or_t))
            all_pn.append(np.array(pn_t))
            all_kc.append(np.array(kc_t))
            all_logits.append(np.array(logit_t))
            all_sp.append(sp_t)
    return {
        'or_patterns': all_or,
        'pn_patterns': all_pn,
        'kc_patterns': all_kc,
        'logits': all_logits,
        'sparsities': all_sp,
    }


# ---------------------------------------------------------------------------
# Per-pair similarity ratios (canonical decorrelation)
# ---------------------------------------------------------------------------

def per_pair_similarity_ratios(or_pats, pn_pats, kc_pats):
    """Compute per-pair cosine similarity ratios.

    This is the canonical decorrelation method: for each odor pair, compute
    the ratio of downstream similarity to upstream similarity.

    Args:
        or_pats: (n_odors, n_features) mean OR patterns (numpy)
        pn_pats: (n_odors, n_features) mean PN patterns (numpy)
        kc_pats: (n_odors, n_features) mean KC patterns (numpy)

    Returns:
        dict with kc_or_ratios, pn_or_ratios, kc_pn_ratios (lists of floats)
    """
    n_odors = len(or_pats)
    kc_or_ratios, pn_or_ratios, kc_pn_ratios = [], [], []
    for i in range(n_odors):
        for j in range(i + 1, n_odors):
            or_sim = cosine_sim_pair(or_pats[i], or_pats[j])
            pn_sim = cosine_sim_pair(pn_pats[i], pn_pats[j])
            kc_sim = cosine_sim_pair(kc_pats[i], kc_pats[j])
            if abs(or_sim) > 1e-8:
                kc_or_ratios.append(kc_sim / or_sim)
                pn_or_ratios.append(pn_sim / or_sim)
            if abs(pn_sim) > 1e-8:
                kc_pn_ratios.append(kc_sim / pn_sim)
    return {
        'kc_or_ratios': kc_or_ratios,
        'pn_or_ratios': pn_or_ratios,
        'kc_pn_ratios': kc_pn_ratios,
    }


# ---------------------------------------------------------------------------
# Centroid classification
# ---------------------------------------------------------------------------

def centroid_classify(centroids, test_trial):
    """Classify a single trial by nearest centroid (cosine similarity).

    Args:
        centroids: (n_classes, n_features) numpy array
        test_trial: (n_features,) numpy array

    Returns:
        Predicted class index, or -1 if trial has near-zero norm.
    """
    norm_t = np.linalg.norm(test_trial)
    if norm_t < 1e-8:
        return -1
    sims = [
        np.dot(test_trial, c) / (norm_t * max(np.linalg.norm(c), 1e-8))
        for c in centroids
    ]
    return int(np.argmax(sims))


# ---------------------------------------------------------------------------
# Hill dose-response
# ---------------------------------------------------------------------------

def hill_effective_concentration(c, ec50=1.0, n=1):
    """Hill dose-response function for concentration scaling.

    Returns the effective concentration relative to reference (c=1.0).
    """
    response_at_c = (c ** n) / (c ** n + ec50 ** n)
    response_at_ref = 1.0 / (1.0 + ec50 ** n)
    return response_at_c / response_at_ref


# ---------------------------------------------------------------------------
# Parameter extraction
# ---------------------------------------------------------------------------

def extract_all_parameters(model):
    """Extract learnable parameters by category (including non-AD and gap junctions).

    Returns dict mapping category name -> concatenated parameter tensor.
    Also includes a 'Total' key with all parameters concatenated.
    """
    params = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        flat = param.data.clone().flatten()

        # OR→ORN
        if 'or_to_orn' in name or 'or_gains' in name:
            params.setdefault('OR→ORN', []).append(flat)

        # Antennal lobe
        elif 'antennal_lobe' in name:
            if 'log_or_gain' in name:
                params.setdefault('OR gain', []).append(flat)
            elif 'orn_neurons' in name:
                if 'v_th' in name:
                    params.setdefault('ORN v_th', []).append(flat)
                elif 'tau' in name or 'log_tau' in name:
                    params.setdefault('ORN τ_m', []).append(flat)
            elif 'ln_neurons' in name:
                if 'v_th' in name:
                    params.setdefault('LN v_th', []).append(flat)
                elif 'tau' in name or 'log_tau' in name:
                    params.setdefault('LN τ_m', []).append(flat)
            elif 'pn_neurons' in name:
                if 'v_th' in name:
                    params.setdefault('PN v_th', []).append(flat)
                elif 'tau' in name or 'log_tau' in name:
                    params.setdefault('PN τ_m', []).append(flat)
            elif 'orn_pn' in name or 'orn_to_pn' in name:
                key = 'ORN→PN nonad' if 'nonad' in name else 'ORN→PN'
                params.setdefault(key, []).append(flat)
            elif 'orn_ln' in name or 'orn_to_ln' in name:
                key = 'ORN→LN nonad' if 'nonad' in name else 'ORN→LN'
                params.setdefault(key, []).append(flat)
            elif 'ln_pn_excit' in name:
                key = 'LN→PN excit nonad' if 'nonad' in name else 'LN→PN excit'
                params.setdefault(key, []).append(flat)
            elif 'ln_pn' in name or 'ln_to_pn' in name:
                key = 'LN→PN nonad' if 'nonad' in name else 'LN→PN'
                params.setdefault(key, []).append(flat)
            elif 'ln_ln' in name:
                key = 'LN→LN nonad' if 'nonad' in name else 'LN→LN'
                params.setdefault(key, []).append(flat)
            elif 'pn_ln' in name:
                key = 'PN→LN nonad' if 'nonad' in name else 'PN→LN'
                params.setdefault(key, []).append(flat)
            elif 'ln_orn' in name:
                key = 'LN→ORN nonad' if 'nonad' in name else 'LN→ORN'
                params.setdefault(key, []).append(flat)
            elif 'log_g_gap' in name:
                if 'ln' in name and 'pn' not in name:
                    params.setdefault('Gap LN-LN', []).append(flat)
                elif 'pn' in name and 'eln' not in name:
                    params.setdefault('Gap PN-PN', []).append(flat)
                elif 'eln' in name:
                    params.setdefault('Gap eLN-PN', []).append(flat)

        # KC layer
        elif 'kc_layer' in name:
            if 'pn_kc' in name or 'pn_to_kc' in name:
                key = 'PN→KC nonad' if 'nonad' in name else 'PN→KC'
                params.setdefault(key, []).append(flat)
            elif 'kc_dend_gain' in name:
                params.setdefault('KC dend gain', []).append(flat)
            elif 'kc_neurons' in name:
                if 'v_th' in name:
                    params.setdefault('KC v_th', []).append(flat)
                elif 'g_soma' in name:
                    params.setdefault('KC g_soma', []).append(flat)
            elif 'kc_apl_log_strength' in name:
                # Store in real space (exp) since log-space CV is misleading when mean≈0
                params.setdefault('KC→APL strength', []).append(torch.exp(flat))
            elif 'kc_apl' in name:
                params.setdefault('KC→APL', []).append(flat)
            elif 'apl_kc' in name:
                params.setdefault('APL→KC', []).append(flat)
            elif 'kc_kc_aa' in name:
                params.setdefault('KC-KC aa', []).append(flat)
            elif 'kc_kc_ad' in name:
                params.setdefault('KC-KC ad', []).append(flat)
            elif 'apl_gain' in name:
                params.setdefault('APL gain', []).append(flat)
            elif 'apl' in name and 'tau' in name:
                params.setdefault('APL τ', []).append(flat)

        # Decoder
        elif 'decoder' in name:
            params.setdefault('Decoder', []).append(flat)

    # Concatenate lists
    for key in params:
        if params[key]:
            params[key] = torch.cat(params[key])

    # Create Total
    all_p = [p for p in params.values()]
    if all_p:
        params['Total'] = torch.cat(all_p)

    return params


# ---------------------------------------------------------------------------
# Cross-model parameter consistency
# ---------------------------------------------------------------------------

def compute_pairwise_correlations(models_params):
    """Compute all pairwise Pearson correlations between models for each parameter category.

    Args:
        models_params: list of dicts from extract_all_parameters()

    Returns:
        dict mapping category -> list of pairwise correlation values
    """
    n_models = len(models_params)
    all_categories = set()
    for mp in models_params:
        all_categories.update(mp.keys())

    correlations = {}

    # Overall correlation (all parameters concatenated)
    all_params_per_model = []
    for mp in models_params:
        all_flat = [mp[cat].cpu().numpy() for cat in sorted(all_categories) if cat in mp]
        if all_flat:
            all_params_per_model.append(np.concatenate(all_flat))

    overall_corrs = []
    for i, j in combinations(range(n_models), 2):
        if len(all_params_per_model[i]) == len(all_params_per_model[j]):
            r, _ = pearsonr(all_params_per_model[i], all_params_per_model[j])
            if not np.isnan(r):
                overall_corrs.append(r)
    if overall_corrs:
        correlations['Overall'] = overall_corrs

    # Per-category correlations
    for cat in sorted(all_categories):
        cat_params = [mp[cat].cpu().numpy() for mp in models_params if cat in mp]
        if (len(cat_params) >= 2 and len(cat_params[0]) >= 2
                and all(len(p) == len(cat_params[0]) for p in cat_params)):
            cat_corrs = []
            for i, j in combinations(range(len(cat_params)), 2):
                r, _ = pearsonr(cat_params[i], cat_params[j])
                if not np.isnan(r):
                    cat_corrs.append(r)
            if cat_corrs:
                correlations[cat] = cat_corrs

    return correlations


def analyze_biological_parameters(model):
    """Analyze learned biological parameters (v_th, g_soma) for realism."""
    from spiking_connectome_models.layers import V_TH_MIN, V_TH_MAX, G_SOMA_MIN, G_SOMA_MAX

    results = {}
    all_vth = []
    vth_populations = {}

    for pop_name, neurons in [
        ('ORN', model.antennal_lobe.orn_neurons),
        ('LN', model.antennal_lobe.ln_neurons),
        ('PN', model.antennal_lobe.pn_neurons),
        ('KC', model.kc_layer.kc_neurons),
    ]:
        vth = neurons.v_th.detach().cpu().numpy()
        all_vth.extend(vth.tolist())
        vth_populations[pop_name] = vth

    all_vth = np.array(all_vth)
    eps = 1e-9
    in_bounds = np.sum((all_vth >= V_TH_MIN - eps) & (all_vth <= V_TH_MAX + eps))

    results['v_th'] = {
        'total_neurons': len(all_vth),
        'in_bounds': int(in_bounds),
        'pct_in_bounds': 100 * in_bounds / len(all_vth),
        'min_mV': float(all_vth.min() * 1000),
        'max_mV': float(all_vth.max() * 1000),
        'mean_mV': float(all_vth.mean() * 1000),
        'populations': {
            name: {
                'n': len(v),
                'min_mV': float(v.min() * 1000),
                'max_mV': float(v.max() * 1000),
                'mean_mV': float(v.mean() * 1000),
            }
            for name, v in vth_populations.items()
        },
    }

    if hasattr(model.kc_layer.kc_neurons, 'g_soma'):
        g_soma = model.kc_layer.kc_neurons.g_soma.item()
        results['g_soma'] = {
            'value_nS': float(g_soma * 1e9),
            'in_bounds': G_SOMA_MIN <= g_soma <= G_SOMA_MAX,
            'bounds_nS': [float(G_SOMA_MIN * 1e9), float(G_SOMA_MAX * 1e9)],
        }

    return results


