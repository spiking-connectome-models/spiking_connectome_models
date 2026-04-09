"""Plotting functions for connectome model analysis.

Each function:
  - Returns the matplotlib Figure object
  - Takes show=False kwarg (True keeps figure open for Jupyter display)
  - Takes output_path=None kwarg (saves to file if provided)
  - Does NOT set matplotlib.use('Agg') (caller's responsibility)
  - Accepts constants (seeds, concentrations, etc.) as arguments
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from matplotlib.colors import SymLogNorm
from matplotlib.ticker import FixedLocator, FixedFormatter

from .utils import hill_effective_concentration


# ---------------------------------------------------------------------------
# Training curves (S1)
# ---------------------------------------------------------------------------

def plot_training_curves(histories, teacher_accs, seeds=None,
                         output_path=None, show=False):
    """Plot training accuracy and sparsity curves for all models."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    for i, hist in enumerate(histories):
        epochs = hist['epochs']
        accs = [a * 100 for a in hist['test_acc']]
        ax.plot(epochs, accs, label=f'Spiking {i+1} (teacher: {teacher_accs[i]:.1%})', alpha=0.8)
    ax.axhline(y=np.mean(teacher_accs) * 100, color='red', linestyle='--',
               label=f'Mean teacher ({np.mean(teacher_accs):.1%})', linewidth=2)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy (%)')
    ax.set_title('Training Accuracy'); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1]
    for i, hist in enumerate(histories):
        epochs = hist['epochs']
        sps = [s * 100 for s in hist['sparsity']]
        ax.plot(epochs, sps, label=f'Model {i+1}', alpha=0.8)
    ax.axhline(y=10, color='green', linestyle='--', label='Target (10%)', linewidth=2)
    ax.set_xlabel('Epoch'); ax.set_ylabel('KC Sparsity (%)')
    ax.set_title('KC Sparsity'); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
    if not show:
        plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# Per-odor breakdown (diagnostic)
# ---------------------------------------------------------------------------

def plot_per_odor_breakdown(per_odor_acc, per_odor_sparsity, odor_names,
                            output_path=None, show=False):
    """Plot per-odor accuracy and sparsity (horizontal bars)."""
    acc_means = [c[0] for c in per_odor_acc]
    acc_stds = [c[1] for c in per_odor_acc]
    sp_means = [c[0] for c in per_odor_sparsity]
    sp_stds = [c[1] for c in per_odor_sparsity]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))
    sorted_idx = np.argsort(acc_means)[::-1]
    sorted_acc = [acc_means[i] * 100 for i in sorted_idx]
    sorted_acc_std = [acc_stds[i] * 100 for i in sorted_idx]
    sorted_sp = [sp_means[i] * 100 for i in sorted_idx]
    sorted_sp_std = [sp_stds[i] * 100 for i in sorted_idx]
    sorted_names = [odor_names[i] for i in sorted_idx]
    y_pos = np.arange(len(sorted_names))

    colors = ['#2ecc71' if a > 70 else '#f39c12' if a > 50 else '#e74c3c' for a in sorted_acc]
    ax1.barh(y_pos, sorted_acc, xerr=sorted_acc_std, color=colors, edgecolor='black',
             alpha=0.8, capsize=2, error_kw={'linewidth': 1})
    ax1.axvline(x=np.mean(acc_means) * 100, color='blue', linestyle='--',
                label=f'Mean: {np.mean(acc_means)*100:.1f}%')
    ax1.set_yticks(y_pos); ax1.set_yticklabels(sorted_names, fontsize=9)
    ax1.set_xlabel('Accuracy (%)'); ax1.set_title('Per-Odor Accuracy (n=5 models)'); ax1.legend()

    colors = ['#3498db' if 5 <= s <= 20 else '#e74c3c' for s in sorted_sp]
    ax2.barh(y_pos, sorted_sp, xerr=sorted_sp_std, color=colors, edgecolor='black',
             alpha=0.8, capsize=2, error_kw={'linewidth': 1})
    ax2.axvline(x=10, color='green', linestyle='--', label='Target: 10%')
    ax2.axvline(x=np.mean(sp_means) * 100, color='blue', linestyle='--',
                label=f'Mean: {np.mean(sp_means)*100:.1f}%')
    ax2.set_yticks(y_pos); ax2.set_yticklabels([]); ax2.set_xlabel('KC Sparsity (%)')
    ax2.set_title('Per-Odor KC Sparsity (n=5 models)'); ax2.legend()

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
    if not show:
        plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# KC sparsity distribution (diagnostic)
# ---------------------------------------------------------------------------

def plot_kc_sparsity_distribution(per_odor_kc, output_path=None, show=False):
    """Plot KC activity heatmap and histogram."""
    if isinstance(per_odor_kc, list):
        kc_activity = torch.stack(per_odor_kc).detach().cpu().numpy()
    else:
        kc_activity = per_odor_kc.detach().cpu().numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    kc_activity_for_plot = kc_activity + 0.1
    vmin = 0.1
    vmax = min(max(kc_activity.max(), 1.0), 10.0)
    im = ax1.imshow(kc_activity_for_plot, aspect='auto', cmap='viridis',
                    norm=mcolors.LogNorm(vmin=vmin, vmax=vmax))
    ax1.set_xlabel('KC Index'); ax1.set_ylabel('Odor Index')
    ax1.set_title('KC Activity Pattern per Odor (Best Model, log scale)')
    cbar = plt.colorbar(im, ax=ax1, label='Mean Spike Count (n=20 trials)')
    cbar.locator = FixedLocator([0.1, 0.5, 1, 2, 5, 10])
    cbar.formatter = FixedFormatter(['0', '0.5', '1', '2', '5', '10'])
    cbar.update_ticks()

    ax2.hist(kc_activity.flatten(), bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    ax2.set_xlabel('KC Activity (Spike Count)'); ax2.set_ylabel('Count')
    ax2.set_title('Distribution of KC Activity')
    ax2.axvline(x=0, color='red', linestyle='--', label='Silent'); ax2.legend()

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
    if not show:
        plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# Biological parameters (diagnostic)
# ---------------------------------------------------------------------------

def plot_biological_parameters(bio_params, output_path=None, show=False):
    """Plot v_th distribution by population and g_soma."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    populations = bio_params['v_th']['populations']
    pop_names = list(populations.keys())
    means = [populations[p]['mean_mV'] for p in pop_names]
    mins = [populations[p]['min_mV'] for p in pop_names]
    maxs = [populations[p]['max_mV'] for p in pop_names]
    x = np.arange(len(pop_names))
    ax.bar(x, means, color='steelblue', edgecolor='black', alpha=0.7)
    ax.errorbar(x, means, yerr=[np.array(means) - np.array(mins),
                                 np.array(maxs) - np.array(means)],
                fmt='none', color='black', capsize=5)
    ax.axhline(y=-55, color='red', linestyle='--', linewidth=2, label='Bio min (-55mV)')
    ax.axhline(y=-30, color='red', linestyle='--', linewidth=2, label='Bio max (-30mV)')
    ax.axhspan(-55, -30, alpha=0.1, color='green', label='Biological range')
    ax.set_xticks(x); ax.set_xticklabels(pop_names); ax.set_ylabel('Threshold (mV)')
    ax.set_title(f"v_th Distribution by Population\n"
                 f"({bio_params['v_th']['pct_in_bounds']:.0f}% in biological bounds)")
    ax.legend(loc='upper right', fontsize=8)

    ax = axes[1]
    if 'g_soma' in bio_params:
        g_val = bio_params['g_soma']['value_nS']
        g_std = bio_params['g_soma'].get('std_nS', 0)
        bounds = bio_params['g_soma']['bounds_nS']
        ax.bar(['g_soma'], [g_val], yerr=[g_std] if g_std > 0 else None, capsize=5,
               color='coral' if bio_params['g_soma']['in_bounds'] else 'red', edgecolor='black')
        ax.axhline(y=bounds[0], color='green', linestyle='--', linewidth=2,
                   label=f'Min ({bounds[0]:.0f} nS)')
        ax.axhline(y=bounds[1], color='green', linestyle='--', linewidth=2,
                   label=f'Max ({bounds[1]:.0f} nS)')
        ax.axhspan(bounds[0], bounds[1], alpha=0.1, color='green')
        ax.set_ylabel('Conductance (nS)')
        n_models = len(bio_params['g_soma'].get('all_values_nS', [1]))
        ax.set_title(f"Learned g_soma (n={n_models})\n({g_val:.1f}+-{g_std:.1f} nS)")
        ax.legend(loc='upper right')
        ax.text(0, g_val + g_std + 2, f'{g_val:.1f}+-{g_std:.1f} nS',
                ha='center', fontweight='bold')

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
    if not show:
        plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# Parameter correlation bars (S2a)
# ---------------------------------------------------------------------------

def plot_correlation_bars(correlations, title, all_params=None,
                          min_params_for_correlation=10,
                          output_path=None, show=False):
    """Plot Pearson correlation bars per parameter category.

    Groups with < min_params_for_correlation are excluded (use plot_few_param_cv).
    """
    exclude_cats = set()
    if all_params:
        ref_params = all_params[0]
        for cat in ref_params:
            if cat in ('Total', 'Overall'):
                continue
            if len(ref_params[cat]) < min_params_for_correlation:
                exclude_cats.add(cat)

    categories, means, stds = [], [], []
    if 'Overall' in correlations:
        categories.append('Overall')
        means.append(np.mean(correlations['Overall']))
        stds.append(np.std(correlations['Overall']))
    for cat in sorted(correlations.keys()):
        if cat != 'Overall' and cat not in exclude_cats:
            categories.append(cat)
            means.append(np.mean(correlations[cat]))
            stds.append(np.std(correlations[cat]))

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(categories))
    colors = ['#2ecc71' if cat == 'Overall' else '#3498db' for cat in categories]
    ax.bar(x, means, yerr=stds, capsize=5, color=colors, edgecolor='black', alpha=0.8)
    ax.set_ylabel('Pairwise Pearson Correlation', fontsize=12)
    ax.set_xlabel('Parameter Category', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x); ax.set_xticklabels(categories, rotation=45, ha='right', fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(axis='y', alpha=0.3)
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 0.03, f'{m:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    if not show:
        plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# Few-parameter CV (S2b)
# ---------------------------------------------------------------------------

def plot_few_param_cv(cv_results, title, output_path=None, show=False):
    """Plot coefficient of variation for few-parameter groups."""
    if not cv_results:
        return None

    categories = sorted(cv_results.keys())
    cvs = [cv_results[cat]['mean_cv'] for cat in categories]
    n_params = [cv_results[cat]['n_params'] for cat in categories]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(categories))
    bars = ax.bar(x, cvs, color='#e67e22', edgecolor='black', alpha=0.8)
    for i, (bar, cv, n) in enumerate(zip(bars, cvs, n_params)):
        ax.text(bar.get_x() + bar.get_width() / 2., cv + 0.005,
                f'{cv:.3f}\n({n}p)', ha='center', va='bottom', fontsize=9)
    ax.set_ylabel('Coefficient of Variation (CV = std/|mean|)', fontsize=12)
    ax.set_xlabel('Parameter Group', fontsize=12); ax.set_title(title, fontsize=14)
    ax.set_xticks(x); ax.set_xticklabels(categories, rotation=45, ha='right', fontsize=10)
    ax.set_ylim(0, max(cvs) * 1.4 if cvs else 1.0); ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.axhline(y=0.05, color='green', linestyle='--', alpha=0.4,
               label='CV=0.05 (very consistent)')
    ax.axhline(y=0.20, color='orange', linestyle='--', alpha=0.4,
               label='CV=0.20 (moderate variation)')
    ax.legend(loc='upper right', fontsize=9)

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    if not show:
        plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# Mancini validation (S3)
# ---------------------------------------------------------------------------

def plot_mancini_validation(mancini_results, seeds, output_path=None, show=False):
    """Plot per-seed Mancini APL inhibition ratios."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ratios = [r['ratio'] for r in mancini_results]
    passes = [r['passes'] for r in mancini_results]
    colors = ['#2ecc71' if p else '#e74c3c' for p in passes]
    x = np.arange(len(seeds))
    ax.bar(x, ratios, color=colors, edgecolor='black', alpha=0.8)
    ax.axhline(y=2.0, color='blue', linestyle='--', linewidth=2, label='Target (2.0)')
    ax.axhspan(1.5, 2.5, alpha=0.1, color='green', label='Acceptable range (1.5-2.5)')
    ax.axhline(y=1.5, color='green', linestyle='--', linewidth=1, alpha=0.5)
    ax.axhline(y=2.5, color='green', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xticks(x); ax.set_xticklabels([f'Seed {s}' for s in seeds])
    ax.set_ylabel('Mancini Ratio (baseline/boosted)', fontsize=12)
    ax.set_title(f'APL Inhibition Validation (Mancini et al. 2023)\n'
                 f'Mean: {np.mean(ratios):.2f} +- {np.std(ratios):.2f}', fontsize=14)
    ax.legend(loc='upper right'); ax.set_ylim(0, 3.0)
    for i, r in enumerate(ratios):
        ax.text(i, r + 0.1, f'{r:.2f}', ha='center', fontsize=10)
    ax.text(0.02, 0.02,
            'Mancini et al. 2023: APL activation reduces KC calcium to ~50%\n'
            'Ratio = KC spikes without APL / KC spikes with APL',
            transform=ax.transAxes, fontsize=9, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
    if not show:
        plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# Gap junction conductances (diagnostic)
# ---------------------------------------------------------------------------

def plot_gap_junction_conductances(all_gap_info, seeds, output_path=None, show=False):
    """Plot gap junction conductances per seed."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    gap_types = [
        ('ln_ln', 'LN-LN', '#8e44ad'),
        ('pn_pn', 'PN-PN', '#2980b9'),
        ('eln_pn', 'eLN-PN', '#27ae60'),
    ]
    for ax, (key, title, color) in zip(axes, gap_types):
        vals = [g[key] for g in all_gap_info if g[key] is not None]
        if not vals:
            ax.set_title(f'{title}\n(not present)')
            continue
        vals_ps = [v * 1e12 for v in vals]
        x = np.arange(len(vals))
        ax.bar(x, vals_ps, color=color, edgecolor='black', alpha=0.8)
        ax.axhline(y=np.mean(vals_ps), color='red', linestyle='--',
                   label=f'Mean: {np.mean(vals_ps):.1f} pS', linewidth=2)
        ax.set_xticks(x); ax.set_xticklabels([f'Seed {s}' for s in seeds[:len(vals)]])
        ax.set_ylabel('Conductance (pS)')
        ax.set_title(f'{title}\n({np.mean(vals_ps):.1f} +- {np.std(vals_ps):.1f} pS)')
        ax.legend(fontsize=9)
        for i, v in enumerate(vals_ps):
            ax.text(i, v + 1, f'{v:.1f}', ha='center', fontsize=9)

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
    if not show:
        plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# LN->PN split (diagnostic)
# ---------------------------------------------------------------------------

def plot_ln_pn_split(all_ln_pn_split, seeds, output_path=None, show=False):
    """Plot LN->PN inhibitory vs excitatory strengths."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    inhib_vals = [s['inhibitory_strength'] * 1e9 for s in all_ln_pn_split]
    excit_vals = [s['excitatory_strength'] * 1e9 for s in all_ln_pn_split]
    x = np.arange(len(seeds)); width = 0.35
    ax1.bar(x - width/2, inhib_vals, width, color='#c0392b', edgecolor='black',
            alpha=0.8, label='Inhibitory')
    ax1.bar(x + width/2, excit_vals, width, color='#27ae60', edgecolor='black',
            alpha=0.8, label='Excitatory')
    ax1.set_xticks(x); ax1.set_xticklabels([f'Seed {s}' for s in seeds])
    ax1.set_ylabel('Synaptic Strength (nA)')
    ax1.set_title(f'LN->PN Pathway Strengths\nInhib: {np.mean(inhib_vals):.3f} nA, '
                  f'Excit: {np.mean(excit_vals)*1e3:.3f} pA')
    ax1.legend(); ax1.set_yscale('log')

    ratios = [i / max(e, 1e-15) for i, e in zip(inhib_vals, excit_vals)]
    ax2.bar(x, ratios, color='#f39c12', edgecolor='black', alpha=0.8)
    ax2.axhline(y=np.mean(ratios), color='red', linestyle='--',
               label=f'Mean ratio: {np.mean(ratios):.0f}x', linewidth=2)
    ax2.set_xticks(x); ax2.set_xticklabels([f'Seed {s}' for s in seeds])
    ax2.set_ylabel('Inhibitory / Excitatory Ratio')
    ax2.set_title(f'LN->PN Inhibition Dominance\n(Ratio: {np.mean(ratios):.0f}x)')
    ax2.legend()

    if 'n_excitatory_ln' in all_ln_pn_split[0]:
        n_excit = all_ln_pn_split[0]['n_excitatory_ln']
        n_total = all_ln_pn_split[0]['n_total_ln']
        fig.text(0.5, 0.01, f'LN subtypes: {n_excit}/{n_total} excitatory (Picky), '
                f'{n_total - n_excit}/{n_total} inhibitory (Broad/Choosy)',
                ha='center', fontsize=10, style='italic')

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
    if not show:
        plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# Core figure: similarity matrices + KC heatmap (Figure 2)
# ---------------------------------------------------------------------------

def plot_core_figure(or_sim, pn_sim, kc_sim, kc_activity=None,
                     output_path=None, show=False):
    """Figure 2: 1x3 — A (OR sim), B (PN sim), C (KC sim).

    Parameters
    ----------
    or_sim, pn_sim, kc_sim : ndarray (n_odors, n_odors)
        Pairwise cosine similarity matrices.
    kc_activity : ignored (kept for backward compatibility).
    """
    n_odors = or_sim.shape[0]

    mat_size = n_odors           # 28
    gap_top = 6
    cb_gap = 3
    cb_w = 2

    pix = 0.12
    top_w = 3 * mat_size + 2 * gap_top + cb_gap + cb_w
    fig_w = top_w * pix + 3.0
    fig_h = mat_size * pix + 3.0

    fig = plt.figure(figsize=(fig_w, fig_h))

    margin_l = 1.4 / fig_w
    margin_b = 1.4 / fig_h
    mat_w_frac = mat_size * pix / fig_w
    mat_h_frac = mat_size * pix / fig_h
    gap_top_frac = gap_top * pix / fig_w
    cb_gap_frac = cb_gap * pix / fig_w
    cb_w_frac = cb_w * pix / fig_w

    x = margin_l
    ax_a = fig.add_axes([x, margin_b, mat_w_frac, mat_h_frac])
    x += mat_w_frac + gap_top_frac
    ax_b = fig.add_axes([x, margin_b, mat_w_frac, mat_h_frac])
    x += mat_w_frac + gap_top_frac
    ax_c = fig.add_axes([x, margin_b, mat_w_frac, mat_h_frac])
    x += mat_w_frac + cb_gap_frac
    ax_cb = fig.add_axes([x, margin_b, cb_w_frac, mat_h_frac])

    title_fs, label_fs, tick_fs, letter_fs, cb_fs = 20, 18, 16, 20, 16

    odor_ticks = list(range(n_odors))
    odor_labels = [str(i + 1) for i in range(n_odors)]

    vmin, vmax = -0.2, 1.0
    for ax, mat, title in [(ax_a, or_sim, 'OR Similarity'),
                            (ax_b, pn_sim, 'PN Similarity'),
                            (ax_c, kc_sim, 'KC Similarity')]:
        im = ax.imshow(mat, cmap='RdYlBu_r', vmin=vmin, vmax=vmax, aspect='equal')
        ax.set_title(title, fontsize=title_fs, fontweight='bold', pad=10)
        ax.set_xlabel('Odor', fontsize=label_fs)
        ax.set_ylabel('Odor', fontsize=label_fs)
        ax.set_xticks(odor_ticks[::5])
        ax.set_xticklabels([odor_labels[i] for i in odor_ticks[::5]], fontsize=tick_fs)
        ax.set_yticks(odor_ticks[::5])
        ax.set_yticklabels([odor_labels[i] for i in odor_ticks[::5]], fontsize=tick_fs)

    cb = fig.colorbar(im, cax=ax_cb)
    cb.set_label('Cosine Similarity', fontsize=cb_fs)
    cb.ax.tick_params(labelsize=tick_fs)

    _letter_kw = dict(fontsize=letter_fs, fontweight='bold', va='bottom', ha='left')
    for ax, letter in [(ax_a, 'A'), (ax_b, 'B'), (ax_c, 'C')]:
        ax.text(-0.02, 1.05, letter, transform=ax.transAxes, **_letter_kw)

    if output_path:
        fig.savefig(str(output_path), dpi=200, bbox_inches='tight')
        pdf_path = str(output_path).replace('.png', '.pdf')
        if pdf_path != str(output_path):
            fig.savefig(pdf_path, dpi=300, bbox_inches='tight')
    if not show:
        plt.close(fig)
    return fig


def plot_kc_heatmap(kc_activity, output_path=None, show=False):
    """KC activity heatmap (separate from Figure 2).

    Parameters
    ----------
    kc_activity : ndarray (n_odors, n_kcs)
        Mean KC firing rates per odor.
    """
    kc_data = kc_activity.detach().cpu().numpy() if hasattr(kc_activity, 'detach') else np.array(kc_activity)
    n_odors = kc_data.shape[0]

    fig, ax = plt.subplots(figsize=(10, 4))

    title_fs, label_fs, tick_fs, cb_fs = 20, 18, 16, 16
    odor_ticks = list(range(n_odors))
    odor_labels = [str(i + 1) for i in range(n_odors)]

    vmax_kc = kc_data.max()
    norm = mcolors.SymLogNorm(linthresh=0.1, vmin=0, vmax=vmax_kc)
    im = ax.imshow(kc_data, aspect='auto', cmap='viridis', norm=norm,
                   interpolation='nearest')
    ax.set_title('KC Activity Heatmap', fontsize=title_fs, fontweight='bold', pad=10)
    ax.set_xlabel('KC Index', fontsize=label_fs)
    ax.set_ylabel('Odor', fontsize=label_fs)
    ax.set_yticks(odor_ticks[::5])
    ax.set_yticklabels([odor_labels[i] for i in odor_ticks[::5]], fontsize=tick_fs)
    ax.tick_params(axis='x', labelsize=tick_fs)

    cb = fig.colorbar(im, ax=ax)
    cb.set_ticks([0, 0.5, 1, 2, 5, 10])
    cb.set_ticklabels(['0', '0.5', '1', '2', '5', '10'])
    cb.set_label('Mean Spike Count', fontsize=cb_fs)
    cb.ax.tick_params(labelsize=tick_fs)

    plt.tight_layout()
    if output_path:
        fig.savefig(str(output_path), dpi=200, bbox_inches='tight')
        pdf_path = str(output_path).replace('.png', '.pdf')
        if pdf_path != str(output_path):
            fig.savefig(pdf_path, dpi=300, bbox_inches='tight')
    if not show:
        plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# Concentration invariance (Figure 3)
# ---------------------------------------------------------------------------

def plot_concentration(conc_data, concentrations=None, hill_ec50=1.0,
                       output_path=None, show=False):
    """Figure 3: 1x3 — (A) gain control, (B) accuracy, (C) similarity."""
    if concentrations is None:
        concentrations = [0.03, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    c_or = '#922b21'
    c_pn = '#1a5276'
    c_kc = '#196f3d'
    c_dec = '#1b2631'

    label_fs, tick_fs, letter_fs, legend_fs = 14, 12, 20, 12

    def _format_conc_ax(ax):
        ax.set_xscale('log')
        ax.set_xticks(concentrations)
        ax.set_xticklabels([f'{c}' for c in concentrations])
        ax.tick_params(labelsize=tick_fs)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # --- A. Gain Control ---
    ax = axes[0]
    ref_or = np.mean(conc_data['1.0']['mean_ors'])
    ref_pn = np.mean(conc_data['1.0']['mean_pns'])
    ref_kc = np.mean(conc_data['1.0']['mean_kcs'])
    or_norm = [np.mean(conc_data[str(c)]['mean_ors']) / max(ref_or, 1e-8) for c in concentrations]
    pn_norm = [np.mean(conc_data[str(c)]['mean_pns']) / max(ref_pn, 1e-8) for c in concentrations]
    kc_norm = [np.mean(conc_data[str(c)]['mean_kcs']) / max(ref_kc, 1e-8) for c in concentrations]
    or_norm_sd = [np.std([v / max(ref_or, 1e-8) for v in conc_data[str(c)]['mean_ors']]) for c in concentrations]
    pn_norm_sd = [np.std([v / max(ref_pn, 1e-8) for v in conc_data[str(c)]['mean_pns']]) for c in concentrations]
    kc_norm_sd = [np.std([v / max(ref_kc, 1e-8) for v in conc_data[str(c)]['mean_kcs']]) for c in concentrations]
    eff_concs = [hill_effective_concentration(c, ec50=hill_ec50) for c in concentrations]

    _eb = dict(linewidth=2, markersize=7, capsize=4, elinewidth=1.2, capthick=1.2)
    ax.errorbar(concentrations, or_norm, yerr=or_norm_sd, fmt='s-', color=c_or, label='OR input', **_eb)
    ax.errorbar(concentrations, pn_norm, yerr=pn_norm_sd, fmt='D-', color=c_pn,
                label='PN (after LN inhibition)', **_eb)
    ax.errorbar(concentrations, kc_norm, yerr=kc_norm_sd, fmt='o-', color=c_kc,
                label='KC (after APL inhibition)', **_eb)
    ax.plot(concentrations, eff_concs, '--', color='gray', alpha=0.5, linewidth=1.5,
            label=f'Hill transfer (EC50={hill_ec50})')
    ax.axvline(x=1.0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Concentration (relative)', fontsize=label_fs)
    ax.set_ylabel('Activity (normalized to c=1.0)', fontsize=label_fs)
    ax.legend(fontsize=legend_fs, loc='upper left')
    _format_conc_ax(ax)
    ax.text(-0.1, 1.02, 'A', transform=ax.transAxes, fontsize=letter_fs,
            fontweight='bold', va='bottom', ha='left')

    # --- B. Classification Accuracy ---
    ax = axes[1]
    dec_mean = [np.mean(conc_data[str(c)]['decoder_accs']) * 100 for c in concentrations]
    dec_std = [np.std(conc_data[str(c)]['decoder_accs']) * 100 for c in concentrations]
    kc_cent_mean = [np.mean(conc_data[str(c)]['kc_centroid_accs']) * 100 for c in concentrations]
    kc_cent_std = [np.std(conc_data[str(c)]['kc_centroid_accs']) * 100 for c in concentrations]

    ax.errorbar(concentrations, dec_mean, yerr=dec_std, fmt='o-', color=c_dec,
                label='Trained decoder', **_eb)
    ax.errorbar(concentrations, kc_cent_mean, yerr=kc_cent_std, fmt='s--', color=c_kc,
                linewidth=1.5, markersize=6, capsize=4, elinewidth=1.2, capthick=1.2,
                label='KC centroid (0 params)')
    ax.axhline(y=100/27, color='red', linestyle=':', alpha=0.4, label=f'Chance ({100/27:.1f}%)')
    ax.axvline(x=1.0, color='gray', linestyle=':', alpha=0.5, label='Training conc.')
    ax.set_xlabel('Concentration (relative)', fontsize=label_fs)
    ax.set_ylabel('Classification Accuracy (%)', fontsize=label_fs)
    ax.legend(fontsize=legend_fs, loc='upper left')
    _format_conc_ax(ax)
    ax.text(-0.1, 1.02, 'B', transform=ax.transAxes, fontsize=letter_fs,
            fontweight='bold', va='bottom', ha='left')

    # --- C. Representation Similarity ---
    ax = axes[2]
    or_s = [np.mean(conc_data[str(c)]['or_sims']) for c in concentrations]
    pn_s = [np.mean(conc_data[str(c)]['pn_sims']) for c in concentrations]
    kc_s = [np.mean(conc_data[str(c)]['kc_sims']) for c in concentrations]
    or_s_sd = [np.std(conc_data[str(c)]['or_sims']) for c in concentrations]
    pn_s_sd = [np.std(conc_data[str(c)]['pn_sims']) for c in concentrations]
    kc_s_sd = [np.std(conc_data[str(c)]['kc_sims']) for c in concentrations]

    ax.errorbar(concentrations, or_s, yerr=or_s_sd, fmt='s-', color=c_or, label='OR', **_eb)
    ax.errorbar(concentrations, pn_s, yerr=pn_s_sd, fmt='D-', color=c_pn, label='PN (after AL)', **_eb)
    ax.errorbar(concentrations, kc_s, yerr=kc_s_sd, fmt='o-', color=c_kc, label='KC (after MB)', **_eb)
    ax.axvline(x=1.0, color='gray', linestyle=':', alpha=0.5, label='Training conc.')
    ax.set_xlabel('Concentration (relative)', fontsize=label_fs)
    ax.set_ylabel('Cosine Similarity to Baseline (c=1.0)', fontsize=label_fs)
    ax.legend(fontsize=legend_fs)
    ax.set_ylim(0, 1.05)
    _format_conc_ax(ax)
    ax.text(-0.1, 1.02, 'C', transform=ax.transAxes, fontsize=letter_fs,
            fontweight='bold', va='bottom', ha='left')

    plt.tight_layout()
    if output_path:
        fig.savefig(str(output_path), dpi=300, bbox_inches='tight')
        pdf_path = str(output_path).replace('.png', '.pdf')
        if pdf_path != str(output_path):
            fig.savefig(pdf_path, dpi=300, bbox_inches='tight')
    if not show:
        plt.close(fig)
    return fig


