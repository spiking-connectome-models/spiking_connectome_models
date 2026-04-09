"""Analysis subpackage for connectome model paper figures.

Re-exports all functions from utils, compute, and plotting modules.

Usage:
    from spiking_connectome_models.analysis import (
        load_models, evaluate_model, compute_per_pair_decorrelation,
        plot_training_curves, plot_summary, ...
    )
"""

from .utils import (
    cosine_sim_pair,
    cosine_sim_matrix,
    noisy_forward_pass,
    per_pair_similarity_ratios,
    centroid_classify,
    hill_effective_concentration,
    extract_all_parameters,
    compute_pairwise_correlations,
    analyze_biological_parameters,
)

from .compute import (
    load_models,
    evaluate_model,
    compute_per_pair_decorrelation,
    compute_mean_sim_decorrelation,
    run_mancini_test,
    evaluate_per_odor_all_models,
    compute_cross_model_consistency,
    compute_kc_consistency_per_odor,
    centroid_accuracy,
    extract_gap_junction_info,
    extract_nonad_strengths,
    run_concentration_invariance,
    compute_few_param_cv,
)

from .plotting import (
    plot_training_curves,
    plot_per_odor_breakdown,
    plot_kc_sparsity_distribution,
    plot_biological_parameters,
    plot_correlation_bars,
    plot_few_param_cv,
    plot_mancini_validation,
    plot_gap_junction_conductances,
    plot_ln_pn_split,
    plot_core_figure,
    plot_kc_heatmap,
    plot_concentration,
)
