"""
run_teacher_consistency.py

C5: Teacher Parameter Consistency Analysis (Reviewer 3, Point 5).
Post-hoc analysis — no training. Loads the 5 cached teacher models and
5 trained student (spiking) models, then computes:

1. Teacher-teacher parameter correlations across 5 seeds
   → How reproducible is the rate-based solution?
2. Teacher-to-student parameter drift during Phase 2 fine-tuning
   → How much do shared parameters change when converting to spiking?
3. Student-student parameter correlations across 5 seeds
   → Does the connectome constrain the spiking solution more or less?

Key shared parameters between teacher and student:
  - or_to_orn.or_gains (21 OR receptor gains) — copied then fine-tuned
  - decoder.weight (28×144 readout matrix) — copied then fine-tuned
  - decoder.bias (28 readout biases) — copied then fine-tuned
  - kc_layer.apl.apl_gain (scalar APL gain) — copied with 4× boost, then fine-tuned

Student-only parameters (analyze consistency across seeds):
  - kc_layer.kc_neurons.v_th (144 per-KC thresholds) — init from scratch
  - antennal_lobe.log_g_gap_* (3 gap conductances) — init from log(1e-10)
  - antennal_lobe.*.log_strength (synaptic pathway strengths)
  - kc_layer.kc_neurons.log_g_soma (somatic conductance)

Results saved to: results/teacher_consistency_c5/
  teacher_consistency_results.json
  c5_convergence_results.json

Notebook section: Section B — C5 (teacher/student consistency figure and table).
"""
import sys
from pathlib import Path
import json
import numpy as np
import torch

sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
sys.stderr.reconfigure(encoding='utf-8')

# ============================================================================
# CONFIG
# ============================================================================
TEACHER_DIR = Path(__file__).parent / 'results' / 'ablations_c7' / 'teachers'
STUDENT_DIR = Path(__file__).parent / 'results' / 'all_connections_nonad_canonical'
OUTPUT_DIR = Path(__file__).parent / 'results' / 'teacher_consistency_c5'

SEEDS = [42, 43, 44, 45, 46]
APL_BOOST = 4.0  # boost factor applied when copying teacher APL gain to student


# ============================================================================
# LOAD MODELS
# ============================================================================
def load_teachers():
    """Load all 5 teacher state dicts."""
    teachers = {}
    for seed in SEEDS:
        path = TEACHER_DIR / f'teacher_seed{seed}.pt'
        if not path.exists():
            print(f"  WARNING: teacher seed {seed} not found at {path}")
            continue
        teachers[seed] = torch.load(path, weights_only=False, map_location='cpu')
        print(f"  Loaded teacher seed {seed}")
    return teachers


def load_students():
    """Load all 5 student state dicts."""
    students = {}
    for seed in SEEDS:
        path = STUDENT_DIR / f'model_seed{seed}.pt'
        if not path.exists():
            print(f"  WARNING: student seed {seed} not found at {path}")
            continue
        students[seed] = torch.load(path, weights_only=False, map_location='cpu')
        print(f"  Loaded student seed {seed}")
    return students


# ============================================================================
# PARAMETER EXTRACTION
# ============================================================================
def extract_teacher_params(state):
    """Extract key parameter vectors from teacher state dict.

    Teacher uses rate-based model (ConnectomeConstrainedModel), which has
    different param names than the spiking student.
    """
    return {
        'or_gains': state['or_to_orn.or_gains'].numpy(),                  # (21,)
        'decoder_weight': state['decoder.weight'].numpy(),                 # (28, 144)
        'decoder_bias': state['decoder.bias'].numpy(),                     # (28,)
        'apl_gain': state['kc_layer.apl.apl_gain'].numpy().item(),        # scalar
        'kc_threshold': state['kc_layer.kc_threshold'].numpy(),           # (144,)
        'orn_pn_strength': state['antennal_lobe.orn_pn.strengths'].numpy().item(),
        'ln_pn_strength': state['antennal_lobe.ln_pn.strengths'].numpy().item(),
    }


def extract_student_params(state):
    """Extract key parameter vectors from student state dict.

    Student uses spiking model (SpikingConnectomeConstrainedModel) with
    different naming: log_strength instead of strengths, v_th instead of
    kc_threshold, etc.
    """
    return {
        'or_gains': state['or_to_orn.or_gains'].numpy(),                  # (21,)
        'decoder_weight': state['decoder.weight'].numpy(),                 # (28, 144)
        'decoder_bias': state['decoder.bias'].numpy(),                     # (28,)
        'apl_gain': state['kc_layer.apl.apl_gain'].numpy().item(),        # scalar
        'kc_v_th': state['kc_layer.kc_neurons.v_th'].numpy(),             # (144,)
        'ln_v_th': state['antennal_lobe.ln_neurons.v_th'].numpy(),        # (108,)
        'orn_v_th': state['antennal_lobe.orn_neurons.v_th'].numpy(),      # (42,)
        'pn_v_th': state['antennal_lobe.pn_neurons.v_th'].numpy(),        # (72,)
        'log_g_gap_ln': state['antennal_lobe.log_g_gap_ln'].numpy().item(),
        'log_g_gap_pn': state['antennal_lobe.log_g_gap_pn'].numpy().item(),
        'log_g_gap_eln_pn': state['antennal_lobe.log_g_gap_eln_pn'].numpy().item(),
        'log_g_soma': state['kc_layer.kc_neurons.log_g_soma'].numpy().item(),
        'orn_pn_log_strength': state['antennal_lobe.orn_pn.log_strength'].numpy().item(),
        'ln_pn_log_strength': state['antennal_lobe.ln_pn.log_strength'].numpy().item(),
        'ln_pn_excit_log_strength': state['antennal_lobe.ln_pn_excit.log_strength'].numpy().item(),
        'kc_kc_aa_log_strength': state['kc_layer.kc_kc_aa.log_strength'].numpy().item(),
        'pn_kc_log_strength': state['kc_layer.pn_kc.log_strength'].numpy().item(),
        'log_tau_apl': state['kc_layer.apl.log_tau_apl'].numpy().item(),
    }


# ============================================================================
# CORRELATION ANALYSIS
# ============================================================================
def pairwise_correlations(param_dict_by_seed, param_name):
    """Compute all pairwise Pearson correlations for a vector parameter.

    C5: Measures how similar the same parameter is across different seeds.
    High correlation = connectome strongly constrains this parameter.
    """
    seeds = sorted(param_dict_by_seed.keys())
    vectors = {s: param_dict_by_seed[s][param_name].flatten() for s in seeds}

    n = len(seeds)
    corr_matrix = np.ones((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            r = np.corrcoef(vectors[seeds[i]], vectors[seeds[j]])[0, 1]
            corr_matrix[i, j] = r
            corr_matrix[j, i] = r

    # Mean of upper triangle
    triu_idx = np.triu_indices(n, k=1)
    mean_corr = float(np.mean(corr_matrix[triu_idx]))
    std_corr = float(np.std(corr_matrix[triu_idx]))

    return {
        'mean_correlation': mean_corr,
        'std_correlation': std_corr,
        'min_correlation': float(np.min(corr_matrix[triu_idx])),
        'max_correlation': float(np.max(corr_matrix[triu_idx])),
        'n_pairs': int(len(triu_idx[0])),
    }


def scalar_consistency(param_dict_by_seed, param_name):
    """Compute mean and CV for a scalar parameter across seeds.

    C5: For scalar parameters (strengths, gains), correlation isn't meaningful.
    Instead, report coefficient of variation (CV = std/mean).
    """
    seeds = sorted(param_dict_by_seed.keys())
    values = [param_dict_by_seed[s][param_name] for s in seeds]
    mean_val = float(np.mean(values))
    std_val = float(np.std(values))
    cv = std_val / abs(mean_val) if abs(mean_val) > 1e-10 else float('nan')

    return {
        'mean': mean_val,
        'std': std_val,
        'cv': cv,
        'values': {s: float(v) for s, v in zip(seeds, values)},
    }


# ============================================================================
# DRIFT ANALYSIS
# ============================================================================
def compute_drift(teacher_params, student_params, seeds):
    """Compute teacher-to-student parameter drift for shared parameters.

    C5: For each seed, measures how much shared parameters change during
    Phase 2 (spiking fine-tuning). Large drift = spiking dynamics reshape
    the solution significantly. Small drift = rate-based solution transfers.

    Note: APL gain is boosted 4x when copied to student, so we compare
    teacher_gain * 4 vs student_gain.
    """
    results = {}

    for param_name in ['or_gains', 'decoder_weight', 'decoder_bias']:
        drifts = []
        for s in seeds:
            t_vec = teacher_params[s][param_name].flatten()
            s_vec = student_params[s][param_name].flatten()

            # Correlation: how much does the pattern change?
            corr = float(np.corrcoef(t_vec, s_vec)[0, 1])

            # Relative magnitude change
            t_norm = float(np.linalg.norm(t_vec))
            s_norm = float(np.linalg.norm(s_vec))
            norm_ratio = s_norm / t_norm if t_norm > 1e-10 else float('nan')

            # Mean absolute relative change per element
            denom = np.maximum(np.abs(t_vec), 1e-8)
            rel_change = float(np.mean(np.abs(s_vec - t_vec) / denom))

            drifts.append({
                'seed': s,
                'correlation': corr,
                'norm_ratio': norm_ratio,
                'mean_rel_change': rel_change,
            })

        avg_corr = float(np.mean([d['correlation'] for d in drifts]))
        avg_rel = float(np.mean([d['mean_rel_change'] for d in drifts]))

        results[param_name] = {
            'per_seed': drifts,
            'avg_correlation': avg_corr,
            'avg_rel_change': avg_rel,
            'interpretation': 'preserved' if avg_corr > 0.9 else
                            'partially preserved' if avg_corr > 0.5 else
                            'substantially reshaped',
        }

    # APL gain (scalar, boosted 4x at init)
    apl_drifts = []
    for s in seeds:
        t_gain = teacher_params[s]['apl_gain'] * APL_BOOST
        s_gain = student_params[s]['apl_gain']
        rel_change = abs(s_gain - t_gain) / abs(t_gain) if abs(t_gain) > 1e-8 else float('nan')
        apl_drifts.append({
            'seed': s,
            'teacher_gain_x4': float(t_gain),
            'student_gain': float(s_gain),
            'relative_change': float(rel_change),
        })
    avg_apl_rel = float(np.mean([d['relative_change'] for d in apl_drifts]))
    results['apl_gain'] = {
        'per_seed': apl_drifts,
        'avg_rel_change': avg_apl_rel,
        'note': f'Teacher gain multiplied by APL_BOOST={APL_BOOST} at student init',
    }

    return results


# ============================================================================
# MAIN
# ============================================================================
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("C5: TEACHER PARAMETER CONSISTENCY ANALYSIS")
    print("="*70)

    # Load models
    print("\nLoading teacher models...")
    teachers = load_teachers()
    print("\nLoading student models...")
    students = load_students()

    available_seeds = sorted(set(teachers.keys()) & set(students.keys()))
    print(f"\nAvailable seeds: {available_seeds}")

    # Extract parameters
    teacher_params = {s: extract_teacher_params(teachers[s]) for s in available_seeds}
    student_params = {s: extract_student_params(students[s]) for s in available_seeds}

    # ====================================================================
    # 1. TEACHER-TEACHER CONSISTENCY
    # ====================================================================
    print(f"\n{'='*70}")
    print("1. TEACHER-TEACHER PARAMETER CONSISTENCY")
    print(f"{'='*70}")
    print("   (High correlation = connectome constrains rate-based solution)")

    teacher_vector_params = ['or_gains', 'decoder_weight', 'decoder_bias', 'kc_threshold']
    teacher_vector_results = {}
    for param in teacher_vector_params:
        result = pairwise_correlations(teacher_params, param)
        teacher_vector_results[param] = result
        print(f"\n  {param}:")
        print(f"    Mean pairwise r = {result['mean_correlation']:.4f} +/- {result['std_correlation']:.4f}")
        print(f"    Range: [{result['min_correlation']:.4f}, {result['max_correlation']:.4f}]")

    teacher_scalar_params = ['apl_gain', 'orn_pn_strength', 'ln_pn_strength']
    teacher_scalar_results = {}
    for param in teacher_scalar_params:
        result = scalar_consistency(teacher_params, param)
        teacher_scalar_results[param] = result
        print(f"\n  {param}:")
        print(f"    Mean = {result['mean']:.6f}, CV = {result['cv']:.4f}")
        print(f"    Values: {result['values']}")

    # ====================================================================
    # 2. STUDENT-STUDENT CONSISTENCY
    # ====================================================================
    print(f"\n{'='*70}")
    print("2. STUDENT-STUDENT PARAMETER CONSISTENCY")
    print(f"{'='*70}")
    print("   (Compare to teacher consistency — more or less constrained?)")

    student_vector_params = ['or_gains', 'decoder_weight', 'decoder_bias', 'kc_v_th',
                             'ln_v_th', 'orn_v_th', 'pn_v_th']
    student_vector_results = {}
    for param in student_vector_params:
        result = pairwise_correlations(student_params, param)
        student_vector_results[param] = result
        print(f"\n  {param}:")
        print(f"    Mean pairwise r = {result['mean_correlation']:.4f} +/- {result['std_correlation']:.4f}")

    student_scalar_params = ['apl_gain', 'log_g_gap_ln', 'log_g_gap_pn', 'log_g_gap_eln_pn',
                             'log_g_soma', 'orn_pn_log_strength', 'ln_pn_log_strength',
                             'kc_kc_aa_log_strength', 'pn_kc_log_strength', 'log_tau_apl']
    student_scalar_results = {}
    for param in student_scalar_params:
        result = scalar_consistency(student_params, param)
        student_scalar_results[param] = result
        print(f"\n  {param}:")
        print(f"    Mean = {result['mean']:.6f}, CV = {result['cv']:.4f}")

    # ====================================================================
    # 3. TEACHER-TO-STUDENT DRIFT
    # ====================================================================
    print(f"\n{'='*70}")
    print("3. TEACHER-TO-STUDENT PARAMETER DRIFT (Phase 2)")
    print(f"{'='*70}")
    print("   (Low drift = rate-based solution transfers to spiking)")

    drift_results = compute_drift(teacher_params, student_params, available_seeds)

    for param, result in drift_results.items():
        if 'avg_correlation' in result:
            print(f"\n  {param}:")
            print(f"    Avg T-S correlation: {result['avg_correlation']:.4f}")
            print(f"    Avg relative change: {result['avg_rel_change']:.4f}")
            print(f"    Interpretation: {result['interpretation']}")
        elif 'avg_rel_change' in result:
            print(f"\n  {param}:")
            print(f"    Avg relative change: {result['avg_rel_change']:.4f}")
            for d in result['per_seed']:
                print(f"      seed {d['seed']}: teacher*4={d['teacher_gain_x4']:.4f} -> student={d['student_gain']:.4f}")

    # ====================================================================
    # 4. COMPARISON: Teacher vs Student consistency
    # ====================================================================
    print(f"\n{'='*70}")
    print("4. COMPARISON: DOES SPIKING FINE-TUNING INCREASE OR DECREASE CONSISTENCY?")
    print(f"{'='*70}")

    shared_params = ['or_gains', 'decoder_weight', 'decoder_bias']
    print(f"\n  {'Parameter':<25} {'Teacher r':>12} {'Student r':>12} {'Delta':>8} {'Change':>15}")
    print(f"  {'-'*72}")
    comparison_results = {}
    for param in shared_params:
        t_r = teacher_vector_results[param]['mean_correlation']
        s_r = student_vector_results[param]['mean_correlation']
        delta = s_r - t_r
        change = 'MORE consistent' if delta > 0.01 else 'LESS consistent' if delta < -0.01 else 'STABLE'
        comparison_results[param] = {'teacher_r': t_r, 'student_r': s_r, 'delta': delta, 'change': change}
        print(f"  {param:<25} {t_r:>12.4f} {s_r:>12.4f} {delta:>+8.4f} {change:>15}")

    # ====================================================================
    # SAVE RESULTS
    # ====================================================================
    full_results = {
        'seeds': available_seeds,
        'teacher_teacher': {
            'vector_params': teacher_vector_results,
            'scalar_params': teacher_scalar_results,
        },
        'student_student': {
            'vector_params': student_vector_results,
            'scalar_params': student_scalar_results,
        },
        'teacher_to_student_drift': drift_results,
        'comparison': comparison_results,
    }

    results_path = OUTPUT_DIR / 'teacher_consistency_results.json'
    def _json_default(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, (np.bool_,)): return bool(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return str(obj)

    with open(results_path, 'w') as f:
        json.dump(full_results, f, indent=2, default=_json_default)
    print(f"\nResults saved: {results_path}")


if __name__ == '__main__':
    main()
