import numpy as np
import math

from statistics import NormalDist

def analyze_ab_test(control_outcomes: list, treatment_outcomes: list, confidence_level: float = 0.95, min_detectable_effect: float = 0.02) -> dict:
    """
    Analyze A/B test results for model comparison with statistical rigor.
    
    Args:
        control_outcomes: List of binary outcomes (0 or 1) for control group
        treatment_outcomes: List of binary outcomes (0 or 1) for treatment group
        confidence_level: Confidence level for statistical tests (default 0.95)
        min_detectable_effect: Minimum absolute effect size considered practically significant
    
    Returns:
        dict with statistical analysis results and recommendation
    """
    if not control_outcomes or not treatment_outcomes:
        return {}

    control = np.asarray(control_outcomes, dtype=float)
    treatment = np.asarray(treatment_outcomes, dtype=float)

    n_c = len(control)
    n_t = len(treatment)

    p_c = np.mean(control)
    p_t = np.mean(treatment)

    absolute_lift = p_t - p_c
    if p_c == 0:
        relative_lift = 0.0
    else:
        relative_lift = absolute_lift / p_c * 100
    
    success_c = np.sum(control)
    success_t = np.sum(treatment)

    pooled_p = (
        success_c + success_t
    ) / (n_c + n_t)

    pooled_se = math.sqrt(
        pooled_p
        * (1 - pooled_p)
        * (1 / n_c + 1 / n_t)
    )

    if pooled_se == 0:
        z_stat = 0.0
        p_value = 1.0
    else:
        z_stat = absolute_lift / pooled_se

        cdf = NormalDist().cdf(abs(z_stat))
        p_value = 2 * (1 - cdf)

    alpha = 1 - confidence_level
    z_critical = NormalDist().inv_cdf(
        1 - alpha / 2
    )
    unpooled_se = math.sqrt(
        p_c * (1 - p_c) / n_c
        +
        p_t * (1 - p_t) / n_t
    )
    ci_low = (
        absolute_lift
        - z_critical * unpooled_se
    )
    ci_high = (
        absolute_lift
        + z_critical * unpooled_se
    )

    statistically_significant = (
        p_value < alpha
    )

    practically_significant = (
        abs(absolute_lift)
        >= min_detectable_effect
    )

    power = 0.80
    z_alpha = NormalDist().inv_cdf(
        1 - alpha / 2
    )
    z_beta = NormalDist().inv_cdf(power)
    p1 = p_c
    p2 = p1 + min_detectable_effect
    if p2 > 1:
        p2 = p1 - min_detectable_effect
    p2 = max(0.0, min(1.0, p2))
    effect = abs(p2 - p1)
    if effect == 0:
        required_sample_size = 0
    else:
        p_bar = (p1 + p2) / 2
        numerator = (
            z_alpha
            * math.sqrt(2 * p_bar * (1 - p_bar))
            +
            z_beta
            * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
        ) ** 2

        required_sample_size = math.ceil(numerator / effect ** 2)
    
    if (
        statistically_significant
        and practically_significant
        and absolute_lift > 0
    ):
        recommendation = "launch_treatment"
    
    elif (
        statistically_significant
        and (
            absolute_lift < 0
            or not practically_significant
        )
    ):
        recommendation = "keep_control"

    else:
        recommendation = "continue_testing"
    
    return {
        "control_rate": round(float(p_c), 4),
        "treatment_rate": round(float(p_t), 4),
        "absolute_lift": round(float(absolute_lift), 4),
        "relative_lift_pct": round(float(relative_lift), 4),
        "z_statistic": round(float(z_stat), 4),
        "p_value": round(float(p_value), 4),
        "confidence_interval": (
            round(float(ci_low), 4),
            round(float(ci_high), 4)
        ),
        "statistically_significant": bool(statistically_significant),
        "practically_significant": bool(practically_significant),
        "required_sample_size": required_sample_size,
        "recommendation": recommendation,
    }