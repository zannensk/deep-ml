import numpy as np
import math

# scipy is not available in this environment. The two helpers below compute
# the regularized incomplete beta function via the Numerical Recipes
# continued-fraction algorithm — you'll need them to compute the p-value
# from the t-distribution.
def _betacf(a, b, x):
    """Continued fraction for the incomplete beta function."""
    max_iter, eps = 200, 3e-12
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 1e-30: d = 1e-30
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30: d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30: c = 1e-30
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30: d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30: c = 1e-30
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps: break
    return h

def _betainc(a, b, x):
    """Regularized incomplete beta function I_x(a, b)."""
    if x <= 0.0: return 0.0
    if x >= 1.0: return 1.0
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(math.log(x) * a + math.log(1.0 - x) * b - lbeta)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def two_sample_t_test(sample1: list[float], sample2: list[float],
                      alpha: float = 0.05) -> dict:
    """
    Perform a two-sample independent t-test (Welch's t-test).

    Args:
        sample1: First sample data
        sample2: Second sample data
        alpha: Significance level (default 0.05)

    Returns:
        Dictionary containing:
        - t_statistic: The calculated t-statistic
        - p_value: Two-tailed p-value
        - degrees_of_freedom: Degrees of freedom (Welch-Satterthwaite)
        - reject_null: Boolean, whether to reject null hypothesis
        - cohens_d: Effect size (Cohen's d)
    """
    # Your code here
    x1 = np.array(sample1, dtype=float)
    x2 = np.array(sample2, dtype=float)
    
    n1, n2 = len(x1), len(x2)

    mean1, mean2 = np.mean(x1), np.mean(x2)

    var1, var2 = np.var(x1, ddof=1), np.var(x2, ddof=1)

    se = np.sqrt(var1 / n1 + var2 / n2)
    t = (mean1 - mean2) / se

    a = var1 / n1
    b = var2 / n2

    df = (a + b)**2 / (
        a**2 / (n1 - 1) + b**2 / (n2 - 1)
    )

    p = _betainc(
        df / 2, 0.5, df / (df + t**2)
    )

    pooled_var = (
        (n1 - 1) * var1 + (n2 - 1) * var2
    ) / (n1 + n2 - 2)

    cohen_d = (mean1 - mean2) / np.sqrt(pooled_var)

    return {
        "t_statistic": round(float(t), 4),
        "p_value": round(float(p), 6),
        "degrees_of_freedom": round(float(df), 4),
        "reject_null": bool(p < alpha),
        "cohens_d": round(float(cohen_d), 4)
    }


