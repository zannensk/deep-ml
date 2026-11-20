import math

def calculate_power(effect_size: float, sample_size_per_group: int, alpha: float = 0.05, two_tailed: bool = True) -> float:
    """
    Calculate statistical power for a two-sample z-test.
    
    Parameters:
    effect_size: Cohen's d (standardized effect size)
    sample_size_per_group: Number of observations per group
    alpha: Significance level (default 0.05)
    two_tailed: Whether the test is two-tailed (default True)
    
    Returns:
    Statistical power as a float rounded to 4 decimal places
    """
    # Your code here
    def norm_cdf(x):
        return 0.5 * (
            1 + math.erf(x / math.sqrt(2))
        )
    
    def norm_ppf(p):
        if p < 0.5:
            return -norm_ppf(1 - p)
        
        t = math.sqrt(-2 * math.log(1 - p))
        c0, c1, c2 = 2.515517, 0.802853, 0.010328
        d1, d2, d3 = 1.432788, 0.189269, 0.001308

        numerator = c0 + c1 * t + c2 * t ** 2
        denominator = (
            1 + d1 * t + d2 * t ** 2 + d3 * t ** 3
        )
        return t - numerator / denominator
    
    n = sample_size_per_group
    lam = effect_size * math.sqrt(n / 2)

    if two_tailed:
        z_crit = norm_ppf(1 - alpha / 2)
        right_power = 1 - norm_cdf(z_crit - lam)
        left_power = norm_cdf(-z_crit - lam)
        power = right_power + left_power

    else:
        z_crit = norm_ppf(1 - alpha)
        power = 1 - norm_cdf(z_crit - lam)
    
    return round(power, 4)


