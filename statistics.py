from scipy.stats import chisquare
import numpy as np
from scipy.stats import fisher_exact
from scipy.stats import chisquare
import numpy as np
import math
from statsmodels.stats.proportion import proportions_ztest

# Confidence interval for 75.5%
p = 0.755
q = 1.96*np.sqrt(0.755*(1-0.755)/203)
print(p+q)
print(p-q)
print(q)

# Confidence interval for VADER 74.3%
p = 0.743
q = 1.96*np.sqrt(0.743*(1-0.743)/203)
print(p+q)
print(p-q)
print(q)

# Estimated sample size for an 5% error margin
n = 1.96**2*(0.755*(1-0.755))/0.05**2
print("Sample size:", n)



# H_0: The predicted distribution from the model is not signif different from the distribution
# obtained by guessing the majority class
# Contingency table for Fisher's Exact Test
table = [[23, 18], [41, 0]]  # Observed and guessed all positives

# Perform Fisher's Exact Test
odds_ratio, p_value = fisher_exact(table, alternative='two-sided')

print(f"Fisher's Exact Test P-Value: {p_value:.10f}")



# H_0: the predicted class distribution does not significantly differ from the true class distribution 
# Observed counts from the model's predictions
observed_counts = np.array([23, 18])  # Predicted Positives, Predicted Negatives

# Expected counts based on the true class distribution
total_test_samples = 41
true_positives = 18
true_negatives = 13
expected_counts = np.array([
    total_test_samples * (true_positives / (true_positives + true_negatives)),  # Expected Positives
    total_test_samples * (true_negatives / (true_positives + true_negatives))  # Expected Negatives
])

# Perform chi-square test
chi2_stat, p_value = chisquare(f_obs=observed_counts, f_exp=expected_counts)

# Print results
print(f"Observed Counts: {observed_counts}")
print(f"Expected Counts: {expected_counts}")
print(f"Chi-Square Statistic: {chi2_stat:.4f}")
print(f"P-Value: {p_value:.4f}")



# Two-proportion z-test
# Parameters
p1 = 0.755  # Accuracy with VADER
p2 = 0.743  # Accuracy without VADER
n1 = n2 = 203  # Sample size

# Calculate number of correct predictions
success1 = p1 * n1
success2 = p2 * n2

# do the two-proportion z-test
count = [success1, success2]
nobs = [n1, n2]

z_stat, p_value = proportions_ztest(count, nobs, alternative='larger')

# Print results
print(f"Z-Statistic: {z_stat:.4f}")
print(f"P-Value: {p_value:.4f}")

# Conclusion
if p_value < 0.05:
    print("Reject the null hypothesis: Adding VADER has a significant impact.")
else:
    print("Fail to reject the null hypothesis: Adding VADER does not have a significant impact.")
