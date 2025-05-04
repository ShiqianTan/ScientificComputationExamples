import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, uniform

# Target function we want to integrate
def f(x):
    return np.exp(-x**2) * (x > 3)  # Indicator function for x > 3

# True value of the integral (for comparison)
true_value = norm.sf(3)  # Survival function (1 - CDF) at 3

# Standard Monte Carlo estimation
def standard_monte_carlo(n_samples):
    samples = np.random.randn(n_samples)
    return np.mean(f(samples)), samples

# Importance sampling Monte Carlo estimation
def importance_sampling(n_samples):
    # We'll use a shifted normal distribution as our importance distribution
    # Shift the mean to 3 since we're interested in x > 3 region
    importance_mean = 3
    importance_std = 1
    
    # Sample from importance distribution
    samples = np.random.normal(importance_mean, importance_std, n_samples)
    
    # Calculate weights (ratio of target density to importance density)
    # Target density is standard normal (mean=0, std=1)
    target_density = norm.pdf(samples)
    importance_density = norm.pdf(samples, loc=importance_mean, scale=importance_std)
    weights = target_density / importance_density
    
    # Compute weighted average
    weighted_values = f(samples) * weights
    estimate = np.mean(weighted_values)
    
    return estimate, samples

# Number of samples
n_samples = 10000

# Run both methods
mc_estimate, mc_samples = standard_monte_carlo(n_samples)
is_estimate, is_samples = importance_sampling(n_samples)

# Calculate errors
mc_error = abs(mc_estimate - true_value)
is_error = abs(is_estimate - true_value)

print(f"True value: {true_value:.6f}")
print(f"Standard Monte Carlo estimate: {mc_estimate:.6f} (error: {mc_error:.6f})")
print(f"Importance Sampling estimate: {is_estimate:.6f} (error: {is_error:.6f})")

# Visualization
plt.figure(figsize=(12, 6))

# Plot the target function and distributions
x = np.linspace(-1, 6, 1000)
plt.plot(x, f(x), 'r-', label='Target function f(x)')
plt.plot(x, norm.pdf(x), 'b-', label='Standard normal PDF (target density)')
plt.plot(x, norm.pdf(x, loc=3, scale=1), 'g-', label='Importance distribution')

# Plot samples
plt.hist(mc_samples, bins=50, density=True, alpha=0.3, color='blue', label='Standard MC samples')
plt.hist(is_samples, bins=50, density=True, alpha=0.3, color='green', label='Importance samples')

plt.title('Importance Sampling vs Standard Monte Carlo')
plt.xlabel('x')
plt.ylabel('Density / Function value')
plt.legend()
plt.grid(True)
plt.show()

# Convergence plot
n_runs = 100
sample_sizes = np.logspace(1, 5, 20).astype(int)

mc_errors = []
is_errors = []

for n in sample_sizes:
    mc_run_errors = []
    is_run_errors = []
    for _ in range(n_runs):
        mc_est, _ = standard_monte_carlo(n)
        is_est, _ = importance_sampling(n)
        mc_run_errors.append(abs(mc_est - true_value))
        is_run_errors.append(abs(is_est - true_value))
    mc_errors.append(np.mean(mc_run_errors))
    is_errors.append(np.mean(is_run_errors))

plt.figure(figsize=(10, 6))
plt.loglog(sample_sizes, mc_errors, 'b-o', label='Standard Monte Carlo')
plt.loglog(sample_sizes, is_errors, 'g-s', label='Importance Sampling')
plt.title('Convergence Comparison')
plt.xlabel('Number of samples')
plt.ylabel('Absolute error')
plt.legend()
plt.grid(True)
plt.show()
