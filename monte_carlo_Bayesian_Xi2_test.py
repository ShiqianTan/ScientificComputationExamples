import numpy as np
from scipy.stats import beta

# 数据
N1 = 52
y1 = 9
N2 = 48
y2 = 4

# 先验参数
alpha1 = beta1 = alpha2 = beta2 = 1

# 计算后验分布参数
posterior_alpha1 = alpha1 + y1
posterior_beta1 = beta1 + N1 - y1
posterior_alpha2 = alpha2 + y2
posterior_beta2 = beta2 + N2 - y2

# 蒙特卡罗模拟
num_samples = 100000
theta1_samples = beta.rvs(posterior_alpha1, posterior_beta1, size=num_samples)
theta2_samples = beta.rvs(posterior_alpha2, posterior_beta2, size=num_samples)

# 计算 delta 样本
delta_samples = theta1_samples - theta2_samples

# 计算 p(theta1 > theta2 | D)
p_theta1_greater_theta2 = np.mean(delta_samples > 0)

print(f"p(theta1 > theta2 | D) = {p_theta1_greater_theta2:.3f}")
# output = 0.901
