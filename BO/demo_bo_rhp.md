以下是在 Linux 系统下安装和配置运行该优化算法所需环境的完整指南。

## 🐧 环境安装和配置

### 1. 创建并激活虚拟环境
```bash
# 创建虚拟环境
python -m venv spab_optimization
source spab_optimization/bin/activate

# 升级pip
pip install --upgrade pip
```

### 2. 安装依赖包
创建 `requirements.txt` 文件：
```txt
torch>=2.0.0
botorch>=0.9.0
gpytorch>=1.10.0
numpy>=1.21.0
scipy>=1.7.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
pandas>=1.3.0
seaborn>=0.11.0
jupyter>=1.0.0
tqdm>=4.62.0
```

安装依赖：
```bash
pip install -r requirements.txt
```

### 3. 验证安装
创建验证脚本 `test_installation.py`：
```python
#!/usr/bin/env python3

import torch
import botorch
import gpytorch
import numpy as np
import sklearn

print("=== 环境验证 ===")
print(f"PyTorch版本: {torch.__version__}")
print(f"BoTorch版本: {botorch.__version__}")
print(f"GPyTorch版本: {gpytorch.__version__}")
print(f"CUDA是否可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"GPU设备: {torch.cuda.get_device_name(0)}")

print("所有依赖包安装成功！")
```

运行验证：
```bash
python test_installation.py
```

### 4. 完整的运行脚本
创建 `run_optimization.py`：
```python
#!/usr/bin/env python3
"""
RHP优化算法运行脚本
使用方法: python run_optimization.py --target [IFN|TNF] --algorithm [GA|BO|BOTH]
"""

import argparse
import numpy as np
import torch
import json
from datetime import datetime
import sys
import os

# 添加自定义模块路径
sys.path.append('./src')

from optimization.ga import GeneticAlgorithm
from optimization.bo import BayesianOptimization

def setup_directories():
    """创建结果保存目录"""
    os.makedirs('results', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    os.makedirs('data/elisa', exist_ok=True)

def load_elisa_data(target_protein):
    """加载ELISA实验数据（示例）"""
    # 这里应该替换为实际的ELISA数据加载逻辑
    if target_protein.upper() == 'IFN':
        target_mean = 0.471  # 根据论文中的随机采样点平均值
        control_mean = 0.630
    elif target_protein.upper() == 'TNF':
        target_mean = 1.0    # 需要根据实际数据调整
        control_mean = 1.0
    else:
        raise ValueError("目标蛋白必须是 IFN 或 TNF")
    
    return target_mean, control_mean

def create_score_function(target_protein, target_mean, control_mean):
    """创建评分函数"""
    def score_function(composition):
        """
        基于ELISA数据的评分函数
        composition: 8维向量，表示RHP中各组分摩尔分数
        """
        # 这里应该替换为实际的ELISA信号预测模型
        # 示例使用随机森林或神经网络预测ELISA信号
        try:
            # 模拟ELISA信号预测
            target_signal = np.dot(composition, np.random.randn(8)) + 1.0
            control_signal = np.dot(composition, np.random.randn(8)) + 0.5
            
            # 计算标准化评分
            score = (target_signal / target_mean) - (control_signal / control_mean)
            return float(score)
        except Exception as e:
            print(f"评分计算错误: {e}")
            return -10.0  # 返回极低分
    
    return score_function

def save_results(algorithm, best_composition, best_score, target_protein, iteration):
    """保存优化结果"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"results/{target_protein}_{algorithm}_{timestamp}.json"
    
    result = {
        'algorithm': algorithm,
        'target_protein': target_protein,
        'best_composition': best_composition.tolist() if hasattr(best_composition, 'tolist') else best_composition,
        'best_score': float(best_score),
        'timestamp': timestamp,
        'iteration': iteration
    }
    
    with open(filename, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"结果已保存至: {filename}")
    return filename

def main():
    parser = argparse.ArgumentParser(description='RHP优化算法')
    parser.add_argument('--target', type=str, required=True, choices=['IFN', 'TNF'], 
                       help='目标蛋白: IFN 或 TNF')
    parser.add_argument('--algorithm', type=str, default='BOTH', 
                       choices=['GA', 'BO', 'BOTH'], help='优化算法')
    parser.add_argument('--iterations', type=int, default=10, help='迭代次数')
    parser.add_argument('--population', type=int, default=50, help='GA种群大小')
    parser.add_argument('--init_samples', type=int, default=10, help='BO初始样本数')
    
    args = parser.parse_args()
    
    # 设置环境
    setup_directories()
    
    # 加载ELISA数据
    target_mean, control_mean = load_elisa_data(args.target)
    score_function = create_score_function(args.target, target_mean, control_mean)
    
    print(f"开始优化目标蛋白: {args.target}")
    print(f"使用算法: {args.algorithm}")
    print(f"迭代次数: {args.iterations}")
    
    results = {}
    
    # 运行遗传算法
    if args.algorithm in ['GA', 'BOTH']:
        print("\n" + "="*50)
        print("运行遗传算法...")
        print("="*50)
        
        ga = GeneticAlgorithm(
            population_size=args.population,
            num_generations=args.iterations,
            mutation_rate=0.1
        )
        
        best_ga, score_ga = ga.run(score_function, num_components=8)
        results['GA'] = {'composition': best_ga, 'score': score_ga}
        
        save_results('GA', best_ga, score_ga, args.target, args.iterations)
    
    # 运行贝叶斯优化
    if args.algorithm in ['BO', 'BOTH']:
        print("\n" + "="*50)
        print("运行贝叶斯优化...")
        print("="*50)
        
        bo = BayesianOptimization(
            n_init=args.init_samples,
            n_iter=args.iterations,
            noise_var=0.12
        )
        
        best_bo, score_bo = bo.run(score_function, num_components=8)
        results['BO'] = {'composition': best_bo, 'score': score_bo}
        
        save_results('BO', best_bo, score_bo, args.target, args.iterations)
    
    # 输出最终结果比较
    print("\n" + "="*50)
    print("优化结果比较")
    print("="*50)
    
    for algo, result in results.items():
        print(f"{algo}:")
        print(f"  最佳评分: {result['score']:.4f}")
        print(f"  最佳组成: {result['composition']}")
        print()

if __name__ == "__main__":
    main()
```

### 5. 项目目录结构
```bash
mkdir -p src/optimization data/elisa results logs

# 创建目录结构
tree .
# .
# ├── run_optimization.py
# ├── requirements.txt
# ├── test_installation.py
# ├── src/
# │   └── optimization/
# │       ├── __init__.py
# │       ├── ga.py
# │       └── bo.py
# ├── data/
# │   └── elisa/
# ├── results/
# └── logs/
```

### 6. 模块化代码
将算法代码分别保存到模块中：

**`src/optimization/ga.py`**:
```python
import numpy as np

class GeneticAlgorithm:
    # 这里放入前面定义的 GeneticAlgorithm 类代码
    pass
```

**`src/optimization/bo.py`**:
```python
import torch
import botorch
from botorch.models import SingleTaskGP
# 这里放入前面定义的 BayesianOptimization 类代码

class BayesianOptimization:
    pass
```

**`src/optimization/__init__.py`**:
```python
from .ga import GeneticAlgorithm
from .bo import BayesianOptimization

__all__ = ['GeneticAlgorithm', 'BayesianOptimization']
```

### 7. 使用示例
```bash
# 激活环境
source spab_optimization/bin/activate

# 运行优化（IFN目标蛋白，使用两种算法）
python run_optimization.py --target IFN --algorithm BOTH --iterations 10

# 仅运行遗传算法
python run_optimization.py --target TNF --algorithm GA --population 100 --iterations 20

# 仅运行贝叶斯优化
python run_optimization.py --target IFN --algorithm BO --init_samples 20 --iterations 15
```

### 8. 系统监控脚本
创建 `monitor_resources.sh`：
```bash
#!/bin/bash
# 监控系统资源使用情况

echo "=== 系统资源监控 ==="
echo "CPU使用率: $(top -bn1 | grep "Cpu(s)" | awk '{print $2}')%"
echo "内存使用: $(free -h | grep Mem | awk '{print $3"/"$2}')"
echo "GPU使用情况:"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv
```

给脚本执行权限：
```bash
chmod +x monitor_resources.sh
./monitor_resources.sh
```

### 9. 环境导出
```bash
# 导出环境配置
pip freeze > requirements_lock.txt

# 在其他机器上恢复环境
pip install -r requirements_lock.txt
```

以下是根据论文《Data-Driven Design of Random Heteropolypeptides as Synthetic Polyclonal Antibodies》中“Design”部分描述的完整代码实现，包括**遗传算法（GA）** 和**贝叶斯优化（BO）** 两种优化算法，用于从 RHP 组合空间中高效搜索高选择性结合能力的合成多克隆抗体。

---

## 🧬 完整代码实现

```python
import numpy as np
import torch
import botorch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from botorch.acquisition import qNoisyExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from sklearn.preprocessing import normalize

# ================================
# 遗传算法（Genetic Algorithm）
# ================================

class GeneticAlgorithm:
    def __init__(self, population_size, num_generations, mutation_rate=0.1):
        self.population_size = population_size
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate

    def initialize_population(self, num_components=8):
        """初始化种群：每个个体是一个8维向量，表示各组分摩尔分数"""
        population = np.random.rand(self.population_size, num_components)
        return normalize(population, norm='l1', axis=1)

    def evaluate_fitness(self, population, score_function):
        """评估种群中每个个体的适应度（Score）"""
        return np.array([score_function(ind) for ind in population])

    def select_parents(self, population, fitness, num_parents):
        """选择适应度最高的个体作为父代"""
        indices = np.argsort(fitness)[-num_parents:]
        return population[indices]

    def crossover(self, parent1, parent2):
        """单点交叉"""
        point = np.random.randint(1, len(parent1))
        child = np.concatenate([parent1[:point], parent2[point:]])
        return child / np.sum(child)

    def mutate(self, individual):
        """高斯噪声突变 + 基因交换"""
        if np.random.rand() < self.mutation_rate:
            # 对非零基因添加高斯噪声
            mask = individual > 0
            noise = np.random.normal(0, 0.1, size=individual.shape)
            individual[mask] += noise[mask]
            individual = np.clip(individual, 0, 1)
            individual /= np.sum(individual)

            # 随机交换两个基因位置
            idx1, idx2 = np.random.choice(len(individual), 2, replace=False)
            individual[idx1], individual[idx2] = individual[idx2], individual[idx1]

        return individual

    def run(self, score_function, num_components=8):
        """运行遗传算法"""
        population = self.initialize_population(num_components)
        best_individual = None
        best_fitness = -np.inf

        for gen in range(self.num_generations):
            fitness = self.evaluate_fitness(population, score_function)
            parents = self.select_parents(population, fitness, self.population_size // 2)

            # 生成下一代
            next_generation = []
            for _ in range(self.population_size):
                p1, p2 = parents[np.random.choice(len(parents), 2, replace=False)]
                child = self.crossover(p1, p2)
                child = self.mutate(child)
                next_generation.append(child)

            population = np.array(next_generation)

            # 更新最佳个体
            current_best_fitness = np.max(fitness)
            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                best_individual = population[np.argmax(fitness)]

            print(f"Generation {gen+1}, Best Fitness: {best_fitness:.4f}")

        return best_individual, best_fitness


# ================================
# 贝叶斯优化（Bayesian Optimization）
# ================================

class BayesianOptimization:
    def __init__(self, n_init=10, n_iter=30, noise_var=0.12):
        self.n_init = n_init
        self.n_iter = n_iter
        self.noise_var = noise_var
        self.X = None
        self.Y = None

    def initialize_data(self, score_function, num_components=8):
        """初始化随机样本"""
        X = np.random.rand(self.n_init, num_components)
        X = normalize(X, norm='l1', axis=1)
        Y = np.array([score_function(x) for x in X]).reshape(-1, 1)
        return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)

    def get_model(self, X, Y):
        """构建高斯过程代理模型"""
        model = SingleTaskGP(X, Y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)
        return model

    def optimize_acquisition(self, model, bounds, q=1):
        """使用 qNEI 获取新候选点"""
        acq_func = qNoisyExpectedImprovement(model, self.X, prune_baseline=True)
        candidates, _ = optimize_acqf(
            acq_func, bounds, q=q, num_restarts=20, raw_samples=512
        )
        return candidates

    def run(self, score_function, num_components=8):
        """运行贝叶斯优化"""
        bounds = torch.tensor([[0.] * num_components, [1.] * num_components])
        self.X, self.Y = self.initialize_data(score_function, num_components)

        for i in range(self.n_iter):
            model = self.get_model(self.X, self.Y)
            candidates = self.optimize_acquisition(model, bounds, q=1)
            new_X = candidates.detach()
            new_Y = torch.tensor([score_function(x.numpy()) for x in new_X]).reshape(-1, 1)

            self.X = torch.cat([self.X, new_X])
            self.Y = torch.cat([self.Y, new_Y])

            print(f"BO Iteration {i+1}, Best Score: {self.Y.max().item():.4f}")

        best_idx = self.Y.argmax()
        return self.X[best_idx].numpy(), self.Y[best_idx].item()


# ================================
# 示例评分函数（需根据实验数据自定义）
# ================================

def example_score_function(composition):
    """
    输入：8维向量，表示 RHP 中各组分摩尔分数
    输出：Score = Target/mean(Target) - Control/mean(Control)
    """
    # 这里应替换为实际的 ELISA 数据或模拟函数
    target_mean = 1.0  # 应根据随机采样点计算
    control_mean = 1.0
    target_signal = np.dot(composition, np.random.randn(8)) + 1.0
    control_signal = np.dot(composition, np.random.randn(8)) + 0.5
    score = target_signal / target_mean - control_signal / control_mean
    return score


# ================================
# 主程序：运行两种优化算法
# ================================

if __name__ == "__main__":
    # 遗传算法
    print("Running Genetic Algorithm...")
    ga = GeneticAlgorithm(population_size=50, num_generations=10)
    best_ga, score_ga = ga.run(example_score_function)
    print(f"GA Best: {best_ga}, Score: {score_ga:.4f}")

    # 贝叶斯优化
    print("\nRunning Bayesian Optimization...")
    bo = BayesianOptimization(n_init=10, n_iter=20)
    best_bo, score_bo = bo.run(example_score_function)
    print(f"BO Best: {best_bo}, Score: {score_bo:.4f}")
```

---

## 📌 说明

- **遗传算法（GA）**：
  - 初始化种群 → 选择父代 → 交叉 → 突变 → 评估适应度 → 迭代优化。
  - 适应度函数为 `Score`，目标是最大化该值。

- **贝叶斯优化（BO）**：
  - 使用 `SingleTaskGP` 作为代理模型，`qNoisyExpectedImprovement` 作为采集函数。
  - 每次迭代提出一个新候选点，更新模型并继续。

- **评分函数**：
  - 此处为示例函数，实际应替换为基于 ELISA 实验数据的真实评分函数。

---

如果需要我为你**适配真实数据接口**或**部署到自动化实验平台**，请告诉我你的具体数据格式或系统环境。
