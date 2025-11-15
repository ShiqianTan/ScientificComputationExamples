我将尝试复现论文中的A*算法用于纳米粒子合成优化。这个实现将基于论文中描述的方法，但由于缺乏真实的实验硬件，我将创建一个模拟环境。

```python
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any
import heapq
from dataclasses import dataclass
from collections import defaultdict
import warnings

@dataclass
class ParameterPoint:
    """表示参数空间中的一个点"""
    params: Dict[str, float]  # 参数名称 -> 参数值
    z_score: float = 0.0      # 评估得分 (Z_LSPR, Z_FWHM, Z_Ratio)
    s_score: float = 0.0      # 启发式得分 (S_UCB)
    visited: bool = False      # 是否已经实验验证
    n_visits: int = 0         # 访问次数
    
    def __lt__(self, other):
        return self.s_score < other.s_score

class NanoparticleAStar:
    """
    基于A*算法的纳米粒子合成参数优化
    根据论文中的方法实现
    """
    
    def __init__(self, 
                 target_lspr: float = 700.0,
                 target_fwhm: float = 50.0, 
                 target_ratio: float = 3.0,
                 param_ranges: Dict[str, Tuple[float, float]] = None,
                 step_sizes: Dict[str, float] = None):
        """
        初始化A*算法优化器
        
        Args:
            target_lspr: 目标LSPR波长 (nm)
            target_fwhm: 目标半高宽 (nm)
            target_ratio: 目标LSPR/TSPR峰比值
            param_ranges: 参数范围 {参数名: (最小值, 最大值)}
            step_sizes: 参数步长 {参数名: 步长}
        """
        
        # 目标参数
        self.target_lspr = target_lspr
        self.target_fwhm = target_fwhm
        self.target_ratio = target_ratio
        
        # 默认参数范围和步长 (基于论文中的典型值)
        if param_ranges is None:
            self.param_ranges = {
                'HCL_concentration': (0.1, 2.0),  # mM
                'AgNO3_dosage': (0.01, 0.1),      # mL
                'seed_amount': (0.01, 0.1),       # mL
                'CTAB_concentration': (0.05, 0.2), # M
                'temperature': (25, 45)           # °C
            }
        else:
            self.param_ranges = param_ranges
            
        if step_sizes is None:
            self.step_sizes = {
                'HCL_concentration': 0.1,
                'AgNO3_dosage': 0.005,
                'seed_amount': 0.005,
                'CTAB_concentration': 0.01,
                'temperature': 2.0
            }
        else:
            self.step_sizes = step_sizes
            
        # 超参数 (基于论文)
        self.A = 0.1    # UCB探索参数
        self.B = 200.0  # LSPR归一化范围
        self.C = 100.0  # FWHM归一化范围  
        self.D = 5.0    # Ratio归一化范围
        self.L = 0.1    # 局部搜索范围常数
        
        # 算法状态
        self.open_set = []  # 优先队列 (最小堆，但我们需要最大堆，所以用负值)
        self.closed_set = set()  # 已探索参数点
        self.all_points = {}     # 所有参数点
        self.experiment_count = 0
        self.best_point = None
        
        # 模拟实验的结果存储
        self.experiment_results = {}
        
    def _params_to_key(self, params: Dict[str, float]) -> str:
        """将参数字典转换为唯一键"""
        return tuple(sorted(params.items()))
    
    def _simulate_nanoparticle_synthesis(self, params: Dict[str, float]) -> Dict[str, float]:
        """
        模拟纳米粒子合成实验
        在实际系统中，这会调用真实的实验设备
        这里我们使用一个简化的物理模型来模拟
        """
        # 基于论文中的关系构建简化模型
        # LSPR与AgNO3用量和HCl浓度相关
        base_lspr = 600.0
        lspr_effect = (params['AgNO3_dosage'] - 0.05) * 800 + \
                     (params['HCL_concentration'] - 1.0) * 50 + \
                     (params['seed_amount'] - 0.05) * 200
        
        # 添加一些随机性模拟实验误差
        noise = np.random.normal(0, 5)  # 5nm的噪声
        
        lspr = base_lspr + lspr_effect + noise
        lspr = max(400, min(900, lspr))  # 限制在合理范围内
        
        # FWHM与浓度均匀性相关
        fwhm = 30 + abs(params['CTAB_concentration'] - 0.1) * 100 + \
               abs(params['temperature'] - 30) * 0.5 + np.random.normal(0, 2)
        fwhm = max(20, min(120, fwhm))
        
        # LSPR/TSPR ratio与种子量和生长条件相关
        ratio = 2.0 + (params['seed_amount'] - 0.05) * 20 + \
                (params['AgNO3_dosage'] - 0.05) * 10 + np.random.normal(0, 0.3)
        ratio = max(1.0, min(6.0, ratio))
        
        return {
            'LSPR': lspr,
            'FWHM': fwhm, 
            'Ratio': ratio
        }
    
    def evaluate_z_scores(self, experimental_data: Dict[str, float]) -> Dict[str, float]:
        """
        计算评估得分Z (论文中的公式1-3)
        """
        z_scores = {}
        
        # Z_LSPR = 1 - |y_LSPR - t| / B
        z_scores['LSPR'] = 1 - abs(experimental_data['LSPR'] - self.target_lspr) / self.B
        
        # Z_FWHM = 1 - y_FWHM / C  
        z_scores['FWHM'] = 1 - experimental_data['FWHM'] / self.C
        
        # Z_Ratio = y_Ratio / D
        z_scores['Ratio'] = experimental_data['Ratio'] / self.D
        
        # 总体Z得分 (论文中未明确说明如何组合，这里使用加权平均)
        z_scores['overall'] = 0.5 * z_scores['LSPR'] + 0.3 * z_scores['FWHM'] + 0.2 * z_scores['Ratio']
        
        return z_scores
    
    def calculate_heuristic_score(self, z_score: float, n_visits: int, total_experiments: int) -> float:
        """
        计算启发式得分S (论文中的公式4)
        S = Z + A * sqrt(ln(n) / N)
        """
        if n_visits == 0:
            return float('inf')  # 未访问的点具有无限潜力
        
        exploration_term = self.A * np.sqrt(np.log(total_experiments) / n_visits)
        return z_score + exploration_term
    
    def generate_neighbors(self, point: ParameterPoint) -> List[Dict[str, float]]:
        """
        在参数点周围生成邻居点 (论文中的Subs集合生成)
        """
        neighbors = []
        base_params = point.params
        
        # 为每个参数生成邻域点
        for param_name, current_value in base_params.items():
            if param_name not in self.step_sizes:
                continue
                
            step = self.step_sizes[param_name]
            min_val, max_val = self.param_ranges[param_name]
            
            # 生成减少步长的邻居
            new_value = current_value - step
            if new_value >= min_val:
                neighbor_params = base_params.copy()
                neighbor_params[param_name] = new_value
                neighbors.append(neighbor_params)
            
            # 生成增加步长的邻居  
            new_value = current_value + step
            if new_value <= max_val:
                neighbor_params = base_params.copy()
                neighbor_params[param_name] = new_value
                neighbors.append(neighbor_params)
        
        return neighbors
    
    def run_experiment(self, params: Dict[str, float]) -> ParameterPoint:
        """
        运行单个实验并返回结果点
        """
        # 模拟实验
        experimental_data = self._simulate_nanoparticle_synthesis(params)
        
        # 计算评估得分
        z_scores = self.evaluate_z_scores(experimental_data)
        
        # 创建参数点
        point = ParameterPoint(
            params=params.copy(),
            z_score=z_scores['overall'],
            visited=True,
            n_visits=1
        )
        
        # 存储实验结果
        param_key = self._params_to_key(params)
        self.experiment_results[param_key] = {
            'experimental_data': experimental_data,
            'z_scores': z_scores
        }
        
        self.experiment_count += 1
        
        return point
    
    def initialize_algorithm(self, initial_params: List[Dict[str, float]] = None):
        """
        初始化算法状态
        """
        if initial_params is None:
            # 生成随机初始点
            initial_params = []
            for _ in range(3):
                params = {}
                for param_name, (min_val, max_val) in self.param_ranges.items():
                    params[param_name] = np.random.uniform(min_val, max_val)
                initial_params.append(params)
        
        # 运行初始实验
        for params in initial_params:
            point = self.run_experiment(params)
            param_key = self._params_to_key(params)
            self.all_points[param_key] = point
            
            # 计算初始启发式得分并加入open set
            point.s_score = self.calculate_heuristic_score(
                point.z_score, point.n_visits, self.experiment_count
            )
            
            # 使用负值实现最大堆
            heapq.heappush(self.open_set, (-point.s_score, param_key))
            
            # 更新最佳点
            if self.best_point is None or point.z_score > self.best_point.z_score:
                self.best_point = point
    
    def run_optimization(self, max_experiments: int = 100, convergence_threshold: float = 0.95):
        """
        运行A*算法优化
        
        Args:
            max_experiments: 最大实验次数
            convergence_threshold: 收敛阈值 (Z得分)
            
        Returns:
            优化历史记录
        """
        optimization_history = {
            'experiment_count': [],
            'best_z_score': [],
            'best_lspr': [],
            'best_fwhm': [],
            'best_ratio': [],
            'parameters': []
        }
        
        print("Starting A* optimization...")
        print(f"Target: LSPR={self.target_lspr}nm, FWHM={self.target_fwhm}nm, Ratio={self.target_ratio}")
        print("-" * 60)
        
        while self.experiment_count < max_experiments and len(self.open_set) > 0:
            # 从open set中选择最佳点
            best_score_neg, best_key = heapq.heappop(self.open_set)
            best_point = self.all_points[best_key]
            
            # 检查是否收敛
            if best_point.z_score >= convergence_threshold:
                print(f"Converged after {self.experiment_count} experiments!")
                break
            
            # 生成邻居点
            neighbors_params = self.generate_neighbors(best_point)
            
            # 筛选未探索的邻居
            new_neighbors = []
            for neighbor_params in neighbors_params:
                neighbor_key = self._params_to_key(neighbor_params)
                if neighbor_key not in self.all_points and neighbor_key not in self.closed_set:
                    new_neighbors.append(neighbor_params)
            
            # 运行新实验
            for neighbor_params in new_neighbors:
                neighbor_point = self.run_experiment(neighbor_params)
                neighbor_key = self._params_to_key(neighbor_params)
                
                # 存储点并计算启发式得分
                neighbor_point.s_score = self.calculate_heuristic_score(
                    neighbor_point.z_score, neighbor_point.n_visits, self.experiment_count
                )
                self.all_points[neighbor_key] = neighbor_point
                
                # 加入open set
                heapq.heappush(self.open_set, (-neighbor_point.s_score, neighbor_key))
                
                # 更新最佳点
                if neighbor_point.z_score > self.best_point.z_score:
                    self.best_point = neighbor_point
            
            # 将当前点标记为已探索
            self.closed_set.add(best_key)
            best_point.n_visits += 1
            
            # 重新计算当前点的启发式得分并可能重新加入open set
            best_point.s_score = self.calculate_heuristic_score(
                best_point.z_score, best_point.n_visits, self.experiment_count
            )
            heapq.heappush(self.open_set, (-best_point.s_score, best_key))
            
            # 记录优化历史
            if self.experiment_count % 5 == 0 or self.experiment_count <= 10:
                best_data = self.experiment_results[self._params_to_key(self.best_point.params)]
                
                optimization_history['experiment_count'].append(self.experiment_count)
                optimization_history['best_z_score'].append(self.best_point.z_score)
                optimization_history['best_lspr'].append(best_data['experimental_data']['LSPR'])
                optimization_history['best_fwhm'].append(best_data['experimental_data']['FWHM'])
                optimization_history['best_ratio'].append(best_data['experimental_data']['Ratio'])
                optimization_history['parameters'].append(self.best_point.params.copy())
                
                print(f"Experiment {self.experiment_count:3d}: "
                      f"Z={self.best_point.z_score:.3f}, "
                      f"LSPR={best_data['experimental_data']['LSPR']:.1f}nm, "
                      f"FWHM={best_data['experimental_data']['FWHM']:.1f}nm, "
                      f"Ratio={best_data['experimental_data']['Ratio']:.2f}")
        
        return optimization_history
    
    def plot_optimization_history(self, history: Dict[str, List]):
        """绘制优化历史"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # Z得分历史
        ax1.plot(history['experiment_count'], history['best_z_score'], 'b-o', linewidth=2)
        ax1.axhline(y=0.95, color='r', linestyle='--', alpha=0.7, label='Convergence Threshold')
        ax1.set_xlabel('Experiment Count')
        ax1.set_ylabel('Best Z Score')
        ax1.set_title('Optimization Progress (Z Score)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # LSPR历史
        ax2.plot(history['experiment_count'], history['best_lspr'], 'g-o', linewidth=2)
        ax2.axhline(y=self.target_lspr, color='r', linestyle='--', alpha=0.7, label='Target LSPR')
        ax2.set_xlabel('Experiment Count')
        ax2.set_ylabel('LSPR (nm)')
        ax2.set_title('Longitudinal Surface Plasmon Resonance')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # FWHM历史
        ax3.plot(history['experiment_count'], history['best_fwhm'], 'm-o', linewidth=2)
        ax3.axhline(y=self.target_fwhm, color='r', linestyle='--', alpha=0.7, label='Target FWHM')
        ax3.set_xlabel('Experiment Count')
        ax3.set_ylabel('FWHM (nm)')
        ax3.set_title('Full Width at Half Maximum')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Ratio历史
        ax4.plot(history['experiment_count'], history['best_ratio'], 'c-o', linewidth=2)
        ax4.axhline(y=self.target_ratio, color='r', linestyle='--', alpha=0.7, label='Target Ratio')
        ax4.set_xlabel('Experiment Count')
        ax4.set_ylabel('LSPR/TSPR Ratio')
        ax4.set_title('Peak Intensity Ratio')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

# 示例使用
def demo_a_star_optimization():
    """演示A*算法优化"""
    
    # 设置目标参数 (基于论文中的示例)
    optimizer = NanoparticleAStar(
        target_lspr=750.0,    # 目标LSPR波长
        target_fwhm=45.0,     # 目标半高宽
        target_ratio=3.5,     # 目标峰比值
    )
    
    # 初始化算法
    optimizer.initialize_algorithm()
    
    # 运行优化
    history = optimizer.run_optimization(
        max_experiments=80,
        convergence_threshold=0.92
    )
    
    # 显示最终结果
    print("\n" + "="*60)
    print("FINAL OPTIMIZATION RESULTS:")
    print("="*60)
    
    best_params = optimizer.best_point.params
    best_data = optimizer.experiment_results[optimizer._params_to_key(best_params)]
    
    print(f"Best Z Score: {optimizer.best_point.z_score:.4f}")
    print(f"Experiments Conducted: {optimizer.experiment_count}")
    print(f"\nBest Parameters:")
    for param, value in best_params.items():
        print(f"  {param}: {value:.4f}")
    
    print(f"\nExperimental Results:")
    exp_data = best_data['experimental_data']
    print(f"  LSPR: {exp_data['LSPR']:.1f} nm (target: {optimizer.target_lspr} nm)")
    print(f"  FWHM: {exp_data['FWHM']:.1f} nm (target: {optimizer.target_fwhm} nm)") 
    print(f"  Ratio: {exp_data['Ratio']:.2f} (target: {optimizer.target_ratio})")
    
    print(f"\nZ Scores:")
    z_scores = best_data['z_scores']
    print(f"  Z_LSPR: {z_scores['LSPR']:.4f}")
    print(f"  Z_FWHM: {z_scores['FWHM']:.4f}")
    print(f"  Z_Ratio: {z_scores['Ratio']:.4f}")
    
    # 绘制优化历史
    optimizer.plot_optimization_history(history)
    
    return optimizer, history

# 运行演示
if __name__ == "__main__":
    optimizer, history = demo_a_star_optimization()
```

这个实现包含了论文中A*算法的关键要素：

## 核心特性：

1. **参数空间表示**：使用离散的参数网格，每个参数有明确的取值范围和步长

2. **评估函数**：实现了论文中的Z得分计算：
   - `Z_LSPR = 1 - |y_LSPR - t| / B`
   - `Z_FWHM = 1 - y_FWHM / C` 
   - `Z_Ratio = y_Ratio / D`

3. **启发式函数**：实现了UCB策略：
   - `S = Z + A * sqrt(ln(n) / N)`

4. **邻居生成**：在参数点周围按照设定步长生成新的实验点

5. **Open/Closed集合管理**：维护待探索和已探索的参数点

## 使用说明：

1. **初始化优化器**：设置目标参数和参数空间
2. **运行优化**：算法会自动探索参数空间并找到最优解
3. **监控进度**：实时显示优化过程和最终结果

## 与实际系统的集成：

要将其用于真实的实验系统，需要：
1. 替换`_simulate_nanoparticle_synthesis`方法为真实的实验调用
2. 集成实验室自动化设备的控制接口
3. 连接UV-vis光谱仪等表征设备的数据采集

这个实现提供了论文中A*算法的完整框架，可以进一步扩展以适应特定的实验需求。
