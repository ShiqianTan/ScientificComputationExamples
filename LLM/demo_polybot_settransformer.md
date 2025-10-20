我将基于论文描述和Set-Transformer架构，完整实现一个可以处理共聚物、预测不确定性并接入自动化工作流的预测模型。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional
import json
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import warnings
warnings.filterwarnings('ignore')

# Set-Transformer 核心模块 (基于官方实现)
class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = torch.softmax(Q_.bmm(K_.transpose(1,2))/np.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O

class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(X, X)

class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)

class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)

class SetTransformer(nn.Module):
    def __init__(self, dim_input, dim_output, num_inds=32, dim_hidden=128, num_heads=4, ln=False):
        super(SetTransformer, self).__init__()
        self.enc = nn.Sequential(
            ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln),
            ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln)
        )
        self.dec = nn.Sequential(
            PMA(dim_hidden, num_heads, 1, ln=ln),
            nn.Linear(dim_hidden, dim_output)
        )

    def forward(self, X):
        return self.dec(self.enc(X)).squeeze(-1)

# 共聚物预测模型 (双解码器 + 不确定性估计)
class CopolymerPredictor(nn.Module):
    def __init__(self, monomer_feature_dim=512, color_dim=3, spectrum_dim=300, 
                 num_inds=32, dim_hidden=256, num_heads=8, dropout_rate=0.1):
        super(CopolymerPredictor, self).__init__()
        
        # 编码器 - 处理共聚物单体集合
        self.encoder = SetTransformer(
            dim_input=monomer_feature_dim + 1,  # +1 for ratio
            dim_output=dim_hidden,
            num_inds=num_inds,
            dim_hidden=dim_hidden,
            num_heads=num_heads,
            ln=True
        )
        
        # 颜色解码器 - 预测 L*a*b* 值
        self.color_decoder = nn.Sequential(
            nn.Linear(dim_hidden, 128),
            nn.Dropout(dropout_rate),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.Dropout(dropout_rate),
            nn.ReLU(),
            nn.Linear(64, color_dim)  # L*, a*, b*
        )
        
        # 光谱解码器 - 预测吸收光谱
        self.spectrum_decoder = nn.Sequential(
            nn.Linear(dim_hidden, 128),
            nn.Dropout(dropout_rate),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.Dropout(dropout_rate),
            nn.ReLU(),
            nn.Linear(256, spectrum_dim)  # 300-point spectrum
        )
        
        self.dropout_rate = dropout_rate
        
    def forward(self, monomer_sets, mc_dropout=False):
        """
        Args:
            monomer_sets: List of tensors [batch_size, num_monomers, feature_dim+1]
            mc_dropout: Whether to use MC dropout for uncertainty estimation
        """
        # 编码共聚物
        encoded = self.encoder(monomer_sets)
        
        # 应用MC Dropout如果启用
        if mc_dropout:
            encoded = F.dropout(encoded, p=self.dropout_rate, training=True)
        
        # 双解码
        color_pred = self.color_decoder(encoded)
        spectrum_pred = self.spectrum_decoder(encoded)
        
        return color_pred, spectrum_pred
    
    def predict_with_uncertainty(self, monomer_sets, num_samples=50):
        """使用MC Dropout进行不确定性估计"""
        color_predictions = []
        spectrum_predictions = []
        
        self.train()  # 保持train模式以启用dropout
        
        with torch.no_grad():
            for _ in range(num_samples):
                color_pred, spectrum_pred = self.forward(monomer_sets, mc_dropout=True)
                color_predictions.append(color_pred.cpu().numpy())
                spectrum_predictions.append(spectrum_pred.cpu().numpy())
        
        color_predictions = np.array(color_predictions)
        spectrum_predictions = np.array(spectrum_predictions)
        
        color_mean = color_predictions.mean(axis=0)
        color_std = color_predictions.std(axis=0)
        spectrum_mean = spectrum_predictions.mean(axis=0)
        spectrum_std = spectrum_predictions.std(axis=0)
        
        return color_mean, color_std, spectrum_mean, spectrum_std

# 分子特征提取器
class MolecularFeatureExtractor:
    def __init__(self, fingerprint_size=2048, radius=3):
        self.fingerprint_size = fingerprint_size
        self.radius = radius
        
    def smiles_to_fingerprint(self, smiles):
        """将SMILES转换为Morgan指纹"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return np.zeros(self.fingerprint_size)
            fingerprint = AllChem.GetMorganFingerprintAsBitVect(
                mol, radius=self.radius, nBits=self.fingerprint_size
            )
            return np.array(fingerprint)
        except:
            return np.zeros(self.fingerprint_size)
    
    def prepare_copolymer_input(self, monomers_dict):
        """
        准备共聚物输入
        Args:
            monomers_dict: {'SMILES1': ratio1, 'SMILES2': ratio2, ...}
        """
        features = []
        ratios = []
        
        for smiles, ratio in monomers_dict.items():
            fp = self.smiles_to_fingerprint(smiles)
            features.append(fp)
            ratios.append(ratio)
        
        # 归一化比例
        ratios = np.array(ratios) / sum(ratios)
        
        # 合并特征和比例
        combined_features = []
        for i in range(len(features)):
            combined = np.concatenate([features[i], [ratios[i]]])
            combined_features.append(combined)
        
        return np.array(combined_features)

# 自动化工作流集成器
class AutonomousWorkflowManager:
    def __init__(self, model, feature_extractor, target_color, available_monomers):
        self.model = model
        self.feature_extractor = feature_extractor
        self.target_color = torch.tensor(target_color, dtype=torch.float32)  # [L*, a*, b*]
        self.available_monomers = available_monomers
        self.database = []
        self.iteration_count = 0
        
    def calculate_color_difference(self, predicted_color):
        """计算ΔE Lab颜色差异"""
        return torch.sqrt(torch.sum((predicted_color - self.target_color) ** 2)).item()
    
    def expected_improvement(self, predicted_color, uncertainty, best_deltaE):
        """期望改进采集函数"""
        deltaE = self.calculate_color_difference(predicted_color)
        improvement = best_deltaE - deltaE
        z = improvement / (uncertainty + 1e-8)
        ei = improvement * torch.special.erf(z) + uncertainty * torch.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)
        return ei.item()
    
    def select_next_candidates(self, num_candidates=6):
        """选择下一批实验候选"""
        if not self.database:
            # 第一轮：随机选择
            candidates = []
            for _ in range(num_candidates):
                num_mers = np.random.choice([2, 3])  # 2或3个单体
                selected_mers = np.random.choice(
                    list(self.available_monomers.keys()), 
                    num_mers, 
                    replace=False
                )
                ratios = np.random.dirichlet(np.ones(num_mers))
                candidate = {mer: ratio for mer, ratio in zip(selected_mers, ratios)}
                candidates.append(candidate)
            return candidates
        
        # 基于EI选择候选
        best_deltaE = min([exp['deltaE'] for exp in self.database])
        candidates = []
        eis = []
        
        # 生成候选并评估
        for _ in range(100):  # 生成大量候选进行筛选
            num_mers = np.random.choice([2, 3])
            selected_mers = np.random.choice(
                list(self.available_monomers.keys()), 
                num_mers, 
                replace=False
            )
            ratios = np.random.dirichlet(np.ones(num_mers))
            candidate_dict = {mer: ratio for mer, ratio in zip(selected_mers, ratios)}
            
            # 预测颜色和不确定性
            features = self.feature_extractor.prepare_copolymer_input(candidate_dict)
            features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            
            color_mean, color_std, _, _ = self.model.predict_with_uncertainty(features_tensor)
            predicted_color = torch.tensor(color_mean[0])
            uncertainty = torch.tensor(color_std[0].mean())
            
            ei = self.expected_improvement(predicted_color, uncertainty, best_deltaE)
            
            candidates.append(candidate_dict)
            eis.append(ei)
        
        # 选择EI最高的候选
        top_indices = np.argsort(eis)[-num_candidates:]
        return [candidates[i] for i in top_indices]
    
    def run_iteration(self, experimental_results=None):
        """运行一次迭代"""
        self.iteration_count += 1
        print(f"=== Iteration {self.iteration_count} ===")
        
        # 选择候选
        candidates = self.select_next_candidates()
        print(f"Selected {len(candidates)} candidates")
        
        if experimental_results is None:
            # 模拟实验阶段 - 在实际系统中这里会调用机器人执行实验
            experimental_results = []
            for i, candidate in enumerate(candidates):
                # 模拟实验测量
                features = self.feature_extractor.prepare_copolymer_input(candidate)
                features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
                color_mean, _, spectrum_mean, _ = self.model.predict_with_uncertainty(features_tensor)
                
                # 添加一些噪声模拟实验误差
                noise = np.random.normal(0, 0.5, 3)
                measured_color = color_mean[0] + noise
                deltaE = self.calculate_color_difference(torch.tensor(measured_color))
                
                result = {
                    'candidate': candidate,
                    'measured_color': measured_color.tolist(),
                    'measured_spectrum': spectrum_mean[0].tolist(),
                    'deltaE': deltaE
                }
                experimental_results.append(result)
                print(f"Candidate {i+1}: ΔE = {deltaE:.2f}")
        
        # 更新数据库
        self.database.extend(experimental_results)
        
        # 重新训练模型 (简化版 - 实际中需要完整训练)
        # self.retrain_model()
        
        # 检查终止条件
        best_deltaE = min([exp['deltaE'] for exp in experimental_results])
        print(f"Best ΔE in this iteration: {best_deltaE:.2f}")
        
        if best_deltaE < 3.0 or self.iteration_count >= 3:
            print("Optimization completed!")
            return True, experimental_results
        
        return False, experimental_results

# 使用示例和测试
def main():
    # 初始化组件
    feature_extractor = MolecularFeatureExtractor()
    model = CopolymerPredictor(
        monomer_feature_dim=2049,  # 2048-bit fingerprint + 1 ratio
        color_dim=3,
        spectrum_dim=300
    )
    
    # 定义可用单体 (示例)
    available_monomers = {
        'c1ccc(-c2ccccc2)cc1': 'Biphenyl',  # 联苯
        'c1ccc(-c2ccsc2)cc1': 'Phenylthiophene',  # 苯基噻吩
        'c1ccsc1': 'Thiophene',  # 噻吩
        'c1ccc(-c2cccc(-c3ccccc3)c2)cc1': 'Terphenyl',  # 三联苯
        'c1cncc(-c2ccccc2)c1': 'Phenylpyridine'  # 苯基吡啶
    }
    
    # 目标颜色 (L*, a*, b*)
    target_color = [65, -10, 30]  # 绿色
    
    # 初始化工作流管理器
    workflow = AutonomousWorkflowManager(
        model=model,
        feature_extractor=feature_extractor,
        target_color=target_color,
        available_monomers=available_monomers
    )
    
    # 运行优化循环
    for iteration in range(3):
        completed, results = workflow.run_iteration()
        if completed:
            break
    
    # 输出最佳结果
    best_result = min(workflow.database, key=lambda x: x['deltaE'])
    print(f"\n🎯 Best candidate: ΔE = {best_result['deltaE']:.2f}")
    print(f"Color: {best_result['measured_color']}")
    print(f"Composition: {best_result['candidate']}")

if __name__ == "__main__":
    main()
```

这个实现包含以下关键特性：

## 🎯 核心功能

1. **共聚物处理**：
   - 使用Set-Transformer处理可变数量的单体输入
   - 支持2-3个单体的共聚物组合
   - 考虑单体比例作为输入特征

2. **不确定性预测**：
   - 实现MC Dropout进行不确定性估计
   - 提供颜色和光谱预测的均值和标准差

3. **自动化工作流集成**：
   - 基于期望改进(EI)的候选选择
   - 迭代优化循环
   - 自动终止条件（ΔE < 3.0 或达到最大迭代次数）

## 🔧 扩展建议

要接入真实的自动化系统，还需要：

```python
# 机器人控制接口 (示例)
class RobotController:
    def __init__(self, chemspeed_ip, tecan_ip, ur5e_ip):
        self.chemspeed_ip = chemspeed_ip
        self.tecan_ip = tecan_ip
        self.ur5e_ip = ur5e_ip
    
    def execute_synthesis(self, candidate):
        """调用Chemspeed执行合成"""
        # 实现具体的机器人控制逻辑
        pass
    
    def perform_characterization(self, sample_id):
        """调用Tecan进行UV-Vis表征"""
        pass
    
    def transfer_sample(self, from_station, to_station):
        """调用UR5e机械臂转移样品"""
        pass

# 数据库集成
class ECPDatabase:
    def __init__(self, db_url):
        self.connection = create_engine(db_url)
    
    def store_experiment(self, candidate, results):
        """存储实验数据"""
        pass
    
    def get_training_data(self):
        """获取训练数据"""
        pass
```

这个实现提供了一个完整的框架，可以处理论文中描述的共聚物预测任务，并能够集成到自动化实验室工作流中形成闭环优化。
