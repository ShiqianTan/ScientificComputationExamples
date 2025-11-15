基于论文中描述的 SCIGEN 算法，我提供以下 Python 风格的伪代码：

```python
import torch
import numpy as np
from typing import Tuple, Dict

class SCIGEN:
    """
    Structural Constraint Integration in a GENerative model
    用于在生成模型中集成结构约束的框架
    """
    
    def __init__(self, base_diffusion_model, T: int = 1000):
        """
        初始化 SCIGEN
        
        Args:
            base_diffusion_model: 预训练的基扩散模型
            T: 扩散步数
        """
        self.base_model = base_diffusion_model
        self.T = T
        
    def initialize_constrained_structure(self, 
                                       lattice_type: str,
                                       magnetic_atoms: list,
                                       bond_lengths: dict,
                                       num_atoms: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        初始化约束结构
        
        Args:
            lattice_type: 晶格类型 ('triangular', 'honeycomb', 'kagome')
            magnetic_atoms: 磁性原子列表
            bond_lengths: 键长分布字典
            num_atoms: 单胞中原子总数
            
        Returns:
            M0_c: 初始约束结构
            mask: 约束掩码
        """
        # 1. 根据晶格类型确定顶点数和晶胞尺寸
        if lattice_type == 'triangular':
            vertices_per_cell = 3
            lattice_vectors = self._calculate_lattice_vectors(bond_lengths, scale=1.0)
        elif lattice_type == 'honeycomb':
            vertices_per_cell = 4
            lattice_vectors = self._calculate_lattice_vectors(bond_lengths, scale=np.sqrt(3))
        elif lattice_type == 'kagome':
            vertices_per_cell = 6
            lattice_vectors = self._calculate_lattice_vectors(bond_lengths, scale=2.0)
        
        # 2. 创建约束掩码
        mask = self._create_constraint_mask(vertices_per_cell, num_atoms, lattice_vectors)
        
        # 3. 初始化约束结构
        M0_c = self._create_initial_constrained_structure(
            lattice_vectors, vertices_per_cell, magnetic_atoms, num_atoms
        )
        
        return M0_c, mask
    
    def generate_with_constraints(self, 
                                M0_c: torch.Tensor,
                                mask: torch.Tensor) -> torch.Tensor:
        """
        带约束的生成过程 - 算法1的核心实现
        
        Args:
            M0_c: 初始约束结构
            mask: 约束掩码
            
        Returns:
            M0: 最终生成的结构
        """
        # 步骤1: 为约束结构创建预定义的加噪路径
        M_t_c_path = self._create_constrained_diffusion_path(M0_c)
        
        # 步骤2: 初始化无约束结构（完全噪声）
        M_t_u = self._sample_from_prior()  # 从先验分布采样
        
        # 步骤3: 初始融合
        M_t = self._fuse_structures(M_t_c_path[self.T], M_t_u, mask)
        
        # 步骤4: 迭代去噪过程 (t = T, T-1, ..., 1)
        for t in range(self.T, 0, -1):
            # 获取当前时间步的约束结构
            M_t_c = M_t_c_path[t]
            
            # 算法1第5行: 对约束结构进行扩散
            M_t_minus_1_c = self.diffusion_inference_model(M_t_c, t, t-1)
            
            # 算法1第6行: 对融合结构进行去噪
            M_t_minus_1_u = self.denoising_generative_model(M_t, t, t-1)
            
            # 算法1第7行: 融合约束和无约束部分
            M_t_minus_1 = self._fuse_structures(M_t_minus_1_c, M_t_minus_1_u, mask)
            
            M_t = M_t_minus_1  # 更新当前结构
        
        return M_t  # 返回最终结构 M0
    
    def _fuse_structures(self, 
                        constrained_struct: torch.Tensor,
                        unconstrained_struct: torch.Tensor,
                        mask: torch.Tensor) -> torch.Tensor:
        """
        融合约束和无约束结构 (算法1第3、7行)
        
        M = mask ⊙ M_c + (1 - mask) ⊙ M_u
        """
        return mask * constrained_struct + (1 - mask) * unconstrained_struct
    
    def _create_constraint_mask(self, 
                              vertices_count: int,
                              total_atoms: int,
                              lattice_vectors: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        创建约束掩码
        
        Returns:
            mask: 包含 {'lattice', 'fractional', 'atom_type'} 的掩码字典
        """
        mask = {}
        
        # 晶格掩码: 约束前两个晶格向量 (定义AL层平面)
        mask['lattice'] = torch.tensor([1.0, 1.0, 0.0])  # 约束 l1, l2，不约束 l3
        
        # 分数坐标掩码: 约束顶点原子的位置
        fractional_mask = torch.zeros(total_atoms)
        fractional_mask[:vertices_count] = 1.0  # 前vertices_count个原子在顶点位置
        mask['fractional'] = fractional_mask
        
        # 原子类型掩码: 约束顶点原子的类型为磁性原子
        atom_type_mask = torch.zeros(total_atoms)
        atom_type_mask[:vertices_count] = 1.0
        mask['atom_type'] = atom_type_mask
        
        return mask
    
    def _create_constrained_diffusion_path(self, M0_c: torch.Tensor) -> Dict[int, torch.Tensor]:
        """
        为约束结构创建预定义的扩散路径
        
        Args:
            M0_c: 初始约束结构
            
        Returns:
            path: 字典 {时间步: 带噪声的约束结构}
        """
        path = {}
        M_current = M0_c.clone()
        
        for t in range(1, self.T + 1):
            # 使用扩散推断模型逐步添加噪声
            M_current = self.diffusion_inference_model(M_current, t-1, t)
            path[t] = M_current.clone()
            
        return path


# DiffCSP 特定实现的伪代码 (算法2)
class SCIGENWithDiffCSP(SCIGEN):
    """
    基于 DiffCSP 基础模型的 SCIGEN 实现
    """
    
    def generate_with_constraints_diffcsp(self,
                                        M0_c: Dict[str, torch.Tensor],
                                        mask: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        DiffCSP 特定的约束生成过程 (算法2)
        
        Args:
            M0_c: 包含 {'lattice', 'fractional', 'atom_type'} 的约束结构
            mask: 对应的约束掩码
            
        Returns:
            M0: 最终生成的结构
        """
        # 步骤2: 从先验分布采样初始无约束结构
        M_t_u = {
            'lattice': torch.randn_like(M0_c['lattice']),
            'fractional': torch.randn_like(M0_c['fractional']),
            'atom_type': torch.randn_like(M0_c['atom_type'])
        }
        
        # 步骤3: 初始融合
        M_t = self._fuse_structures_diffcsp(M_t_c_path[self.T], M_t_u, mask)
        
        # 迭代去噪过程
        for t in range(self.T, 0, -1):
            # 步骤5: 对约束结构进行扩散
            M_t_minus_1_c = self._diffuse_constrained_diffcsp(M_t_c_path[t], t, t-1)
            
            # 步骤6-8: 对无约束部分进行去噪
            M_t_minus_1_u = self._denoise_unconstrained_diffcsp(M_t, t, t-1)
            
            # 步骤7-8, 13: 分别融合晶格、原子类型和分数坐标
            M_t_minus_1 = {
                'lattice': self._fuse_component(
                    M_t_minus_1_c['lattice'], M_t_minus_1_u['lattice'], mask['lattice']
                ),
                'atom_type': self._fuse_component(
                    M_t_minus_1_c['atom_type'], M_t_minus_1_u['atom_type'], mask['atom_type']
                ),
                'fractional': self._fuse_component(
                    M_t_minus_1_c['fractional'], M_t_minus_1_u['fractional'], mask['fractional']
                )
            }
            
            M_t = M_t_minus_1
        
        return M_t
    
    def _denoise_unconstrained_diffcsp(self, 
                                     M_t: Dict[str, torch.Tensor],
                                     t: int, 
                                     t_minus_1: int) -> Dict[str, torch.Tensor]:
        """
        DiffCSP 特定的无约束部分去噪
        """
        # 步骤10-12: 使用预测器-校正器机制处理分数坐标
        L_t_minus_1 = self.denoising_model_lattice(M_t, t, t_minus_1)
        A_t_minus_1 = self.denoising_model_atom_type(M_t, t, t_minus_1)
        
        # 预测器步骤
        F_t_minus_1_half = self.predictor_model(M_t, t, t_minus_1)
        
        # 校正器步骤
        F_t_minus_1 = self.corrector_model(F_t_minus_1_half, L_t_minus_1, A_t_minus_1)
        
        return {
            'lattice': L_t_minus_1,
            'fractional': F_t_minus_1,
            'atom_type': A_t_minus_1
        }


# 使用示例
def example_usage():
    """
    SCIGEN 使用示例
    """
    # 1. 加载预训练的基础扩散模型
    base_model = load_pretrained_diffusion_model('DiffCSP')
    
    # 2. 初始化 SCIGEN
    scigen = SCIGENWithDiffCSP(base_model, T=1000)
    
    # 3. 定义约束条件
    lattice_type = 'kagome'
    magnetic_atoms = ['Fe', 'Co', 'Ni']  # 磁性原子
    bond_lengths = {'Fe': 2.5, 'Co': 2.4, 'Ni': 2.3}  # 键长分布
    num_atoms = 12  # 单胞中原子总数
    
    # 4. 初始化约束结构和掩码
    M0_c, mask = scigen.initialize_constrained_structure(
        lattice_type, magnetic_atoms, bond_lengths, num_atoms
    )
    
    # 5. 生成带约束的材料结构
    generated_structure = scigen.generate_with_constraints_diffcsp(M0_c, mask)
    
    # 6. 后续稳定性筛选
    stable_candidates = prescreen_stability(generated_structure)
    
    return stable_candidates


def prescreen_stability(structures):
    """
    四阶段稳定性预筛选过程
    """
    candidates = structures
    
    # 阶段1: 电荷中性筛选
    candidates = stage1_charge_neutrality(candidates)
    
    # 阶段2: 空间占据率筛选
    candidates = stage2_space_occupancy(candidates, threshold=1.7)
    
    # 阶段3: GNN 形成能分类器
    candidates = stage3_gnn_energy_above_hull(candidates, threshold=0.1)
    
    # 阶段4: GNN 稳定结构判别器
    candidates = stage4_gnn_stability_classifier(candidates)
    
    return candidates


# 简化的稳定性筛选函数
def stage1_charge_neutrality(structures):
    """阶段1: 电荷中性筛选"""
    neutral_structures = []
    for struct in structures:
        if is_charge_neutral(struct):
            neutral_structures.append(struct)
    return neutral_structures

def stage2_space_occupancy(structures, threshold=1.7):
    """阶段2: 空间占据率筛选"""
    filtered_structures = []
    for struct in structures:
        R_occ = calculate_space_occupancy_ratio(struct)
        if R_occ <= threshold:
            filtered_structures.append(struct)
    return filtered_structures

def stage3_gnn_energy_above_hull(structures, threshold=0.1):
    """阶段3: GNN 形成能分类器"""
    stable_structures = []
    for struct in structures:
        e_hull = gnn_classifier.predict_energy_above_hull(struct)
        if e_hull <= threshold:
            stable_structures.append(struct)
    return stable_structures
```

这个伪代码完整地实现了论文中描述的 SCIGEN 算法，包括：

1. **核心融合算法**：实现了掩码指导的约束-无约束结构融合
2. **DiffCSP 特定实现**：处理了预测器-校正器机制
3. **约束初始化**：支持不同几何图案的初始化
4. **稳定性筛选**：实现了四阶段预筛选流程

关键创新点在于**无需重新训练基础模型**，而是通过迭代的掩码融合过程将几何约束注入到生成过程中。
