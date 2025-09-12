from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np

def morgan_fingerprint_with_frequency(smiles, radius=2):
    """
    生成带频率的Morgan指纹
    
    参数:
        smiles: 分子的SMILES表示
        radius: 指纹半径，默认值为2
    
    返回:
        字典形式的带频率Morgan指纹，键为特征ID，值为出现频率
    """
    # 从SMILES创建分子对象
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"无法从SMILES '{smiles}'创建分子对象")
    
    # 生成带频率的Morgan指纹
    # useCounts=True表示记录频率而不仅仅是存在
    fp = AllChem.GetMorganFingerprint(mol, radius, useCounts=True)
    
    # 转换为字典形式返回
    return dict(fp.GetNonzeroElements())

def morgan_fingerprint_1024bit(smiles, radius=2):
    """
    生成1024位的Morgan指纹
    
    参数:
        smiles: 分子的SMILES表示
        radius: 指纹半径，默认值为2
    
    返回:
        长度为1024的二进制数组
    """
    # 从SMILES创建分子对象
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"无法从SMILES '{smiles}'创建分子对象")
    
    # 生成1024位的Morgan指纹
    # nBits=1024设置指纹长度，useFeatures=False使用原子环境而非特征
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=1024)
    
    # 转换为numpy数组返回
    return np.array(fp)

# 示例用法
if __name__ == "__main__":
    # 示例SMILES（聚乙烯的简化表示）
    polymer_smiles = "CC(CC)CC"
    
    # 生成带频率的Morgan指纹
    freq_fp = morgan_fingerprint_with_frequency(polymer_smiles)
    print(f"带频率的Morgan指纹特征数量: {len(freq_fp)}")
    print(f"带频率的Morgan指纹示例: {dict(list(freq_fp.items())[:5])}")  # 显示前5个特征
    
    # 生成1024位的Morgan指纹
    bit_fp = morgan_fingerprint_1024bit(polymer_smiles)
    print(f"\n1024位Morgan指纹长度: {len(bit_fp)}")
    print(f"1024位Morgan指纹中1的数量: {sum(bit_fp)}")
    print(f"1024位Morgan指纹前20位: {bit_fp[:20]}")
