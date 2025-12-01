import numpy as np
from Bio import SeqIO
from scipy.optimize import minimize

def compute_mfe(sequence):
    """
    计算RNA序列的最小自由能(MFE)。
    这里我们使用一个假设的计算方式。可以替换为具体的二级结构预测方法。
    
    参数:
    - sequence: RNA序列字符串

    返回:
    - MFE值
    """
    # 假设的简单MFE计算：GC含量越高，MFE越小
    gc_content = (sequence.count('G') + sequence.count('C')) / len(sequence)
    mfe = -gc_content  # 这里只是一个示例计算方法，实际情况需要更复杂的计算
    return mfe

def compute_cai(sequence):
    """
    计算RNA序列的密码子适应性指数(CAI)。
    这里假设我们有一个CAI计算的方法，实际情况应依赖于密码子使用数据库。

    参数:
    - sequence: RNA序列字符串

    返回:
    - CAI值
    """
    # 假设的CAI计算：此处可以根据密码子的出现频率和特定物种的数据库进行计算
    # 简化为密码子种类的比例，实际应用需要根据具体的蛋白质数据库进行计算
    unique_codons = set([sequence[i:i+3] for i in range(0, len(sequence), 3)])
    cai = len(unique_codons) / (len(sequence) // 3)  # 假设CAI与密码子种类数的关系
    return cai

def compute_score(sequence, lambda_value):
    """
    计算RNA序列的目标函数得分，目标函数是MFE和CAI的加权和。
    
    参数:
    - sequence: RNA序列字符串
    - lambda_value: MFE和CAI的权重系数
    
    返回:
    - 目标函数得分
    """
    mfe = compute_mfe(sequence)
    cai = compute_cai(sequence)
    return lambda_value * mfe + (1 - lambda_value) * cai

def load_initial_sequence(file_path):
    """
    从文件加载初始RNA序列。
    
    参数:
    - file_path: 文件路径
    
    返回:
    - 初始RNA序列字符串
    """
    with open(file_path, 'r') as file:
        # 假设序列是FASTA格式
        seq_record = SeqIO.read(file, "fasta")
        return str(seq_record.seq)

def save_results(sequence, file_path):
    """
    保存优化后的RNA序列到文件。
    
    参数:
    - sequence: RNA序列字符串
    - file_path: 输出文件路径
    """
    with open(file_path, 'w') as file:
        file.write(">Optimized RNA Sequence\n")
        file.write(sequence)

def encode_sequence_as_continuous(sequence):
    """
    将RNA序列编码为连续变量（例如基于密码子频率的表示）。
    
    参数:
    - sequence: RNA序列字符串
    
    返回:
    - 连续参数（这里以简单的例子表示）
    """
    # 假设每个密码子对应一个数字表示（实际应使用更复杂的编码方式）
    codons = [sequence[i:i+3] for i in range(0, len(sequence), 3)]
    codon_set = sorted(set(codons))
    codon_to_num = {codon: i / len(codon_set) for i, codon in enumerate(codon_set)}
    
    return np.array([codon_to_num[codon] for codon in codons])

def discretize_to_sequence(continuous_params):
    """
    将连续参数离散化为RNA序列（根据优化得到的参数生成密码子序列）。
    
    参数:
    - continuous_params: 连续优化后的参数
    
    返回:
    - 离散化的RNA序列字符串
    """
    # 假设我们用每个数字对应的密码子来重建序列
    codon_set = ['AUG', 'GCC', 'AUU', 'GUA', 'GCG', 'UAG']  # 示例密码子集
    num_to_codon = {i / len(codon_set): codon for i, codon in enumerate(codon_set)}
    
    codons = [num_to_codon[param] for param in continuous_params]
    return ''.join(codons)
