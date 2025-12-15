import numpy as np
from Bio import SeqIO
from scipy.optimize import minimize

human_codon_usage = {
    'AAA': 0.054, 'AAC': 0.137, 'AAG': 0.067, 'AAU': 0.091,
    'ACA': 0.140, 'ACC': 0.133, 'ACG': 0.085, 'ACU': 0.094,
    'AGA': 0.058, 'AGC': 0.122, 'AGG': 0.059, 'AGU': 0.072,
    'AUA': 0.030, 'AUC': 0.105, 'AUG': 0.160, 'AUU': 0.155,
    'CAA': 0.075, 'CAC': 0.114, 'CAG': 0.073, 'CAU': 0.084,
    'CCA': 0.141, 'CCC': 0.125, 'CCG': 0.097, 'CCU': 0.107,
    'CGA': 0.025, 'CGC': 0.060, 'CGG': 0.025, 'CGU': 0.034,
    'CUA': 0.064, 'CUC': 0.120, 'CUG': 0.091, 'CUU': 0.089,
    'GAA': 0.084, 'GAC': 0.144, 'GAG': 0.084, 'GAU': 0.106,
    'GCA': 0.130, 'GCC': 0.151, 'GCG': 0.058, 'GCU': 0.078,
    'GGA': 0.057, 'GGC': 0.092, 'GGG': 0.062, 'GGU': 0.082,
    'GUA': 0.075, 'GUC': 0.126, 'GUG': 0.085, 'GUU': 0.090,
    'UAA': 0.008, 'UAC': 0.124, 'UAG': 0.005, 'UAU': 0.102,
    'UCA': 0.103, 'UCC': 0.137, 'UCG': 0.044, 'UCU': 0.089,
    'UGA': 0.003, 'UGC': 0.085, 'UGG': 0.146, 'UGU': 0.115,
    'UUA': 0.038, 'UUC': 0.097, 'UUG': 0.042, 'UUU': 0.126
}


def compute_cai(sequence):
    """
    计算给定RNA序列的CAI，使用人类的密码子使用频率表。
    """
    codons = [sequence[i:i+3] for i in range(0, len(sequence), 3)]
    cai_values = []

    for codon in codons:
        if codon in human_codon_usage:
            max_usage = max(human_codon_usage.values())
            cai_values.append(human_codon_usage[codon] / max_usage)
    
    # 计算 CAI: 几何平均
    if len(cai_values) > 0:
        cai = np.prod(cai_values) ** (1 / len(cai_values))
        return cai
    else:
        return 0.0



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
    mfe = gc_content  # 这里只是一个示例计算方法，实际情况需要更复杂的计算
    return mfe

# def compute_cai(sequence):
#     """
#     计算RNA序列的密码子适应性指数(CAI)。
#     这里假设我们有一个CAI计算的方法，实际情况应依赖于密码子使用数据库。

#     参数:
#     - sequence: RNA序列字符串

#     返回:
#     - CAI值
#     """
#     # 假设的CAI计算：此处可以根据密码子的出现频率和特定物种的数据库进行计算
#     # 简化为密码子种类的比例，实际应用需要根据具体的蛋白质数据库进行计算
#     unique_codons = set([sequence[i:i+3] for i in range(0, len(sequence), 3)])
#     cai = len(unique_codons) / (len(sequence) // 3)  # 假设CAI与密码子种类数的关系
#     return cai

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
    return lambda_value * mfe - (1 - lambda_value) * cai

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
