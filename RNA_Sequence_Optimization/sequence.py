import random
import numpy as np

# 定义常见的同义密码子映射
SYNONYMOUS_CODONS = {
    'A': ['GCU', 'GCC', 'GCA', 'GCG'],
    'C': ['UGU', 'UGC'],
    'D': ['GAU', 'GAC'],
    'E': ['GAA', 'GAG'],
    'F': ['UUU', 'UUC'],
    'G': ['GGU', 'GGC', 'GGA', 'GGG'],
    'H': ['CAU', 'CAC'],
    'I': ['AUU', 'AUC', 'AUA'],
    'K': ['AAA', 'AAG'],
    'L': ['UUA', 'UUG', 'CUU', 'CUC', 'CUA', 'CUG'],
    'M': ['AUG'],
    'N': ['AAU', 'AAC'],
    'P': ['CCU', 'CCC', 'CCA', 'CCG'],
    'Q': ['CAA', 'CAG'],
    'R': ['CGU', 'CGC', 'CGA', 'CGG', 'AGA', 'AGG'],
    'S': ['UCU', 'UCC', 'UCA', 'UCG', 'AGU', 'AGC'],
    'T': ['ACU', 'ACC', 'ACA', 'ACG'],
    'V': ['GUU', 'GUC', 'GUA', 'GUG'],
    'W': ['UGG'],
    'Y': ['UAU', 'UAC'],
    '*': ['UAA', 'UAG', 'UGA']
}

# 1. 同义密码子替换：从同义密码子集合中随机选择一个密码子
def replace_with_synonymous_codon(codon):
    """
    替换RNA中的密码子为同义密码子。
    
    参数:
    - codon: 要替换的RNA密码子
    
    返回:
    - 替换后的同义密码子
    """
    amino_acid = get_amino_acid(codon)
    return random.choice(SYNONYMOUS_CODONS[amino_acid])

# 2. 获取密码子对应的氨基酸
def get_amino_acid(codon):
    """
    获取RNA密码子对应的氨基酸。
    
    参数:
    - codon: 3个碱基组成的RNA密码子
    
    返回:
    - 氨基酸的1个字母代码
    """
    codon_table = {
        'GCU': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
        'UGU': 'C', 'UGC': 'C',
        'GAU': 'D', 'GAC': 'D',
        'GAA': 'E', 'GAG': 'E',
        'UUU': 'F', 'UUC': 'F',
        'GGU': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G',
        'CAU': 'H', 'CAC': 'H',
        'AUU': 'I', 'AUC': 'I', 'AUA': 'I',
        'AAA': 'K', 'AAG': 'K',
        'UUA': 'L', 'UUG': 'L', 'CUU': 'L', 'CUC': 'L', 'CUA': 'L', 'CUG': 'L',
        'AUG': 'M',
        'AAU': 'N', 'AAC': 'N',
        'CCU': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
        'CAA': 'Q', 'CAG': 'Q',
        'CGU': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R', 'AGA': 'R', 'AGG': 'R',
        'UCU': 'S', 'UCC': 'S', 'UCA': 'S', 'UCG': 'S', 'AGU': 'S', 'AGC': 'S',
        'ACU': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
        'GUU': 'V', 'GUC': 'V', 'GUA': 'V', 'GUG': 'V',
        'UGG': 'W',
        'UAU': 'Y', 'UAC': 'Y',
        'UAA': '*', 'UAG': '*', 'UGA': '*'
    }
    return codon_table.get(codon, '*')  # 返回氨基酸的1个字母代码

# 3. 同义密码子替换生成候选序列
def generate_candidates(sequence, num_candidates=10):
    """
    生成多个候选RNA序列，通过同义密码子替换生成新序列。
    
    参数:
    - sequence: 初始RNA序列
    - num_candidates: 生成候选序列的数量
    
    返回:
    - candidates: 生成的候选序列列表
    """
    candidates = []
    for _ in range(num_candidates):
        candidate = ''.join([replace_with_synonymous_codon(sequence[i:i+3]) if i % 3 == 0 else sequence[i:i+3] for i in range(0, len(sequence), 3)])
        candidates.append(candidate)
    return candidates

# 4. 局部搜索：通过小范围变动优化RNA序列
def local_search(sequence):
    """
    通过局部搜索优化RNA序列，替换序列中的一小部分密码子。
    
    参数:
    - sequence: 初始RNA序列
    
    返回:
    - optimized_sequence: 优化后的RNA序列
    """
    # 随机选择一个密码子进行替换
    random_index = random.randint(0, len(sequence)//3 - 1) * 3
    codon = sequence[random_index:random_index+3]
    new_codon = replace_with_synonymous_codon(codon)
    
    # 替换密码子并返回新的序列
    optimized_sequence = sequence[:random_index] + new_codon + sequence[random_index+3:]
    return optimized_sequence

# 5. 连续优化：优化RNA序列的连续表示
def continuous_optimization(sequence, lambda_value):
    """
    对RNA序列进行连续优化，这里使用一个简化的优化方法。
    参数：
    - sequence: RNA序列
    - lambda_value: MFE与CAI的权重系数
    返回：
    - 优化后的序列参数
    """
    # 简化的优化方式，这里我们使用简单的GA或模拟退火来进行优化
    # 实际上，应该基于目标函数和约束条件进行优化
    optimized_params = np.random.rand(len(sequence))  # 示例：使用随机生成的连续值表示优化结果
    return optimized_params

# 6. 局部搜索与突变：结合局部搜索和突变进一步优化
def local_search_with_mutation(sequence):
    """
    在局部搜索的基础上，进行突变操作，以进一步优化RNA序列。
    
    参数:
    - sequence: 初始RNA序列
    
    返回:
    - optimized_sequence: 优化后的RNA序列
    """
    # 局部搜索：替换一个密码子
    sequence = local_search(sequence)
    
    # 突变：随机改变一个密码子
    random_index = random.randint(0, len(sequence)//3 - 1) * 3
    codon = sequence[random_index:random_index+3]
    new_codon = replace_with_synonymous_codon(codon)
    
    optimized_sequence = sequence[:random_index] + new_codon + sequence[random_index+3:]
    return optimized_sequence

def generate_candidates_with_heuristics(sequence, num_candidates=10, preferred_codons=None):
    """
    生成多个候选RNA序列，通过同义密码子替换生成新序列，同时考虑目标蛋白的密码子偏好。
    
    参数:
    - sequence: 初始RNA序列
    - num_candidates: 生成候选序列的数量
    - preferred_codons: 目标蛋白的密码子偏好字典（可选）
    
    返回:
    - candidates: 生成的候选序列列表
    """
    candidates = []
    
    # 如果没有提供密码子偏好，则默认使用随机同义密码子替换
    if preferred_codons is None:
        preferred_codons = SYNONYMOUS_CODONS
    
    for _ in range(num_candidates):
        candidate = []
        
        # 遍历RNA序列，进行同义密码子替换
        for i in range(0, len(sequence), 3):
            codon = sequence[i:i+3]
            amino_acid = get_amino_acid(codon)
            
            # 选择偏好密码子，优先选择目标蛋白的密码子
            if amino_acid in preferred_codons:
                candidate.append(random.choice(preferred_codons[amino_acid]))
            else:
                candidate.append(replace_with_synonymous_codon(codon))
        
        # 将候选序列添加到列表中
        candidates.append(''.join(candidate))
    
    return candidates