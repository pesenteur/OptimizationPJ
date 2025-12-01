import random
import numpy as np
from utils import compute_cai,compute_mfe

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
    替换RNA中的密码子为同义密码子（确保替换后的密码子与原密码子编码相同的氨基酸）。
    
    参数:
    - codon: 要替换的RNA密码子
    
    返回:
    - 替换后的同义密码子
    """
    amino_acid = get_amino_acid(codon)  # 获取原密码子的氨基酸
    if amino_acid == '*':
        return codon  # 如果是终止密码子，则返回原密码子
    return random.choice(SYNONYMOUS_CODONS[amino_acid])  # 从同义密码子中随机选择一个

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
import numpy as np
from scipy.optimize import minimize

def continuous_optimization(sequence, lambda_value):
    """
    对RNA序列进行连续优化，基于目标函数和约束条件进行优化。
    
    参数：
    - sequence: RNA序列
    - lambda_value: MFE与CAI的权重系数
    
    返回：
    - optimized_params: 优化后的序列参数
    """
    
    # 将RNA序列转换为连续参数表示
    initial_params = encode_sequence_as_continuous(sequence)
    
    # 定义目标函数
    def objective_function(params):
        # 将连续参数转换为RNA序列
        rna_sequence = discretize_to_sequence(params, sequence)
        
        # 计算MFE和CAI
        mfe = compute_mfe(rna_sequence)
        cai = compute_cai(rna_sequence)
        
        # 目标函数: λ * MFE + (1 - λ) * CAI
        return lambda_value * mfe + (1 - lambda_value) * cai

    # 约束条件：可以定义多个约束，示例中限制GC含量在0.4到0.6之间
    def constraint_gc_content(params):
        # 将连续参数转换为RNA序列
        rna_sequence = discretize_to_sequence(params, sequence)
        
        # 计算GC含量
        gc_content = (rna_sequence.count('G') + rna_sequence.count('C')) / len(rna_sequence)
        
        # 约束：GC含量应该在0.4到0.6之间
        return gc_content - 0.5  # 例如，要求GC含量尽量接近0.5
    
    # 设置约束条件，要求GC含量在0.4到0.6之间
    constraints = [{'type': 'ineq', 'fun': constraint_gc_content}]
    
    # 使用约束优化方法（例如使用信赖域牛顿法 'trust-constr'）
    result = minimize(objective_function, initial_params, method='trust-constr', constraints=constraints, options={'maxiter': 100})
    
    # 获取优化后的参数
    optimized_params = result.x
    
    # 返回优化后的序列参数
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

def encode_sequence_as_continuous(sequence):
    """
    将RNA序列转换为连续参数。每个密码子通过映射到[0, 1]区间的数字表示。
    
    参数:
    - sequence: RNA序列字符串
    
    返回:
    - 连续参数表示的数组
    """
    # 将RNA序列按每3个碱基分割为密码子
    codons = [sequence[i:i+3] for i in range(0, len(sequence), 3)]
    
    # 获取所有独特的密码子
    codon_set = sorted(set(codons))
    
    # 创建一个字典，将密码子映射到[0, 1]区间的数字
    codon_to_num = {codon: i / len(codon_set) for i, codon in enumerate(codon_set)}
    
    # 将RNA序列中的每个密码子转换为连续数值
    continuous_params = np.array([codon_to_num[codon] for codon in codons])
    
    return continuous_params

def discretize_to_sequence(continuous_params, original_sequence):
    """
    将连续参数离散化为RNA序列，确保映射回的密码子与原密码子同义。
    
    参数:
    - continuous_params: 优化后的连续参数数组
    - original_sequence: 原始RNA序列（用于确保同义密码子一致）
    
    返回:
    - 离散化的RNA序列字符串
    """
    # 将原始序列分割为密码子
    original_codons = [original_sequence[i:i+3] for i in range(0, len(original_sequence), 3)]
    
    # 获取密码子集
    codon_set = sorted(SYNONYMOUS_CODONS.keys())  # 获取氨基酸字母代码
    
    # 创建一个字典将每个氨基酸映射到它的同义密码子
    amino_acid_to_codons = {amino_acid: SYNONYMOUS_CODONS[amino_acid] for amino_acid in codon_set}
    
    # 将连续参数映射为最接近的密码子
    codons = []
    for i, param in enumerate(continuous_params):
        # 获取原始密码子的氨基酸
        original_codon = original_codons[i]
        amino_acid = get_amino_acid(original_codon)  # 获取该密码子对应的氨基酸
        
        # 从同义密码子中选择一个
        possible_codons = amino_acid_to_codons[amino_acid]
        codons.append(random.choice(possible_codons))  # 随机选择一个同义密码子
    
    # 拼接密码子为RNA序列
    return ''.join(codons)

def mutation(sequence):
    """
    随机突变RNA序列中的一个密码子（仅同义替换）。
    
    参数:
    - sequence: RNA序列
    
    返回:
    - new_sequence: 进行突变后的新RNA序列
    """
    # 随机选择一个密码子位置进行替换
    idx = random.randint(0, len(sequence) // 3 - 1) * 3
    codon = sequence[idx:idx+3]
    new_codon = replace_with_synonymous_codon(codon)
    
    new_sequence = sequence[:idx] + new_codon + sequence[idx+3:]
    return new_sequence

def codon_swap(sequence):
    """
    随机交换RNA序列中的两个密码子的位置（仅交换同义密码子）。
    
    参数:
    - sequence: RNA序列
    
    返回:
    - new_sequence: 交换密码子位置后的新RNA序列
    """
    codons = [sequence[i:i+3] for i in range(0, len(sequence), 3)]
    
    # 随机选择两个密码子交换
    idx1, idx2 = random.sample(range(len(codons)), 2)
    
    # 确保交换的密码子是同义密码子
    codons[idx1], codons[idx2] = codons[idx2], codons[idx1]
    
    # 生成新的序列
    new_sequence = ''.join(codons)
    return new_sequence