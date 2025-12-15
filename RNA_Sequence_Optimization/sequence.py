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
def generate_candidates(sequence, num_candidates=10, max_mutations=3, protect_ends=True):
    """
    生成多个候选RNA序列：每个候选只随机替换少量密码子（同义替换，保证氨基酸不变）。

    参数:
    - sequence: 初始RNA序列
    - num_candidates: 候选数量
    - max_mutations: 每条候选最多替换多少个密码子（建议 1~3）
    - protect_ends: 是否保护起始密码子和末尾终止密码子（若存在）

    返回:
    - candidates: 候选序列列表
    """
    codons = [sequence[i:i+3] for i in range(0, len(sequence), 3)]
    n = len(codons)

    # 可变位置：同义集合大小>1，且（可选）不动起始/终止
    mutable_positions = []
    for i, c in enumerate(codons):
        aa = get_amino_acid(c)
        syn_list = SYNONYMOUS_CODONS.get(aa, [c])

        if len(syn_list) <= 1:
            continue

        if protect_ends:
            if i == 0:  # 起始位点通常是 AUG（本身也只有一个）
                continue
            if i == n - 1 and aa == '*':  # 末尾终止密码子不动
                continue

        mutable_positions.append(i)

    # 如果没有可变位置，返回原序列的拷贝
    if not mutable_positions:
        return [sequence for _ in range(num_candidates)]

    candidates = []
    for _ in range(num_candidates):
        new_codons = codons.copy()

        k = random.randint(1, min(max_mutations, len(mutable_positions)))
        positions = random.sample(mutable_positions, k)

        for pos in positions:
            old = new_codons[pos]
            aa = get_amino_acid(old)
            syn_list = SYNONYMOUS_CODONS[aa]

            # 尽量选一个不同于原来的同义密码子
            if len(syn_list) > 1:
                choices = [x for x in syn_list if x != old]
                new_codons[pos] = random.choice(choices) if choices else old

        candidates.append(''.join(new_codons))

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
    将RNA序列转换为连续参数：
    对每个密码子位点 i，只在该位点对应氨基酸的同义密码子集合内编码，
    得到 params[i] ∈ [0, 1)。
    """
    codons = [sequence[i:i+3] for i in range(0, len(sequence), 3)]
    params = []

    for codon in codons:
        aa = get_amino_acid(codon)
        syn_list = SYNONYMOUS_CODONS.get(aa, [codon])

        # 终止/未知：固定不动
        if aa == '*' or len(syn_list) == 1:
            params.append(0.0)
            continue

        # 在该位点同义集合中的“索引”
        try:
            j = syn_list.index(codon)
        except ValueError:
            j = 0

        # 映射到 [0,1)：用“区间中心”更稳定
        m = len(syn_list)
        params.append((j + 0.5) / m)

    return np.array(params, dtype=float)

def discretize_to_sequence(continuous_params, original_sequence):
    """
    将连续参数离散化为RNA序列（确定性映射）：
    第 i 个参数只会在 original_sequence 第 i 个密码子对应的同义集合内选取，
    从而保证翻译出的氨基酸序列不变。
    """
    original_codons = [original_sequence[i:i+3] for i in range(0, len(original_sequence), 3)]
    params = np.asarray(continuous_params, dtype=float)

    if len(params) != len(original_codons):
        raise ValueError(f"Length mismatch: params({len(params)}) vs codons({len(original_codons)})")

    new_codons = []
    for i, (p, orig_codon) in enumerate(zip(params, original_codons)):
        aa = get_amino_acid(orig_codon)
        syn_list = SYNONYMOUS_CODONS.get(aa, [orig_codon])

        # 终止/未知：保持原样（更符合“蛋白不变”）
        if aa == '*' or len(syn_list) == 1:
            new_codons.append(orig_codon)
            continue

        m = len(syn_list)
        # 把 p 映射到 [0,1)，再映射到 {0,...,m-1}
        # 关键：确定性（不再 random.choice）
        p = float(p)
        p = p - np.floor(p)          # 支持优化器跑出范围：wrap 到 [0,1)
        idx = int(np.floor(p * m))   # 0..m-1
        idx = max(0, min(m - 1, idx))
        new_codons.append(syn_list[idx])

    return ''.join(new_codons)


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
    随机交换两个位点的密码子，但仅允许：
    - 两个位点对应的氨基酸相同（交换后氨基酸序列保持不变）
    - 默认不动起始密码子(第0位)和末尾终止密码子(最后一位，如果是stop)
    """
    codons = [sequence[i:i+3] for i in range(0, len(sequence), 3)]
    n = len(codons)
    if n < 3:
        return sequence

    aa_list = [get_amino_acid(c) for c in codons]

    # 候选位点：不动起始位；如果最后是 stop，也不动最后一位
    start = 1
    end = n - 1
    if aa_list[-1] == '*':
        end = n - 2
    if end <= start:
        return sequence

    # 找到“氨基酸相同”的可交换对
    candidates = []
    for i in range(start, end + 1):
        if aa_list[i] == '*':
            continue
        for j in range(i + 1, end + 1):
            if aa_list[i] == aa_list[j]:
                candidates.append((i, j))

    if not candidates:
        # 没有可交换对就退化成一次同义替换扰动
        return local_search(sequence)

    i, j = random.choice(candidates)
    codons[i], codons[j] = codons[j], codons[i]
    return ''.join(codons)
