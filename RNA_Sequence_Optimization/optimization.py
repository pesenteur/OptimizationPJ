import numpy as np
from sequence import generate_candidates, local_search
from scipy.optimize import minimize
from utils import compute_mfe, compute_cai,compute_score
from sequence import encode_sequence_as_continuous,discretize_to_sequence


def local_search_A(sequence, lambda_value=None, current_score=None, accept_if_improves=False):
    """
    局部扰动：随机选择一个密码子做同义替换。
    - 默认：只做扰动并返回（适合模拟退火/扰动算子）
    - 若 accept_if_improves=True：只有当新序列 score 更优才接受（适合贪心/爬山）

    参数:
    - sequence: RNA序列
    - lambda_value/current_score: 开启接受规则时必须提供
    - accept_if_improves: 是否启用“只接受更优”

    返回:
    - new_sequence 或原 sequence
    """
    from utils import compute_score  # 避免循环引用风险

    # 选择一个密码子位置扰动
    idx = random.randint(0, len(sequence)//3 - 1) * 3
    old_codon = sequence[idx:idx+3]
    new_codon = replace_with_synonymous_codon(old_codon)

    new_sequence = sequence[:idx] + new_codon + sequence[idx+3:]

    # 不启用接受规则：直接返回扰动结果（给 SA 用）
    if not accept_if_improves:
        return new_sequence

    # 启用接受规则：只接受更优
    if lambda_value is None or current_score is None:
        raise ValueError("accept_if_improves=True 时，需要提供 lambda_value 和 current_score")

    new_score = compute_score(new_sequence, lambda_value)
    return new_sequence if new_score < current_score else sequence


def optimize_method_a(initial_sequence, lambda_value, max_iterations):
    """
    随机 + 贪婪搜索（局部优化）优化RNA序列。

    参数:
    - initial_sequence: 初始RNA序列
    - lambda_value: MFE和CAI的权重系数
    - max_iterations: 最大迭代次数

    返回:
    - optimized_sequence: 最优RNA序列
    """
    
    # 1. 初始化：当前最优序列为初始序列
    current_best_sequence = initial_sequence
    current_best_score = compute_score(current_best_sequence, lambda_value)

    # 2. 开始迭代过程
    iteration = 0
    while iteration < max_iterations:
        iteration += 1
        
        # 生成候选序列（通过同义密码子替换）
        candidates = generate_candidates(current_best_sequence)

        # 计算候选序列的目标函数得分
        scores = [compute_score(candidate, lambda_value) for candidate in candidates]
        
        # 3. 贪婪选择最优序列：选择得分最低的序列
        best_candidate = candidates[np.argmin(scores)]
        best_candidate_score = min(scores)

        # 如果找到的候选序列比当前最优序列好，更新最优序列
        if best_candidate_score < current_best_score:
            current_best_sequence = best_candidate
            current_best_score = best_candidate_score

            # 4. 局部优化：对选中的最优序列进行局部优化
            # 4. 局部优化：只在更优时才接受
            current_best_sequence = local_search_A(
                current_best_sequence,
                lambda_value=lambda_value,
                current_score=current_best_score,
                accept_if_improves=True
            )
            current_best_score = compute_score(current_best_sequence, lambda_value)
        
        # 5. 停止准则：当目标函数收敛时停止
        # if iteration > 1 and abs(previous_best_score - current_best_score) < 1e-10:
        #     print(f"收敛，停止迭代（迭代次数：{iteration}）")
        #     break
        
        previous_best_score = current_best_score
    
    return current_best_sequence


def optimize_method_b(initial_sequence, lambda_value, max_iterations):
    """
    连续优化 + 牛顿法 / 约束牛顿法优化RNA序列。

    参数:
    - initial_sequence: 初始RNA序列
    - lambda_value: MFE和CAI的权重系数
    - max_iterations: 最大迭代次数

    返回:
    - optimized_sequence: 最优RNA序列
    """
    
    # 1. 将RNA序列转换为连续参数
    initial_params = encode_sequence_as_continuous(initial_sequence)
    
    # 2. 定义目标函数
    def objective_function(params):
        sequence = discretize_to_sequence(params, initial_sequence)
        return compute_score(sequence, lambda_value)

    
    # 3. 使用牛顿法 / 约束牛顿法进行优化
    result = minimize(objective_function, initial_params, method='trust-constr', options={'maxiter': max_iterations})
    
    # 4. 获取优化后的连续参数
    optimized_params = result.x
    
    # 5. 将连续参数离散化为RNA序列
    optimized_sequence = discretize_to_sequence(optimized_params,initial_sequence)
    
    # 6. 计算优化后的RNA序列的目标函数值
    final_score = compute_score(optimized_sequence, lambda_value)
    
    return optimized_sequence


from sequence import mutation, codon_swap, local_search_with_mutation

import random
import math


def optimize_method_c(initial_sequence, lambda_value, max_iterations, initial_temperature, cooling_rate):
    """
    使用改进版模拟退火算法优化RNA序列。
    
    参数:
    - initial_sequence: 初始RNA序列
    - lambda_value: MFE和CAI的权重系数
    - max_iterations: 最大迭代次数
    - initial_temperature: 初始温度
    - cooling_rate: 温度衰减率

    返回:
    - optimized_sequence: 最优RNA序列
    """
    
    # 1. 初始化
    current_sequence = initial_sequence
    current_score = compute_score(current_sequence, lambda_value)
    temperature = initial_temperature
    
    print(f"初始序列得分: {current_score}")
    
    # 2. 进行模拟退火优化
    for iteration in range(max_iterations):
        # 生成一个新的候选序列，通过多种方式扰动
        new_sequence = perturb_sequence(current_sequence)
        
        # 计算新序列的目标函数值
        new_score = compute_score(new_sequence, lambda_value)
        
        # 计算能量差值（目标函数值的变化）
        delta_energy = new_score - current_score
        
        # 如果新序列更优，直接接受
        if delta_energy < 0:
            current_sequence = new_sequence
            current_score = new_score
        else:
            # 如果新序列更差，以一定概率接受
            probability = math.exp(-delta_energy / temperature)
            if random.random() < probability:
                current_sequence = new_sequence
                current_score = new_score
        
        # 降温
        temperature *= cooling_rate
        
        # 输出每个迭代步骤的状态
        if iteration % 10 == 0:  # 每10步打印一次状态
            print(f"迭代 {iteration}, 当前得分: {current_score}, 当前温度: {temperature}")
    
    print(f"最终优化序列得分: {current_score}")
    return current_sequence

def perturb_sequence(sequence):
    """
    对RNA序列进行扰动，增加多样化扰动方式。
    
    参数:
    - sequence: 初始RNA序列
    
    返回:
    - new_sequence: 扰动后的新RNA序列
    """
    # 随机选择一种扰动方式
    perturbation_type = random.choice(['local_search', 'mutation', 'codon_swap'])
    
    if perturbation_type == 'local_search':
        return local_search(sequence)
    
    elif perturbation_type == 'mutation':
        return mutation(sequence)
    
    elif perturbation_type == 'codon_swap':
        return codon_swap(sequence)