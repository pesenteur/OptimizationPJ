import numpy as np
from sequence import generate_candidates, local_search, compute_score
from scipy.optimize import minimize
from utils import compute_mfe, compute_cai
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
            current_best_sequence = local_search(current_best_sequence)
            current_best_score = compute_score(current_best_sequence, lambda_value)
        
        # 5. 停止准则：当目标函数收敛时停止
        if iteration > 1 and abs(previous_best_score - current_best_score) < 1e-6:
            print(f"收敛，停止迭代（迭代次数：{iteration}）")
            break
        
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
        # 计算MFE
        mfe = compute_mfe(params)
        # 计算CAI
        cai = compute_cai(params)
        # 目标函数: λ * MFE + (1 - λ) * CAI
        return lambda_value * mfe + (1 - lambda_value) * cai
    
    # 3. 使用牛顿法 / 约束牛顿法进行优化
    result = minimize(objective_function, initial_params, method='trust-constr', options={'maxiter': max_iterations})
    
    # 4. 获取优化后的连续参数
    optimized_params = result.x
    
    # 5. 将连续参数离散化为RNA序列
    optimized_sequence = discretize_to_sequence(optimized_params)
    
    # 6. 计算优化后的RNA序列的目标函数值
    final_score = compute_score(optimized_sequence, lambda_value)
    print(f"优化后的序列得分: {final_score}")
    
    return optimized_sequence


from sequence import generate_candidates_with_heuristics, continuous_optimization, local_search_with_mutation, compute_score


def optimize_method_c(initial_sequence, lambda_value, max_iterations):
    """
    混合启发式优化 + 连续优化 + 局部搜索优化RNA序列。

    参数:
    - initial_sequence: 初始RNA序列
    - lambda_value: MFE和CAI的权重系数
    - max_iterations: 最大迭代次数

    返回:
    - optimized_sequence: 最优RNA序列
    """
    
    # 1. 生成候选序列（基于启发式方法，如蛋白特定密码子偏好）
    candidates = generate_candidates_with_heuristics(initial_sequence)
    
    # 2. 计算候选序列的目标函数得分
    scores = [compute_score(candidate, lambda_value) for candidate in candidates]
    
    # 3. 选择最优候选序列
    best_candidate = candidates[np.argmin(scores)]
    best_candidate_score = min(scores)
    
    print(f"初步优化候选序列得分: {best_candidate_score}")
    
    # 4. 连续优化：对最优候选序列进行连续优化
    best_params = continuous_optimization(best_candidate, lambda_value)
    
    # 5. 局部搜索和突变：对优化结果进行局部搜索和突变
    optimized_sequence = local_search_with_mutation(best_candidate)
    
    # 6. 计算优化后的RNA序列的目标函数值
    final_score = compute_score(optimized_sequence, lambda_value)
    print(f"优化后的序列得分: {final_score}")
    
    return optimized_sequence
