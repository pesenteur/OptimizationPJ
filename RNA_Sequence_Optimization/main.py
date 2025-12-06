import argparse
from optimization import optimize_method_a, optimize_method_b, optimize_method_c
from utils import load_initial_sequence, save_results
from sequence import generate_candidates
from utils import compute_score

def main():
    # 1. 初始化参数
    parser = argparse.ArgumentParser(description="RNA Sequence Optimization")
    parser.add_argument('--method', type=str, choices=['A', 'B', 'C'], required=True, 
                        help="选择优化方法: A (随机 + 贪婪搜索), B (连续优化 + 牛顿法), C (混合启发式优化)")
    parser.add_argument('--sequence', type=str, required=True, 
                        help="初始RNA序列")
    parser.add_argument('--lambda_value', type=float, default=0.5, 
                        help="MFE与CAI的权重系数（默认：0.5）")
    parser.add_argument('--iterations', type=int, default=100, 
                        help="最大迭代次数（默认：100）")
    parser.add_argument('--output', type=str, default="optimized_sequence.txt", 
                        help="输出优化后的序列文件名（默认：optimized_sequence.txt）")
    args = parser.parse_args()

    # 2. 加载初始RNA序列
    initial_sequence = load_initial_sequence(args.sequence)

    # 3. 根据选择的优化方法进行RNA序列优化
    if args.method == 'A':
        print("开始方法A: 随机 + 贪婪搜索（局部优化）")
        optimized_sequence = optimize_method_a(initial_sequence, args.lambda_value, args.iterations)
    
    elif args.method == 'B':
        print("开始方法B: 连续优化 + 牛顿法")
        optimized_sequence = optimize_method_b(initial_sequence, args.lambda_value, args.iterations)
    
    elif args.method == 'C':
        print("开始方法C: 模拟退火")
        # 定义参数
        lambda_value = 0.5  # MFE和CAI的权重系数
        max_iterations = 10000  # 最大迭代次数
        initial_temperature = 1000  # 初始温度
        cooling_rate = 0.99  # 温度衰减率

        # 调用模拟退火优化方法
        optimized_sequence = optimize_method_c(initial_sequence, lambda_value, max_iterations, initial_temperature, cooling_rate)


    # 4. 计算优化后的序列的目标函数得分
    final_score = compute_score(optimized_sequence,args.lambda_value)
    print(f"优化后的序列得分: {final_score}")

    # 5. 保存优化后的序列
    save_results(optimized_sequence, args.output)

if __name__ == "__main__":
    main()
