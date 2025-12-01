import unittest
from sequence import replace_with_synonymous_codon, generate_candidates, local_search, continuous_optimization, local_search_with_mutation
from utils import compute_mfe, compute_cai, compute_score, load_initial_sequence, save_results

class TestRNAOptimization(unittest.TestCase):

    def test_replace_with_synonymous_codon(self):
        """
        测试同义密码子替换功能
        """
        codon = "GCU"  # 代表氨基酸A
        new_codon = replace_with_synonymous_codon(codon)
        self.assertIn(new_codon, ['GCC', 'GCA', 'GCG'], "同义密码子替换失败")
    
    def test_generate_candidates(self):
        """
        测试生成候选序列的功能
        """
        sequence = "AUGGCCAUUGUACUGGCCGUAUUAG"
        candidates = generate_candidates(sequence, num_candidates=5)
        self.assertEqual(len(candidates), 5, "候选序列生成失败")
        self.assertTrue(all(len(candidate) == len(sequence) for candidate in candidates), "候选序列长度不一致")
    
    def test_local_search(self):
        """
        测试局部优化功能
        """
        sequence = "AUGGCCAUUGUACUGGCCGUAUUAG"
        optimized_sequence = local_search(sequence)
        self.assertNotEqual(sequence, optimized_sequence, "局部优化未改变序列")
    
    def test_continuous_optimization(self):
        """
        测试连续优化功能
        """
        sequence = "AUGGCCAUUGUACUGGCCGUAUUAG"
        optimized_params = continuous_optimization(sequence, lambda_value=0.5)
        self.assertEqual(len(optimized_params), len(sequence) // 3, "连续优化结果不符合预期")
    
    def test_local_search_with_mutation(self):
        """
        测试局部搜索与突变功能
        """
        sequence = "AUGGCCAUUGUACUGGCCGUAUUAG"
        optimized_sequence = local_search_with_mutation(sequence)
        self.assertNotEqual(sequence, optimized_sequence, "局部搜索与突变未改变序列")
    
    def test_compute_mfe(self):
        """
        测试MFE计算
        """
        sequence = "AUGGCCAUUGUACUGGCCGUAUUAG"
        mfe = compute_mfe(sequence)
        self.assertIsInstance(mfe, float, "MFE计算结果类型错误")
    
    def test_compute_cai(self):
        """
        测试CAI计算
        """
        sequence = "AUGGCCAUUGUACUGGCCGUAUUAG"
        cai = compute_cai(sequence)
        self.assertIsInstance(cai, float, "CAI计算结果类型错误")
    
    def test_compute_score(self):
        """
        测试目标函数得分计算
        """
        sequence = "AUGGCCAUUGUACUGGCCGUAUUAG"
        score = compute_score(sequence, lambda_value=0.5)
        self.assertIsInstance(score, float, "目标函数得分类型错误")
    
    def test_load_initial_sequence(self):
        """
        测试加载初始RNA序列
        """
        # 假设有一个名为test_sequence.fasta的测试文件
        sequence = load_initial_sequence('test_sequence.fasta')
        self.assertIsInstance(sequence, str, "加载初始RNA序列失败")
    
    def test_save_results(self):
        """
        测试保存优化结果
        """
        sequence = "AUGGCCAUUGUACUGGCCGUAUUAG"
        save_results(sequence, 'optimized_sequence.txt')
        with open('optimized_sequence.txt', 'r') as file:
            saved_sequence = file.read().strip()
        self.assertEqual(sequence, saved_sequence, "保存的RNA序列与原序列不一致")

if __name__ == "__main__":
    unittest.main()
