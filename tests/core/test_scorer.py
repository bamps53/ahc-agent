import unittest

from ahc_agent.core.scorer import calculate_score, create_score_calculator


class TestScorer(unittest.TestCase):
    def test_calculate_score(self):
        """
        calculate_scoreが仮のスコア0.0を返すことをテストします。
        """
        input_data = "some_input"
        output_data = "some_output"
        expected_score = 0.0
        actual_score = calculate_score(input_data, output_data)
        self.assertEqual(actual_score, expected_score)

    def test_create_score_calculator(self):
        """
        create_score_calculatorが正しい関数を返し、その関数が動作することをテストします。
        """
        score_func = create_score_calculator()
        self.assertTrue(callable(score_func), "create_score_calculatorは呼び出し可能なオブジェクトを返すべきです。")

        # 返された関数が calculate_score と同じように動作するか(ここでは0.0を返すか)確認
        input_data = "test_input"
        output_data = "test_output"
        expected_score = 0.0
        try:
            actual_score = score_func(input_data, output_data)
            self.assertEqual(actual_score, expected_score, "返されたスコア計算関数が期待されるスコアを返しませんでした。")
        except Exception as e:
            self.fail(f"返されたスコア計算関数の呼び出し中にエラーが発生しました: {e}")


if __name__ == "__main__":
    unittest.main()
