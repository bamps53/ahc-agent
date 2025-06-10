from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ahc_agent.core.problem_logic import ProblemLogic
from ahc_agent.core.scorer import create_score_calculator as create_actual_scorer_calculator


@pytest.fixture
def mock_llm_client_fixture():
    return MagicMock()


@pytest.fixture
def test_workspace_dir_fixture(tmp_path: Path) -> Path:
    # tmp_path is a pytest fixture providing a temporary directory unique to the test invocation
    d = tmp_path / "test_workspace"
    d.mkdir(exist_ok=True)
    return d


@pytest.fixture
def problem_logic_instance(mock_llm_client_fixture: MagicMock, test_workspace_dir_fixture: Path) -> ProblemLogic:
    return ProblemLogic(llm_client=mock_llm_client_fixture, workspace_dir=test_workspace_dir_fixture, config={})


class TestProblemLogic:
    @pytest.mark.asyncio
    async def test_create_score_calculator_returns_valid_function(self, problem_logic_instance: ProblemLogic):
        """
        ProblemLogic.create_score_calculatorが、
        scorer.pyベースのスコア計算関数を正しく返すことをテストします。
        """
        problem_info_mock = {
            "title": "Test Problem",
        }

        score_calculator_from_pl = await problem_logic_instance.create_score_calculator(problem_info_mock)

        assert callable(score_calculator_from_pl), "create_score_calculatorは呼び出し可能なオブジェクトを返すべきです。"

        test_input = "dummy_input"
        test_output = "dummy_output"
        expected_score = 0.0
        try:
            actual_score = score_calculator_from_pl(test_input, test_output)
            assert actual_score == expected_score, "ProblemLogicから返されたスコア計算関数が期待されるスコアを返しませんでした。"
        except Exception as e:
            pytest.fail(f"ProblemLogicから返されたスコア計算関数の呼び出し中にエラーが発生しました: {e}")

        actual_scorer_func = create_actual_scorer_calculator()
        try:
            score_from_actual_scorer = actual_scorer_func(test_input, test_output)
            assert actual_score == score_from_actual_scorer, "ProblemLogic経由のスコアとscorer.py直接のスコアが一致しません。"
        except Exception as e:
            pytest.fail(f"scorer.pyから直接取得したスコア計算関数の呼び出し中にエラーが発生しました: {e}")
