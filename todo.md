# TODO

- [x] 1.  ***`todo.md` の作成***: 実施計画を `todo.md`として書き出します。
- [x] 2.  ***スコア計算スクリプトの作成 (`ahc_agent/core/scorer.py`)***:
    -   [x] 問題の入力と解答プログラムの出力を受け取り、スコアを計算して返す機能を実装します。
    -   [x] 初期実装では、基本的なスコア計算の枠組みを用意します。
- [x] 3.  ***`ProblemLogic` の修正 (`ahc_agent/core/problem_logic.py`)***:
    -   [x] `create_score_calculator` メソッドを修正し、新しく作成した `scorer.py` を使用してスコア計算関数を返すように変更します。
- [x] 4.  ***`SolveService` の確認 (`ahc_agent/services/solve_service.py`)***:
    -   [x] `_evaluate_solution_wrapper` メソッド内でスコア計算関数が正しく呼び出されることを確認します。
- [x] 5.  ***テストの追加***:
    -   [x] `scorer.py` の単体テスト (`tests/core/test_scorer.py`) を作成します。
    -   [x] `ProblemLogic` が正しくスコア計算関数を生成することをテストします (`tests/core/test_problem_logic.py` に追記または修正)。
- [x] 6.  ***テストの実行と確認***: すべてのテストを実行し、成功することを確認します。
- [x] 7.  ***Lint/Format の実行***: `ruff format` と `ruff check --fix` を実行します。
- [x] 8.  ***ドキュメントの更新***: 必要に応じて `README.md` や `docs/*.md` を更新します。 (今回は変更なし)
- [ ] 9.  ***変更差分の確認***: `git diff` を使用して変更内容を確認します。
- [ ] 10. ***変更の送信***: 全て問題がなければ、変更をコミットして送信します。
