# TODO

## 概要

`solve` コマンド実行時の評価パイプラインを確立し、C++ソリューションの自動評価（コンパイル、実行、スコアリング）とエラー報告の堅牢化を行う。

## 完了済みの主な作業

-   **ベースラインC++ソリューション生成ロジックの明確化:**
    -   `ProblemLogic._get_basic_template` が提供するテンプレートがベースラインとして適切であることを確認。
-   **コンパイル＆実行スクリプトの抽象化:**
    -   `ProblemLogic.execute_cpp_code` を新規追加。C++コードのコンパイルと実行を詳細な結果と共に返す。
-   **評価スクリプトの堅牢化と明確化:**
    -   `ProblemLogic.evaluate_solution_code` を新規追加。C++コード、テストケース群、スコア計算関数を使い、全体スコアとテストケース毎の詳細評価（エラー情報含む）を返す。
-   **評価パイプラインの整備:**
    -   `SolveService._evaluate_solution_wrapper` が `ProblemLogic.evaluate_solution_code` を呼び出すように変更。
    -   `EvolutionaryEngine._evaluate_population` が新しい評価詳細構造を利用するように変更。
    -   `EvolutionaryEngine._mutate` が新しい評価詳細構造からLLM用のエラーメッセージを生成するように改善。
-   **エラー報告の強化:**
    -   `EvolutionaryEngine._mutate` でLLMに渡すエラー情報が、評価パイプラインからの詳細なコンパイルエラー、実行エラー、スコア計算エラーを含むように強化されたことを確認。
-   **ユニットテストの追加:**
    -   `ProblemLogic.execute_cpp_code` および `ProblemLogic.evaluate_solution_code` に対するユニットテストを `tests/core/test_problem_logic.py` に追加。

## 今後の予定

1.  **`todo.md` の作成** (完了)
2.  **ドキュメント更新:**
    -   README.md や docs/\*.md に、新しい評価フローや関連するモジュールの変更点を反映させる。
3.  **Lint/Format の実行:**
    -   プロジェクト全体のコードに対して `uv run --frozen ruff format .` および `uv run --frozen ruff check . --fix` を実行し、コードスタイルを統一する。
4.  **最終確認と提出:**
    -   全ての変更差分を確認し、意図しない変更や冗長なコードがないか最終チェックを行う。
    -   問題がなければ、変更をコミットして提出する。
