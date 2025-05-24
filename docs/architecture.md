# AHCAgent CLI ツール アーキテクチャ設計

## 1. パッケージ構造

```
ahc_agent_cli/
├── __init__.py            # バージョン情報、パッケージメタデータ
├── cli.py                 # CLIコマンド定義 (Clickベース)
├── config.py              # 設定管理 (デフォルト設定、YAML/環境変数からの読み込み)
├── core/                  # コアロジックモジュール群
│   ├── __init__.py
│   ├── analyzer.py        # 問題分析モジュール (LLM使用)
│   ├── debugger.py        # 実装・デバッグモジュール (Docker経由でのコンパイル・テスト、LLMによるエラー修正)
│   ├── engine.py          # 進化的探索エンジン (LLMによる解の生成・変異・交叉)
│   ├── knowledge.py       # 知識ベース・実験ログ管理 (ファイルシステムベース)
│   ├── problem_logic.py   # AHC問題固有ロジック (LLMによるテストケース生成、スコア計算機生成など)
│   └── strategist.py      # 解法戦略モジュール (LLM使用)
└── utils/                 # ユーティリティモジュール群
    ├── __init__.py
    ├── docker_manager.py  # Docker操作ユーティリティ
    ├── file_io.py         # ファイル入出力ユーティリティ
    ├── llm.py             # LLM (LiteLLM) 通信ユーティリティ
    └── logging.py         # ログ管理ユーティリティ

tests/                     # テストコード
├── __init__.py
├── conftest.py            # Pytestフィクスチャ
├── test_cli.py            # CLIモジュールのテスト
├── test_config.py         # 設定管理モジュールのテスト
├── test_docker_manager.py # Docker管理モジュールのテスト
├── test_file_io.py        # ファイルIOユーティリティのテスト
└── test_llm.py            # LLMクライアントのテスト

.gitignore                 # Git無視ファイル
.pre-commit-config.yaml    # pre-commitフック設定
architecture.md            # (このファイル) アーキテクチャ設計書
pyproject.toml             # プロジェクト設定、依存関係
README.md                  # プロジェクト概要、使い方
sample_problem.md          # サンプル問題定義
validate.sh                # 検証用シェルスクリプト
```

## 2. CLIコマンド設計

```
ahc-agent [OPTIONS] COMMAND [ARGS]...
```

### グローバルオプション

* `--config PATH` (`-c`): 設定ファイルのパスを指定します。
* `--workspace PATH` (`-w`): 作業ディレクトリのパスを指定します。
* `--verbose` (`-v`): 詳細なログ（デバッグレベル）を出力します。
* `--quiet` (`-q`): 最小限のログ（エラーレベルのみ）を出力します。
* `--no-docker`: Dockerを使用せずにローカル環境で実行を試みます（`docker.enabled`設定を`False`に上書き）。

### コマンド

1.  **`init`**
    ```
    ahc-agent init [OPTIONS]
    ```
    * 新しいAHCプロジェクトを指定されたワークスペースに初期化します。
    * ワークスペースディレクトリと設定ファイル (`ahc_config.yaml`) を作成します。
    * オプション:
        * `--template NAME` (`-t`): 使用するテンプレートを指定します (現在、具体的なテンプレート機能は限定的)。
        * `--docker-image IMAGE` (`-i`): プロジェクトのデフォルトDockerイメージを設定ファイルに記録します。

2.  **`solve`**
    ```
    ahc-agent solve [OPTIONS] PROBLEM_FILE
    ```
    * 指定された問題記述ファイル (`PROBLEM_FILE`) に基づいて問題を解きます。
    * 問題分析、戦略立案、解の進化、評価のプロセスを実行します。
    * オプション:
        * `--session-id ID` (`-s`): 既存のセッションIDを指定して処理を再開します。
        * `--time-limit SECONDS` (`-t`): 進化アルゴリズムの時間制限（秒）を設定します。
        * `--generations NUM` (`-g`): 進化アルゴリズムの最大世代数を設定します。
        * `--population-size NUM` (`-p`): 進化アルゴリズムの個体群サイズを設定します。
        * `--interactive` (`-i`): 対話モードで問題解決プロセスを進めます。

3.  **`status`**
    ```
    ahc-agent status [OPTIONS] [SESSION_ID]
    ```
    * 指定されたセッションIDのステータス、または全セッションのリストを表示します。
    * オプション:
        * `--watch` (`-w`): 指定されたセッションのステータスを継続的に監視し、更新表示します。

4.  **`stop`**
    ```
    ahc-agent stop [OPTIONS] SESSION_ID
    ```
    * 指定されたセッションIDの処理（主に監視モードや将来的な長時間実行プロセス）を停止状態にマークします。

5.  **`submit`**
    ```
    ahc-agent submit [OPTIONS] SESSION_ID
    ```
    * 指定されたセッションIDから最良解を取得し、標準出力または指定ファイルに出力します。
    * オプション:
        * `--output PATH` (`-o`): 解のコードを出力するファイルパスを指定します。

6.  **`batch`**
    ```
    ahc-agent batch [OPTIONS] BATCH_CONFIG
    ```
    * 指定されたバッチ設定ファイル (`BATCH_CONFIG`) に基づいて、複数の実験を連続的または並列的に実行します。
    * オプション:
        * `--parallel NUM` (`-p`): 並列実行数を指定します。
        * `--output-dir PATH` (`-o`): バッチ処理結果の出力ディレクトリを指定します。

7.  **`config`** (サブコマンドグループ)
    ```
    ahc-agent config [OPTIONS] COMMAND [ARGS]...
    ```
    * 設定の管理を行います。
    * サブコマンド:
        * `get KEY`: 指定されたキーの設定値を取得して表示します。
        * `set KEY VALUE`: 指定されたキーに新しい値を設定します（現在のセッションのみ、永続化はしない想定）。
        * `export PATH`: 現在の有効な設定（デフォルト、ファイル、環境変数をマージ後）を指定ファイルにエクスポートします。
        * `import PATH`: 指定ファイルから設定を読み込み、現在の設定にマージします。

8.  **`docker`** (サブコマンドグループ)
    ```
    ahc-agent docker [OPTIONS] COMMAND [ARGS]...
    ```
    * Docker環境の管理を行います。
    * サブコマンド:
        * `setup`: 設定ファイルで指定されたDockerイメージをプルします。
        * `status`: Dockerデーモンの接続状態やテストコマンドの実行により、Docker環境が利用可能か表示します。
        * `cleanup`: 不要なDockerリソース（停止したコンテナなど）をクリーンアップします (`docker container prune -f`)。

## 3. 設定管理

### 設定ファイル形式 (YAML)

設定は `config.py` 内の `Config` クラスによって管理されます。デフォルト設定、YAMLファイル、環境変数の順でマージされます。

**デフォルト設定の構造 (`ahc_agent_cli/config.py` より):**
```yaml
# AHCAgent CLI 設定 (ahc_config.yaml の例)

# LLM設定 (LiteLLM経由)
llm:
  provider: "litellm"  # LiteLLMがサポートするプロバイダ名 (例: "openai", "anthropic", "google")
  model: "gpt-4"       # 使用するモデル名
  temperature: 0.7     # 生成時の多様性 (0.0-1.0)
  max_tokens: 4000     # 最大生成トークン数
  timeout: 60          # APIリクエストのタイムアウト（秒）
  # api_key: "YOUR_API_KEY" # APIキー (環境変数での設定を推奨)

# Docker設定
docker:
  enabled: true       # Dockerを使用するかどうか
  image: "mcr.microsoft.com/devcontainers/rust:1-1-bullseye" # C++開発ツールも含むRust開発コンテナイメージ
  mount_path: "/workspace" # コンテナ内のワークスペースマウントパス
  cpp_compiler: "g++"      # C++コンパイラコマンド
  cpp_flags: "-std=c++17 -O2 -Wall" # C++コンパイルフラグ
  # timeout: 300 (run_commandのデフォルトタイムアウト、DockerManagerのDEFAULT_CONFIGで定義)
  # build_timeout: 300 (build_imageのデフォルトタイムアウト、DockerManagerのDEFAULT_CONFIGで定義)

# ワークスペース設定
workspace:
  base_dir: "~/ahc_workspace" # プロジェクトファイルやセッションデータが保存されるベースディレクトリ
  # keep_history: true (KnowledgeBaseでセッションごとに管理)
  # max_sessions: 10 (現在、明示的な制限はなし)

# 進化的アルゴリズム設定
evolution:
  max_generations: 30                # 最大世代数
  population_size: 10                # 個体群サイズ
  time_limit_seconds: 1800           # 進化処理全体の時間制限（秒）
  score_plateau_generations: 5       # スコア改善が見られない場合に停止するまでの世代数

# 各コアモジュール向け設定 (例)
analyzer:
  detailed_analysis: true # 問題分析の詳細度など

strategist:
  detailed_strategy: true # 戦略立案の詳細度など

debugger:
  execution_timeout: 10    # C++コードの実行タイムアウト（秒）

problem_logic:
  test_cases_count: 3    # 自動生成するテストケースのデフォルト数

# バッチ処理設定
batch:
  parallel: 1                 # デフォルトの並列実行数
  output_dir: "~/ahc_batch"   # バッチ処理結果のデフォルト出力ディレクトリ
```

### 環境変数

設定は環境変数によっても上書き可能です。環境変数は `AHC_` プレフィックスに続き、セクションとキーをアンダースコアで連結した形式で指定します (例: `AHC_LLM_MODEL="gpt-4-turbo"`)。

* `AHC_LLM_PROVIDER`: LLMプロバイダ (例: `openai`)
* `AHC_LLM_MODEL`: LLMモデル (例: `gpt-4-turbo`)
* `AHC_LLM_TEMPERATURE`: 温度設定 (例: `0.5`)
* `AHC_DOCKER_ENABLED`: Docker有効化 (`true`/`false`)
* `AHC_DOCKER_IMAGE`: Dockerイメージ名
* `AHC_WORKSPACE_BASE_DIR`: ワークスペースのベースディレクトリ
* LLMのAPIキーは、`AHC_LLM_API_KEY` または各プロバイダ固有の環境変数 (例: `OPENAI_API_KEY`) を `LLMClient` (`llm.py`) が参照します。

## 4. Docker統合

`ahc_agent_cli.utils.docker_manager.DockerManager` クラスがDocker操作を担当します。

### Dockerコンテナ管理

1.  **イメージ取得**:
    * 設定されたDockerイメージ (`docker.image`) を `docker pull` します (`ahc-agent docker setup` コマンド経由など)。
2.  **コマンド実行**:
    * `docker run --rm -v <host_work_dir>:<mount_path> -w <mount_path> <image> /bin/bash -c "<command>"` の形式で汎用コマンドを実行します。
    * ホストの作業ディレクトリがコンテナの `/workspace` (デフォルト) にマウントされます。
3.  **C++コードのコンパイルと実行**:
    * `compile_cpp`: 指定されたC++ソースファイルをコンテナ内でコンパイルします。コンパイラやフラグは設定に基づきます (`docker.cpp_compiler`, `docker.cpp_flags`)。
    * `run_cpp` (現在は `run_executable` として `ImplementationDebugger` 内で利用): コンパイルされた実行可能ファイルをコンテナ内で実行し、標準入出力を扱います。実行時間制限も適用されます。
4.  **ファイルコピー**:
    * `copy_to_container`: ホストからコンテナへファイルをコピーします。
    * `copy_from_container`: コンテナからホストへファイルをコピーします。
5.  **イメージビルド**:
    * `build_image`: 指定されたコンテキストパスとDockerfileを使って新しいDockerイメージをビルドします。
6.  **リソース管理**:
    * `cleanup`: 停止したコンテナを削除 (`docker container prune -f`) します。

### Dockerfileテンプレートの想定

`DockerManager`自体は特定のDockerfileテンプレートを持ちませんが、ユーザーがカスタムイメージをビルドする際のDockerfileの例として以下のようなものが考えられます (C++開発環境を想定):

```dockerfile
FROM mcr.microsoft.com/devcontainers/rust:1-1-bullseye
# AtCoderのVisualizerがRustで書かれているため、Rust開発環境をベースイメージとしています
# このベースイメージには既に g++, cmake, make, python3, python3-pip などが含まれていることが多い

# 必要に応じて追加のパッケージをインストール
# RUN apt-get update && apt-get install -y \
#     my-other-tool \
#     && rm -rf /var/lib/apt/lists/*

# 作業ディレクトリの設定 (DockerManagerのmount_pathと合わせる)
WORKDIR /workspace

# エントリーポイントやデフォルトコマンドはDockerManager側で指定されるため、ここでは通常不要
```

## 5. 対話モード (`ahc-agent solve --interactive`)

`cli.py` 内の `_interactive_solve` 関数によって対話的な問題解決プロセスが提供されます。

### 対話フローの例

1.  **初期化**: セッション情報、問題の概要表示。
2.  **コマンド入力**: ユーザーが次の操作を選択。
    * `analyze`: 問題分析 (`ProblemAnalyzer` を使用)。結果の概要表示。
    * `strategy`: 解法戦略の立案 (`SolutionStrategist` を使用)。戦略の概要表示。
    * `testcases`: テストケース生成 (`ProblemLogic` を使用)。スコア計算機も準備。
    * `initial`: 初期解生成 (`ProblemLogic` を使用)。
    * `evolve`: 進化的探索の実行 (`EvolutionaryEngine` を使用)。パラメータは対話的に設定可能。
    * `status`: 現在の各ステップの完了状態や最良スコアなどを表示。
    * `help`: コマンド一覧を表示。
    * `exit`: 対話モードを終了。
3.  各コマンド実行後、結果の概要が表示され、再度コマンド入力状態に戻ります。

### インタラクティブコマンド

* `analyze`: 問題分析の実行と結果表示。
* `strategy`: 解法戦略の立案と結果表示。
* `testcases`: テストケースの生成とスコア計算機の準備。
* `initial`: 初期解の生成と保存。
* `evolve`: 進化プロセスの実行と結果表示。
* `status`: 現在のセッションの状態（分析、戦略、テストケース、最良スコアなど）を表示。
* `help`: 利用可能なコマンドを表示。
* `exit`: 対話モードを終了。

## 6. バッチ処理モード (`ahc-agent batch`)

`cli.py` 内の `batch` コマンドと `_run_batch_experiments` 関数によって、複数の実験設定に基づいたバッチ処理が可能です。

### バッチ設定ファイル形式 (YAML)

```yaml
# AHCAgent バッチ設定 (例: batch_config.yaml)

# 共通設定 (各実験のデフォルト値として使用可能)
common:
  workspace: "~/ahc_batch_results" # バッチ処理全体の出力先 (オプション)
  time_limit: 1800                 # 1実験あたりの時間制限 (秒)
  max_generations: 30              # 1実験あたりの最大世代数
  # DockerイメージやLLMモデルなども共通設定可能

# 問題ファイル定義
problems:
  - path: "problems/problem1.md" # 問題記述ファイルへのパス
    name: "problem1"             # 問題の識別名
  - path: "problems/problem2.txt"
    name: "problem2"

# パラメータセット定義
parameter_sets:
  - name: "small_population"       # パラメータセットの識別名
    evolution.population_size: 5 # 上書きする設定 (Configのキー形式)
    evolution.score_plateau_generations: 3
  - name: "large_population"
    evolution.population_size: 20
    llm.model: "gpt-4-turbo"

# 実験設定
experiments:
  - problem: "problem1"                 # 使用する問題名 (problemsで定義)
    parameter_set: "small_population" # 使用するパラメータセット名 (parameter_setsで定義)
    repeats: 3                        # この実験設定での繰り返し回数
  - problem: "problem2"
    parameter_set: "large_population"
    repeats: 2
  - problem: "problem1"                 # パラメータセットを指定しない場合、共通設定やデフォルト設定を使用
    repeats: 1
```

### 並列実行

* `ahc-agent batch --parallel NUM` で指定された数に基づき、実験タスクが並列実行されます (`asyncio.gather` を使用)。
* 各実験は、その実験用の設定（共通設定、問題設定、パラメータセットをマージ）に基づいて独立して実行されます。
* 結果は実験ごとに指定された出力ディレクトリ (デフォルトは `common.workspace` / `experiment_id`) に保存され、最後に全実験結果のサマリーがJSON形式で出力されます。

## 7. エラー処理

### エラータイプ (想定されるもの)

1.  **設定エラー**:
    * 設定ファイルの読み込みエラー (YAML形式不正など)。
    * 必須設定値の欠如。
2.  **Docker関連エラー**:
    * Dockerデーモン非実行・接続エラー。
    * 指定イメージの取得失敗。
    * コンテナ作成・実行エラー (リソース不足、コマンドエラーなど)。
    * コンパイルエラー、実行タイムアウト。
3.  **LLM APIエラー**:
    * APIキー未設定・不正。
    * API接続エラー (ネットワーク問題、レート制限)。
    * LLMからの無効なレスポンス (期待した形式でない、JSONパースエラーなど)。
4.  **ファイルI/Oエラー**:
    * 問題ファイルやワークスペースへのアクセス権限不足。
    * ディスク容量不足。
5.  **実行時エラー**:
    * Pythonコード内のバグ。
    * 外部プロセス呼び出しの失敗。

### エラーハンドリング戦略

* **明確なエラーメッセージ**: ユーザーに問題の原因と可能な対処法を伝えるログメッセージ。
* **リトライメカニズム**: LLM API通信など、一時的なエラーが想定される箇所でのリトライ処理 (`LLMClient`内で実装)。
* **グレースフルデグラデーション**: Dockerが無効な場合でも、限定的な機能（ローカル実行など、ただし現状はDocker利用が前提の機能が多い）の提供を試みるか、明確にエラーを通知。
* **状態の保存**: `KnowledgeBase` を通じて、セッションごとの分析結果、戦略、生成された解などをファイルに保存し、エラー発生時や再開時に利用可能にする。
* **詳細なログ記録**: `logging` モジュールを使用して、エラー発生時のコンテキスト情報（スタックトレース、関連データなど）を記録。デバッグレベルのログも活用。

## 8. ログ管理

`ahc_agent_cli.utils.logging` モジュールで設定。

### ログレベル

`click` の `--verbose` / `--quiet` オプション、または設定ファイルや環境変数で制御可能。

* `ERROR`: エラー発生時のみ。
* `WARNING`: 警告およびエラー。
* `INFO`: 通常の動作情報（デフォルト）。
* `DEBUG`: 詳細なデバッグ情報。
* `TRACE`: (現在、明示的なTRACEレベルはなし。DEBUGが最も詳細)

### ログ出力先

* **コンソール**: 標準出力/標準エラー出力。
* **ファイル**: (オプション) 設定でログファイルパスを指定可能。
* **JSONフォーマット**: (オプション) 設定でJSON形式のログ出力を有効化可能。

---

## 9. テスト戦略

`tests/` ディレクトリにテストコードを配置。`pytest` を使用。

### 単体テスト

* 各コアモジュール (`analyzer`, `config`, `docker_manager`, `file_io`, `llm` など) の独立した機能テスト。
* モック (`unittest.mock`) を使用して、LLM API呼び出しやDockerプロセス実行などの外部依存を分離。
* 設定値の境界値テスト、エラーケースのテスト。

### 統合テスト (部分的なもの)

* CLIコマンドの呼び出しと基本的な動作確認 (`test_cli.py`)。
* 複数のモジュールが連携するシナリオのテスト（例: `solve` コマンド実行時の主要フローの一部を模倣）。

### `validate.sh` によるエンドツーエンドに近いテスト

* `validate.sh` スクリプトは、`ahc-agent init`, `docker status`, `solve` (サンプル問題使用), `status`, `submit` といった一連のコマンドを実行し、ツール全体の基本的な動作を検証します。これは手動実行を想定した簡易的なE2Eテストとして機能します。

### パフォーマンステスト

* 現状、自動化されたパフォーマンステストの具体的な仕組みはリポジトリからは明確ではないが、`EvolutionaryEngine` の時間制限機能や、バッチ処理モードでの実行時間計測などが手動での性能評価に利用可能。
