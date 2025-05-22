# AHCAgent CLI ツールのアーキテクチャ設計

## パッケージ構造

```
ahc_agent_cli/
├── __init__.py           # バージョン情報、パッケージメタデータ
├── __main__.py           # CLIエントリーポイント
├── cli.py                # CLIコマンド定義
├── config.py             # 設定管理
├── docker_manager.py     # Docker実行環境管理
├── core/                 # コアモジュール
│   ├── __init__.py
│   ├── analyzer.py       # 問題分析モジュール
│   ├── strategist.py     # 解法戦略モジュール
│   ├── engine.py         # 進化的探索エンジン
│   ├── debugger.py       # 実装・デバッグモジュール
│   ├── knowledge.py      # 知識ベース・実験ログ
│   └── problem_logic.py  # AHC問題固有ロジック
├── utils/                # ユーティリティ
│   ├── __init__.py
│   ├── llm.py            # LLM通信ユーティリティ
│   ├── file_io.py        # ファイル操作ユーティリティ
│   └── logging.py        # ログ管理ユーティリティ
└── templates/            # テンプレート
    ├── __init__.py
    └── template.cpp      # C++テンプレート
```

## CLIコマンド設計

```
ahc-agent [OPTIONS] COMMAND [ARGS]...
```

### グローバルオプション
- `--config PATH`: 設定ファイルのパス
- `--workspace PATH`: 作業ディレクトリのパス
- `--verbose`: 詳細なログ出力
- `--quiet`: 最小限のログ出力
- `--no-docker`: Dockerを使用せずにローカル環境で実行

### コマンド

1. **init**
   ```
   ahc-agent init [OPTIONS]
   ```
   - 新しいAHCプロジェクトを初期化
   - オプション:
     - `--template NAME`: 使用するテンプレート
     - `--docker-image IMAGE`: 使用するDockerイメージ

2. **solve**
   ```
   ahc-agent solve [OPTIONS] PROBLEM_FILE
   ```
   - 問題を解く
   - オプション:
     - `--session-id ID`: セッションID（再開用）
     - `--time-limit SECONDS`: 時間制限
     - `--generations NUM`: 最大世代数
     - `--population-size NUM`: 個体群サイズ
     - `--interactive`: 対話モード

3. **status**
   ```
   ahc-agent status [OPTIONS] [SESSION_ID]
   ```
   - セッションのステータスを表示
   - オプション:
     - `--watch`: 継続的に更新

4. **stop**
   ```
   ahc-agent stop [OPTIONS] SESSION_ID
   ```
   - セッションを停止

5. **submit**
   ```
   ahc-agent submit [OPTIONS] SESSION_ID
   ```
   - 最良解を提出
   - オプション:
     - `--output PATH`: 出力ファイルパス

6. **batch**
   ```
   ahc-agent batch [OPTIONS] BATCH_CONFIG
   ```
   - バッチ処理を実行
   - オプション:
     - `--parallel NUM`: 並列実行数
     - `--output-dir PATH`: 出力ディレクトリ

7. **config**
   ```
   ahc-agent config [OPTIONS] COMMAND [ARGS]...
   ```
   - 設定管理
   - サブコマンド:
     - `get KEY`: 設定値を取得
     - `set KEY VALUE`: 設定値を設定
     - `export PATH`: 設定をエクスポート
     - `import PATH`: 設定をインポート

8. **docker**
   ```
   ahc-agent docker [OPTIONS] COMMAND [ARGS]...
   ```
   - Docker環境管理
   - サブコマンド:
     - `setup`: Docker環境をセットアップ
     - `status`: Docker環境のステータスを表示
     - `cleanup`: Docker環境をクリーンアップ

## 設定管理

### 設定ファイル形式 (YAML)

```yaml
# AHCAgent CLI設定

# LLM設定
llm:
  provider: "openai"  # または "anthropic", "google", etc.
  model: "gpt-4"
  api_key: "${OPENAI_API_KEY}"  # 環境変数から読み込み

# Docker設定
docker:
  image: "mcr.microsoft.com/devcontainers/rust:1-1-bullseye"
  mount_path: "/workspace"
  timeout: 300  # 秒

# 進化的アルゴリズム設定
evolution:
  max_generations: 30
  population_size: 10
  time_limit_seconds: 1800
  score_plateau_generations: 5

# C++コンパイル設定
cpp:
  compiler: "g++"
  flags: "-std=c++17 -O2 -Wall"
  execution_timeout: 10  # 秒

# ワークスペース設定
workspace:
  base_dir: "~/ahc_workspace"
  keep_history: true
  max_sessions: 10
```

### 環境変数

- `AHCAGENT_CONFIG`: 設定ファイルのパス
- `AHCAGENT_WORKSPACE`: 作業ディレクトリのパス
- `AHCAGENT_LLM_PROVIDER`: LLMプロバイダ
- `AHCAGENT_LLM_MODEL`: LLMモデル
- `AHCAGENT_LLM_API_KEY`: LLM APIキー
- `AHCAGENT_DOCKER_IMAGE`: Dockerイメージ
- `AHCAGENT_NO_DOCKER`: Dockerを使用しない場合は"1"

## Docker統合

### Dockerコンテナ管理

1. **コンテナ作成**
   - 指定されたイメージを使用
   - ワークスペースディレクトリをマウント
   - 必要なツールをインストール

2. **コード実行**
   - C++コードのコンパイルと実行
   - 入出力のリダイレクト
   - タイムアウト管理

3. **リソース管理**
   - 使用していないコンテナのクリーンアップ
   - ボリュームとイメージの管理

### Dockerfileテンプレート

```dockerfile
FROM mcr.microsoft.com/devcontainers/rust:1-1-bullseye

# 必要なパッケージのインストール
RUN apt-get update && apt-get install -y \
    g++ \
    cmake \
    make \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# 作業ディレクトリの設定
WORKDIR /workspace

# エントリーポイントの設定
ENTRYPOINT ["/bin/bash", "-c"]
```

## 対話モード

### 対話フロー

1. **問題入力**
   - ファイルからの読み込み
   - 直接入力
   - URLからの取得

2. **分析と戦略**
   - 問題の分析結果を表示
   - 提案された戦略を表示
   - ユーザーによる戦略の調整

3. **進化プロセス**
   - リアルタイムの進捗表示
   - 世代ごとの最良スコア表示
   - 一時停止と再開
   - パラメータの動的調整

4. **結果と分析**
   - 最終結果の表示
   - 解の可視化
   - 実験ログの表示

### インタラクティブコマンド

- `status`: 現在のステータスを表示
- `pause`: 処理を一時停止
- `resume`: 処理を再開
- `modify`: パラメータを変更
- `save`: 現在の状態を保存
- `visualize`: 現在の最良解を可視化
- `help`: ヘルプを表示
- `exit`: 終了

## バッチ処理モード

### バッチ設定ファイル形式 (YAML)

```yaml
# AHCAgent バッチ設定

# 共通設定
common:
  workspace: "~/ahc_batch"
  time_limit: 1800
  max_generations: 30

# 問題ファイル
problems:
  - path: "problem1.md"
    name: "problem1"
  - path: "problem2.md"
    name: "problem2"

# パラメータセット
parameter_sets:
  - name: "small_population"
    population_size: 5
    mutation_rate: 0.3
    crossover_rate: 0.7
  - name: "large_population"
    population_size: 20
    mutation_rate: 0.2
    crossover_rate: 0.8

# 実験設定
experiments:
  - problem: "problem1"
    parameter_set: "small_population"
    repeats: 3
  - problem: "problem2"
    parameter_set: "large_population"
    repeats: 3
```

### 並列実行

- 指定された並列数に基づいて実験を分散
- 各実験は独立したDockerコンテナで実行
- リソース使用量の監視と調整
- 結果の集約と比較

## 依存関係

### 必須依存

```
litellm>=1.0.0
docker>=6.0.0
click>=8.0.0
pyyaml>=6.0.0
```

### オプション依存

```
rich>=13.0.0
tqdm>=4.65.0
```

## エラー処理

### エラータイプ

1. **設定エラー**
   - 無効な設定値
   - 設定ファイルの読み込みエラー

2. **Docker関連エラー**
   - Dockerデーモン接続エラー
   - イメージ取得エラー
   - コンテナ作成・実行エラー

3. **LLM APIエラー**
   - API接続エラー
   - レート制限エラー
   - 無効なレスポンス

4. **実行時エラー**
   - C++コンパイルエラー
   - 実行タイムアウト
   - メモリ不足エラー

### エラーハンドリング戦略

- 明確なエラーメッセージ
- リトライメカニズム（特にLLM API）
- グレースフルデグラデーション
- 状態の保存と復元
- 詳細なログ記録

## ログ管理

### ログレベル

- `ERROR`: エラーのみ
- `WARNING`: 警告とエラー
- `INFO`: 基本的な情報（デフォルト）
- `DEBUG`: 詳細なデバッグ情報
- `TRACE`: 非常に詳細なトレース情報

### ログ出力先

- コンソール
- ファイル
- JSONフォーマット（分析用）

## テスト戦略

### 単体テスト

- 各モジュールの独立したテスト
- モックを使用したLLMとDockerの依存関係テスト
- パラメータ境界値テスト

### 統合テスト

- エンドツーエンドのワークフローテスト
- CLIコマンドテスト
- 設定管理テスト

### パフォーマンステスト

- 並列実行のスケーラビリティテスト
- メモリ使用量テスト
- 長時間実行テスト
