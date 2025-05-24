# AHCAgent CLI

AlphaEvolve アルゴリズムを使用して AtCoder Heuristic Contest (AHC) の問題を解決するためのスタンドアロンコマンドラインツールです。

## ✨ 特徴

  * LLM を活用した問題分析と解法戦略の立案
  * 進化的アルゴリズムに基づく解の探索と最適化
  * C++コードのコンパイルと実行のための Docker 統合
  * 対話的な問題解決プロセスをサポートするインタラクティブモード
  * 複数の実験設定を効率的に実行するバッチ処理モード
  * 実験ごとの進捗や結果を管理する知識ベース機能
  * YAMLファイル、環境変数、コマンドラインオプションによる柔軟な設定管理

## 🛠️ インストール

```bash
pip install ahc-agent-cli
```

## 📋 要件

  * Python 3.8 以上
  * Docker (C++コードのコンパイルおよび実行に必要)
  * LLM API アクセス (LiteLLM を介して OpenAI, Anthropic などに対応)

## 🚀 クイックスタート

### 1\. プロジェクトの初期化

作業ディレクトリと設定ファイル (`ahc_config.yaml`) を作成します。

```bash
ahc-agent init --workspace ~/ahc_workspace --docker-image my-cpp-dev-env:latest
```

  * `--workspace PATH` (`-w`): 作業ディレクトリのパスを指定します。
  * `--docker-image IMAGE` (`-i`): プロジェクトのデフォルトDockerイメージを設定ファイルに記録します。

### 2\. 問題の解決

指定された問題記述ファイルに基づいて問題を解きます。

```bash
ahc-agent solve path/to/problem_description.md --time-limit 1800 --generations 50 --population-size 20
```

  * `PROBLEM_FILE`: 問題記述ファイルのパス。
  * `--time-limit SECONDS` (`-t`): 進化アルゴリズムの時間制限（秒）を設定します。
  * `--generations NUM` (`-g`): 進化アルゴリズムの最大世代数を設定します。
  * `--population-size NUM` (`-p`): 進化アルゴリズムの個体群サイズを設定します。

### 3\. インタラクティブモードでの問題解決

対話形式で問題解決プロセスを進めます。

```bash
ahc-agent solve path/to/problem_description.md --interactive
```

インタラクティブモードでは、以下のコマンドが利用可能です:

  * `analyze`: 問題分析を実行
  * `strategy`: 解法戦略を立案
  * `testcases`: テストケースとスコア計算機を生成
  * `initial`: 初期解を生成
  * `evolve`: 進化的探索を実行
  * `status`: 現在の状況を表示
  * `help`: コマンド一覧を表示
  * `exit`: 対話モードを終了

### 4\. セッションステータスの確認

指定されたセッションIDのステータス、または全セッションのリストを表示します。

```bash
ahc-agent status <session_id>
```

特定のセッションのステータスを継続的に監視することも可能です。

```bash
ahc-agent status <session_id> --watch
```

### 5\. 最良解の提出用出力

指定されたセッションIDから最良解を取得し、標準出力または指定ファイルに出力します。

```bash
ahc-agent submit <session_id> --output solution.cpp
```

## ⚙️ 設定

設定は以下の優先順位で管理されます:

1.  コマンドラインオプション
2.  環境変数 (例: `AHC_LLM_MODEL="gpt-4-turbo"`)
3.  設定ファイル (`ahc_config.yaml`)
4.  デフォルト設定

### 設定の管理コマンド

```bash
# 特定の設定値を取得
ahc-agent config get llm.model

# 設定値をセット (現在のセッションのみ、永続化はしない想定)
ahc-agent config set llm.temperature 0.5

# 現在の有効な設定をファイルにエクスポート
ahc-agent config export current_config.yaml

# ファイルから設定を読み込みマージ
ahc-agent config import custom_config.yaml
```

## 📊 バッチ処理

複数の実験を異なるパラメータで連続的または並列的に実行します。

```bash
ahc-agent batch path/to/batch_config.yaml --parallel 4 --output-dir ~/ahc_batch_results
```

  * `BATCH_CONFIG`: バッチ設定ファイルのパス。
  * `--parallel NUM` (`-p`): 並列実行数を指定します。
  * `--output-dir PATH` (`-o`): バッチ処理結果の出力ディレクトリを指定します。

バッチ設定ファイルの例は `architecture.md` を参照してください。

## 🐳 Docker 管理

Docker 環境のセットアップや状態確認、クリーンアップを行います。

```bash
# 設定ファイルで指定されたDockerイメージをプル
ahc-agent docker setup

# Dockerデーモンの接続状態やテストコマンドの実行により、Docker環境が利用可能か表示
ahc-agent docker status

# 不要なDockerリソース (停止したコンテナなど) をクリーンアップ
ahc-agent docker cleanup
```

## 📜 ライセンス

[MIT](https://www.google.com/search?q=LICENSE)
