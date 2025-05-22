# AHCAgent CLI ツール化のための分析

## 既存コンポーネントの分析

### コアモジュール
1. **agent.py** - AHCAgentのメインクラス
   - OpenHandsの`Agent`クラスを継承
   - コマンドハンドラ（solve, status, stop, submit）を実装
   - 問題分析、戦略開発、進化的プロセスの実行を管理

2. **modules/problem_analyzer.py** - 問題分析モジュール
   - LLMを使用して問題文を構造化分析
   - OpenHandsの`LLMClient`に依存

3. **modules/solution_strategist.py** - 解法戦略モジュール
   - LLMを使用して問題に適した解法戦略を開発
   - OpenHandsの`LLMClient`に依存

4. **modules/evolutionary_engine.py** - 進化的探索エンジン
   - AlphaEvolveアルゴリズムのコア実装
   - LLMを使用したコード生成・変異・交叉
   - OpenHandsの`LLMClient`に依存

5. **modules/implementation_debugger.py** - 実装・デバッグモジュール
   - C++コードの生成・コンパイル・実行・デバッグ
   - OpenHandsの`LLMClient`に依存

6. **modules/knowledge_base.py** - 知識ベース・実験ログ
   - 実験結果の保存と分析
   - OpenHandsの`StorageClient`に依存

### 問題固有ロジック
1. **evaluation/ahc/ahc_problem_logic.py** - AHC問題固有のロジック
   - 問題文解析、スコア計算、初期解生成、解の変異・交叉
   - OpenHandsの`LLMClient`に依存

2. **evaluation/ahc/validate_ahc_agent.py** - 検証スクリプト
   - サンプルケース生成と検証
   - OpenHandsの`LLMClient`、`StorageClient`、`AHCAgent`に依存

3. **evaluation/ahc/templates/ahc_template.cpp** - C++テンプレート
   - AHC問題解決のためのC++テンプレート
   - 外部依存なし

## 依存関係の分析

### OpenHands依存
1. **LLMClient** - LLMとの通信を管理
2. **StorageClient** - ストレージ操作を管理
3. **Agent** - エージェントの基本クラス

### 外部ライブラリ依存
1. **litellm** - LLM APIの抽象化レイヤー
2. **numpy** - 数値計算
3. **json** - JSON操作
4. **asyncio** - 非同期処理
5. **subprocess** - 外部プロセス実行
6. **os**, **sys**, **time**, **uuid**, **logging** - 標準ライブラリ

## CLIツール化のための移植計画

### 移植するモジュール
1. **problem_analyzer.py** → `ahc_agent_cli/analyzer.py`
   - OpenHands依存を削除し、直接LiteLLMを使用

2. **solution_strategist.py** → `ahc_agent_cli/strategist.py`
   - OpenHands依存を削除し、直接LiteLLMを使用

3. **evolutionary_engine.py** → `ahc_agent_cli/engine.py`
   - OpenHands依存を削除し、直接LiteLLMを使用
   - Dockerベースの実行環境サポートを追加

4. **implementation_debugger.py** → `ahc_agent_cli/debugger.py`
   - OpenHands依存を削除し、直接LiteLLMを使用
   - Dockerベースのコンパイル・実行サポートを追加

5. **knowledge_base.py** → `ahc_agent_cli/knowledge.py`
   - OpenHandsの`StorageClient`依存を削除し、ローカルファイルシステムを直接使用

6. **ahc_problem_logic.py** → `ahc_agent_cli/problem_logic.py`
   - OpenHands依存を削除し、直接LiteLLMを使用

7. **ahc_template.cpp** → `ahc_agent_cli/templates/template.cpp`
   - そのまま移植（依存なし）

### 新規作成するモジュール
1. **cli.py** - CLIエントリーポイントとコマンドハンドラ
   - Click/Typerを使用したCLIインターフェース
   - 対話モード、バッチモードのサポート
   - 設定のインポート/エクスポート機能

2. **docker_manager.py** - Docker実行環境の管理
   - Dockerコンテナの作成・実行・管理
   - C++コンパイル・実行環境の提供

3. **config.py** - 設定管理
   - 環境変数、設定ファイルの読み込み
   - 設定のインポート/エクスポート

4. **utils.py** - ユーティリティ関数
   - ファイル操作、ログ管理など

5. **__main__.py** - パッケージ実行エントリーポイント
   - `python -m ahc_agent_cli` での実行をサポート

### 削除するコンポーネント
1. **agent.py** - OpenHandsの`Agent`クラスに依存するため、完全に再設計

### 依存関係の最小化
1. **必須依存**:
   - litellm: LLM APIの抽象化
   - docker: Dockerコンテナ管理
   - click/typer: CLIインターフェース
   - pyyaml: 設定ファイル管理

2. **オプション依存**:
   - rich: リッチなターミナル出力
   - tqdm: プログレスバー
