import os  # For os.getcwd
from pathlib import Path
from unittest.mock import mock_open, patch
from urllib.parse import urljoin

import pytest

from ahc_agent.services.init_service import InitService


class TestInitService:
    @patch("ahc_agent.services.init_service.scrape_and_setup_problem")
    @patch("ahc_agent.services.init_service.yaml.dump")
    @patch("pathlib.Path.mkdir")  # Patching Path.mkdir directly
    @patch("builtins.open", new_callable=mock_open)  # Mocks open for writing config
    def test_initialize_project_success_with_args_and_workspace(self, mock_file_open, mock_path_mkdir, mock_yaml_dump, mock_scraper, tmp_path):
        init_service = InitService()

        contest_id = "ahc001"
        workspace_name = "test_workspace_ahc001"
        # Define workspace path relative to tmp_path for test isolation
        workspace_path_str = str(tmp_path / workspace_name)

        # Act
        result = init_service.initialize_project(contest_id=contest_id, workspace=workspace_path_str)

        # Assert
        expected_project_dir_obj = Path(workspace_path_str) / contest_id
        # Check that Path(workspace_path_str).mkdir() was called
        # The actual call is on an instance of Path, so we check the mock_path_mkdir directly
        # as if it's the method of the specific Path instance.
        # Path(workspace_path_str) itself isn't mocked, its .mkdir method is.
        # To check which Path instance mkdir was called on, we would need to inspect Path() calls.
        # For now, assume Path.mkdir is globally patched and any call to it is caught.
        mock_path_mkdir.assert_called_once_with(parents=True, exist_ok=False)

        # Check config.yaml write
        expected_config_file_path = expected_project_dir_obj / "config.yaml"
        # First call to open is for config.yaml
        # mock_file_open.assert_any_call(expected_config_file_path, "w")

        mock_yaml_dump.assert_called_once()
        dumped_data = mock_yaml_dump.call_args[0][0]

        assert dumped_data["contest_id"] == contest_id
        # assert dumped_data["template"] == custom_template_arg  # Uses passed template
        # assert dumped_data["docker_image"] == custom_image_arg  # Uses passed image

        problem_url = f"https://atcoder.jp/contests/{contest_id}/tasks/{contest_id}_a"
        mock_scraper.assert_called_once_with(url=problem_url, base_output_dir=str(expected_project_dir_obj), contest_id_for_filename=contest_id)

        assert result["project_dir"] == str(expected_project_dir_obj)
        assert result["config_file_path"] == str(expected_config_file_path)

    @patch("ahc_agent.services.init_service.scrape_and_setup_problem")
    @patch("ahc_agent.services.init_service.yaml.dump")
    @patch("pathlib.Path.mkdir")
    @patch("builtins.open", new_callable=mock_open)
    def test_initialize_project_success_defaults_no_workspace(self, mock_file_open, mock_path_mkdir, mock_yaml_dump, mock_scraper, tmp_path):
        # Temporarily change CWD for this test to ensure default workspace logic is predictable
        original_cwd = os.getcwd()
        os.chdir(tmp_path)

        init_service = InitService()
        contest_id = "ahc002"

        # Act: workspace=None, template=None, docker_image=None
        result = init_service.initialize_project(contest_id=contest_id)

        # Assert
        # Default project_dir is os.getcwd() / contest_id
        expected_project_dir_obj = tmp_path / contest_id
        mock_path_mkdir.assert_called_once_with(parents=True, exist_ok=False)

        expected_config_file_path = expected_project_dir_obj / "config.yaml"
        # mock_file_open.assert_any_call(expected_config_file_path, "w")

        mock_yaml_dump.assert_called_once()
        dumped_data = mock_yaml_dump.call_args[0][0]
        assert dumped_data["contest_id"] == contest_id

        problem_url = f"https://atcoder.jp/contests/{contest_id}/tasks/{contest_id}_a"
        mock_scraper.assert_called_once_with(url=problem_url, base_output_dir=str(expected_project_dir_obj), contest_id_for_filename=contest_id)

        assert result["project_dir"] == str(expected_project_dir_obj)
        assert result["config_file_path"] == str(expected_config_file_path)

        # Restore CWD
        os.chdir(original_cwd)

    @patch("pathlib.Path.mkdir")
    def test_initialize_project_mkdir_fails(self, mock_path_mkdir, tmp_path):
        init_service = InitService()
        # Configure the specific Path instance's mkdir method to raise error
        # This requires a bit more advanced patching or a different approach if we need to target
        # the specific Path instance created inside initialize_project.
        # For now, patching pathlib.Path.mkdir globally is simpler.
        mock_path_mkdir.side_effect = FileExistsError("Mocked FileExistsError from Path.mkdir")

        with pytest.raises(RuntimeError) as exc_info:
            init_service.initialize_project(contest_id="ahc002", workspace=str(tmp_path / "anywhere"))

        assert "Error creating project directory" in str(exc_info.value)
        # エラーメッセージはサービスの実装によって変更されているので、より汎用的なチェックに変更
        assert "already exists" in str(exc_info.value)

    @patch("pathlib.Path.mkdir")  # Mock mkdir to succeed
    @patch("builtins.open", new_callable=mock_open)  # Mock open to succeed for config
    @patch("ahc_agent.services.init_service.yaml.dump")  # Mock yaml.dump
    @patch("ahc_agent.services.init_service.scrape_and_setup_problem")
    def test_initialize_project_scraper_fails(self, mock_scraper, mock_yaml_dump, mock_file_open, mock_path_mkdir, tmp_path):
        init_service = InitService()

        # Mock scraper to raise an exception
        scraper_error_message = "Mocked Scraper Error"
        mock_scraper.side_effect = Exception(scraper_error_message)

        workspace_path_str = str(tmp_path / "scraper_fail_ws")

        # InitService's initialize_project currently logs the scraper error and continues,
        # it does not re-raise it as a RuntimeError. Let's verify the logging and successful return.
        # If the requirement changes to re-raise, this test needs adjustment.

        # To check logger calls, we patch the module-level logger
        with patch("ahc_agent.services.init_service.logger.error") as mock_logger_error:
            result = init_service.initialize_project(contest_id="ahc003", workspace=workspace_path_str)

            # Assert that directory creation and config writing still happened
            mock_path_mkdir.assert_called_once()
            mock_yaml_dump.assert_called_once()

            # Assert scraper was called
            expected_project_dir_path = Path(workspace_path_str) / "ahc003"  # Renamed for clarity
            problem_url = "https://atcoder.jp/contests/ahc003/tasks/ahc003_a"
            mock_scraper.assert_called_once_with(url=problem_url, base_output_dir=str(expected_project_dir_path), contest_id_for_filename="ahc003")

            # Assert logger.error was called due to scraper failure
            mock_logger_error.assert_called_once()

            # Check returned dict - project_dir should be the one created
            assert result["project_dir"] == str(expected_project_dir_path)  # Use the correct variable
            assert result is not None

    @patch("pathlib.Path.mkdir")
    @patch("builtins.open", new_callable=mock_open)
    @patch("ahc_agent.services.init_service.yaml.dump")
    def test_initialize_project_config_save_fails(self, mock_yaml_dump, mock_file_open, mock_path_mkdir, tmp_path):
        init_service = InitService()

        # Mock yaml.dump to raise an exception
        yaml_error_message = "Mocked YAML Dump Error"
        mock_yaml_dump.side_effect = Exception(yaml_error_message)

        workspace_path_str = str(tmp_path / "yaml_fail_ws")

        with pytest.raises(RuntimeError) as exc_info:
            init_service.initialize_project(contest_id="ahc004", workspace=workspace_path_str)

        mock_path_mkdir.assert_called_once()  # Directory creation was attempted
        # mock_file_open should have been called for config.yaml
        # The path is Path(workspace_path_str) / "config.yaml"
        # Check that open was called with the correct path and mode "w"
        # This depends on the exact sequence in InitService.
        # If yaml.dump fails, open was successful.

        assert "Error saving project configuration" in str(exc_info.value)
        assert yaml_error_message in str(exc_info.value)

    def test_initialize_project_writes_contest_id_to_config(self, tmp_path):
        # This test verifies that contest_id is written to config.yaml.

        with patch("ahc_agent.services.init_service.scrape_and_setup_problem"), patch(
            "ahc_agent.services.init_service.yaml.dump"
        ) as mock_yaml_dump, patch("pathlib.Path.mkdir"), patch("builtins.open", new_callable=mock_open):
            init_service = InitService()
            contest_id = "ahc005"
            init_service.initialize_project(contest_id, workspace=str(tmp_path))

            expected_data_to_dump = {
                "contest_id": contest_id,
            }
            mock_yaml_dump.assert_called_once()
            actual_data_dumped = mock_yaml_dump.call_args[0][0]
            assert actual_data_dumped == expected_data_to_dump

    @patch("ahc_agent.utils.scraper.download_and_extract_visualizer")  # scraper内の関数をモック
    @patch("ahc_agent.utils.scraper.fetch_problem_statement")  # scraper内の関数をモック
    @patch("ahc_agent.services.init_service.yaml.dump")
    @patch("pathlib.Path.mkdir")
    @patch("builtins.open", new_callable=mock_open)
    def test_initialize_project_with_local_html_and_relative_tool_link_using_contest_id(
        self,
        mock_builtin_open,
        mock_path_mkdir,
        mock_yaml_dump,
        mock_fetch_problem_statement,  # utils.scraper.fetch_problem_statement
        mock_download_visualizer,  # utils.scraper.download_and_extract_visualizer
        tmp_path,
    ):
        init_service = InitService()
        contest_id = "ahc997"
        workspace_name = "test_workspace_html_tools"
        workspace_path = tmp_path / workspace_name
        project_dir_path = workspace_path / contest_id

        dummy_html_content_with_tools = f"""
        <html><body>
        <div id=\"task-statement\">
            <h1>Problem Title for {contest_id}</h1>
            <p>This is the problem statement.</p>
            <p>Local tools (relative): <a href=\"tools.zip\">tools.zip</a></p>
            <p>Other tools (absolute): <a href=\"https://img.atcoder.jp/{contest_id}/another_tool.zip\">another_tool.zip</a></p>
        </div>
        </body></html>
        """
        dummy_html_file_path = tmp_path / f"dummy_{contest_id}_problem.html"
        with open(dummy_html_file_path, "w", encoding="utf-8") as f:
            f.write(dummy_html_content_with_tools)

        # builtins.open のモック設定を調整して、HTMLファイル読み込みとconfig.yaml書き込みを正しく処理する
        m_config_open = mock_open()  # config.yaml書き込み用のモックインスタンス

        def open_side_effect(path, mode="r", **kwargs):
            # Pathオブジェクトに変換してresolve()することで、パス比較の信頼性を上げる
            resolved_path_str = str(Path(path).resolve())
            expected_html_path_str = str(dummy_html_file_path.resolve())
            expected_config_path_str = str((project_dir_path / "config.yaml").resolve())

            if resolved_path_str == expected_html_path_str and "r" in mode:
                return mock_open(read_data=dummy_html_content_with_tools).return_value
            if resolved_path_str == expected_config_path_str and "w" in mode:  # elif を if に変更
                return m_config_open.return_value  # config.yaml書き込み用

            # 他の予期しないファイルアクセスは、デフォルトのmock_openの振る舞いに任せるか、エラーにする
            # ここでは、他のテストに影響を与えないよう、新しいモックインスタンスを返す
            return mock_open().return_value

        mock_builtin_open.side_effect = open_side_effect

        # fetch_problem_statement のモック設定
        # 戻り値: (md_content, filename_suggestion, visualizer_zip_url)
        # visualizer_zip_url には、HTMLから抽出された(と仮定する)最初のzipリンクを返す
        # 実際の scraper.py の fetch_problem_statement はもっと複雑なロジックで visualizer_zip_url を決定する
        mock_fetch_problem_statement.return_value = ("# Dummy Problem Statement", "tools.zip")

        # download_and_extract_visualizer のモックは、呼び出されたことを確認するために使う
        mock_download_visualizer.return_value = True

        # InitService の initialize_project を呼び出す
        # これにより、内部で scrape_and_setup_problem が呼び出され、
        # さらにその内部で fetch_problem_statement と download_and_extract_visualizer が呼び出される(はず)
        init_service.initialize_project(contest_id=contest_id, workspace=str(workspace_path), html_file=str(dummy_html_file_path))

        # アサーション
        # 1. fetch_problem_statement が期待通り呼び出されたか
        #    html_content が渡され、url は None またはベースURLとして使われるダミーURL
        #    (現在の scrape_and_setup_problem の実装では、html_file_path がある場合、url は actual_url_for_visualizer_context になる)
        #    ここでは html_content が渡されることを確認
        mock_fetch_problem_statement.assert_called_once()
        call_args = mock_fetch_problem_statement.call_args
        assert call_args[1]["html_content"] == dummy_html_content_with_tools
        # url は None か、あるいは scrape_and_setup_problem に渡された url (このテストケースでは None)
        # もし scrape_and_setup_problem に url が渡された場合、それが fetch_problem_statement にも渡る
        # 今回は init_service に url を渡していないので、scraper にも url=None で渡るはず
        assert call_args[1]["url"] is None

        expected_resolved_url = f"https://img.atcoder.jp/{contest_id}/tools.zip"
        mock_download_visualizer.assert_called_once_with(
            expected_resolved_url,  # contest_id を使って解決されたURL
            str(project_dir_path / "tools"),
        )

    @patch("ahc_agent.utils.scraper.download_and_extract_visualizer")
    @patch("ahc_agent.utils.scraper.fetch_problem_statement")
    @patch("ahc_agent.services.init_service.yaml.dump")
    @patch("pathlib.Path.mkdir")
    @patch("builtins.open", new_callable=mock_open)
    def test_initialize_project_with_local_html_and_absolute_tool_link(
        self, mock_builtin_open, mock_path_mkdir, mock_yaml_dump, mock_fetch_problem_statement, mock_download_visualizer, tmp_path
    ):
        init_service = InitService()
        contest_id = "ahc998"
        workspace_name = "test_workspace_html_absolute_tools"
        workspace_path = tmp_path / workspace_name
        project_dir_path = workspace_path / contest_id

        absolute_tool_url = f"https://example.com/specific_tools_for_{contest_id}.zip"
        dummy_html_content_with_absolute_tools = f"""
        <html><body>
        <div id=\"task-statement\">
            <h1>Problem Title for {contest_id}</h1>
            <p>Absolute tools: <a href=\"{absolute_tool_url}\">{absolute_tool_url.split("/")[-1]}</a></p>
        </div>
        </body></html>
        """
        dummy_html_file_path = tmp_path / f"dummy_{contest_id}_problem_absolute.html"
        with open(dummy_html_file_path, "w", encoding="utf-8") as f:
            f.write(dummy_html_content_with_absolute_tools)

        m_config_open = mock_open()

        def open_side_effect(path, mode="r", **kwargs):
            resolved_path_str = str(Path(path).resolve())
            expected_html_path_str = str(dummy_html_file_path.resolve())
            expected_config_path_str = str((project_dir_path / "config.yaml").resolve())
            if resolved_path_str == expected_html_path_str and "r" in mode:
                return mock_open(read_data=dummy_html_content_with_absolute_tools).return_value
            if resolved_path_str == expected_config_path_str and "w" in mode:
                return m_config_open.return_value
            return mock_open().return_value

        mock_builtin_open.side_effect = open_side_effect

        mock_fetch_problem_statement.return_value = ("# Dummy Absolute Problem", absolute_tool_url)
        mock_download_visualizer.return_value = True

        init_service.initialize_project(contest_id=contest_id, workspace=str(workspace_path), html_file=str(dummy_html_file_path))

        mock_fetch_problem_statement.assert_called_once()
        call_args_fetch = mock_fetch_problem_statement.call_args
        assert call_args_fetch[1]["html_content"] == dummy_html_content_with_absolute_tools
        assert call_args_fetch[1]["url"] is None

        mock_download_visualizer.assert_called_once_with(
            absolute_tool_url,  # 絶対URLはそのまま使われる
            str(project_dir_path / "tools"),
        )

    @patch("ahc_agent.utils.scraper.download_and_extract_visualizer")
    @patch("ahc_agent.utils.scraper.fetch_problem_statement")
    @patch("ahc_agent.services.init_service.yaml.dump")
    @patch("pathlib.Path.mkdir")
    @patch("builtins.open", new_callable=mock_open)
    def test_initialize_project_with_local_html_and_relative_tool_link_using_base_url(
        self,
        mock_builtin_open,
        mock_path_mkdir,
        mock_yaml_dump,  # これが正しい位置
        mock_fetch_problem_statement,
        mock_download_visualizer,
        tmp_path,
    ):
        init_service = InitService()
        contest_id = "ahc999"
        workspace_name = "test_workspace_html_base_url_tools"
        workspace_path = tmp_path / workspace_name
        project_dir_path = workspace_path / contest_id
        base_url_for_html_context = f"https://atcoder.jp/contests/{contest_id}/tasks/"
        relative_tool_path = "another_tool.zip"
        expected_resolved_tool_url = urljoin(base_url_for_html_context, relative_tool_path)

        dummy_html_content_with_relative_tools = f"""
        <html><body>
        <div id=\"task-statement\">
            <h1>Problem Title for {contest_id}</h1>
            <p>Relative tools: <a href=\"{relative_tool_path}\">{relative_tool_path}</a></p>
        </div>
        </body></html>
        """
        dummy_html_file_path = tmp_path / f"dummy_{contest_id}_problem_relative_base.html"
        with open(dummy_html_file_path, "w", encoding="utf-8") as f:
            f.write(dummy_html_content_with_relative_tools)

        m_config_open = mock_open()

        def open_side_effect(path, mode="r", **kwargs):
            resolved_path_str = str(Path(path).resolve())
            expected_html_path_str = str(dummy_html_file_path.resolve())
            expected_config_path_str = str((project_dir_path / "config.yaml").resolve())
            if resolved_path_str == expected_html_path_str and "r" in mode:
                return mock_open(read_data=dummy_html_content_with_relative_tools).return_value
            if resolved_path_str == expected_config_path_str and "w" in mode:
                return m_config_open.return_value
            return mock_open().return_value

        mock_builtin_open.side_effect = open_side_effect

        mock_fetch_problem_statement.return_value = ("# Dummy Relative Base Problem", relative_tool_path)
        mock_download_visualizer.return_value = True

        init_service.initialize_project(
            contest_id=contest_id,
            workspace=str(workspace_path),
            html_file=str(dummy_html_file_path),
            url=base_url_for_html_context,
        )

        mock_fetch_problem_statement.assert_called_once()
        call_args_fetch = mock_fetch_problem_statement.call_args
        assert call_args_fetch[1]["html_content"] == dummy_html_content_with_relative_tools
        assert call_args_fetch[1]["url"] == base_url_for_html_context

        mock_download_visualizer.assert_called_once_with(
            expected_resolved_tool_url,  # ベースURLで解決されたURL
            str(project_dir_path / "tools"),
        )

    @patch("ahc_agent.services.init_service.scrape_and_setup_problem")
    @patch("ahc_agent.services.init_service.yaml.dump")  # config.yaml書き込みのモック
    @patch("pathlib.Path.mkdir")
    @patch("builtins.open", new_callable=mock_open)  # config.yaml書き込みのモック用
    def test_initialize_project_with_local_html(self, mock_builtin_open, mock_path_mkdir, mock_yaml_dump, mock_scraper_setup, tmp_path):
        init_service = InitService()
        contest_id = "ahc999"
        workspace_name = "test_workspace_html"
        workspace_path = tmp_path / workspace_name
        # workspace_path.mkdir() # この行は削除されました。InitServiceがディレクトリ作成を処理します。

        # ダミーのHTMLファイルを作成
        dummy_html_content = """
        <html><body>
        <div id="task-statement">
            <h1>Problem Title</h1>
            <p>This is the problem statement.</p>
            <section id="constraints"><h2>Constraints</h2><var>N</var> &le; 100</section>
            <a href="path/to/visualizer.zip">Visualizer</a>
        </div>
        </body></html>
        """
        dummy_html_file = workspace_path / "problem.html"
        with open(dummy_html_file, "w", encoding="utf-8") as f:
            f.write(dummy_html_content)

        # Act
        result = init_service.initialize_project(contest_id=contest_id, workspace=str(workspace_path), html_file=str(dummy_html_file))

        # Assert
        expected_project_dir = workspace_path / contest_id
        mock_path_mkdir.assert_called_once_with(parents=True, exist_ok=False)  # プロジェクトディレクトリ作成の確認

        # config.yaml の書き込み確認
        expected_config_path = expected_project_dir / "config.yaml"
        # mock_builtin_open.assert_any_call(expected_config_path, "w") # openが呼ばれたかの確認
        mock_yaml_dump.assert_called_once()
        dumped_data = mock_yaml_dump.call_args[0][0]
        assert dumped_data["contest_id"] == contest_id

        # scrape_and_setup_problem の呼び出し確認
        mock_scraper_setup.assert_called_once_with(
            url=None,
            base_output_dir=str(expected_project_dir),
            html_file_path=str(dummy_html_file),
            contest_id_for_filename=contest_id,
        )

        assert result["project_dir"] == str(expected_project_dir)
        assert result["config_file_path"] == str(expected_config_path)
        assert result["contest_id"] == contest_id
