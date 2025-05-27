from unittest.mock import MagicMock, patch

import pytest
import requests

from ahc_agent.utils.scraper import fetch_problem_statement


@pytest.fixture
def mock_requests_get():
    with patch("requests.get") as mock_get:
        yield mock_get


def create_mock_response(html_content, status_code=200):
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.content = html_content.encode("utf-8")
    mock_resp.raise_for_status = MagicMock()
    if status_code != 200:
        mock_resp.raise_for_status.side_effect = requests.exceptions.HTTPError
    return mock_resp


HTML_TEMPLATE_TASK_STATEMENT = """
<html>
<body>
  <div id='task-statement'>
    {problem_html}
  </div>
</body>
</html>
"""

HTML_VISUALIZER_LINK_HERE = """
<h3>ツール</h3>
<p>入力ジェネレータとビジュアライザは
<a href="https://img.atcoder.jp/ahc001/tools.zip">ここ</a>
からダウンロード出来ます。</p>
"""

HTML_VISUALIZER_LINK_LOCAL = """
<h3>ツール</h3>
<p><a href="https://img.atcoder.jp/ahc002/tools_local.zip">ローカル版</a></p>
"""

HTML_VISUALIZER_LINK_LOCAL_VERSION = """
<h3>Tools</h3>
<p><a href="https://img.atcoder.jp/ahc003/tools_local_version.zip">Local version</a></p>
"""

HTML_NO_VISUALIZER_LINK = """
<h3>ツール</h3>
<p>ビジュアライザはありません。</p>
"""


def test_fetch_problem_statement_visualizer_link_here(mock_requests_get):
    html_content = HTML_TEMPLATE_TASK_STATEMENT.format(problem_html=HTML_VISUALIZER_LINK_HERE)
    mock_requests_get.return_value = create_mock_response(html_content)

    md_content, filename_suggestion, visualizer_zip_url = fetch_problem_statement("https://atcoder.jp/contests/ahc001/tasks/ahc001_a")

    assert md_content is not None
    assert filename_suggestion == "ahc001_a_problem.md"
    assert visualizer_zip_url == "https://img.atcoder.jp/ahc001/tools.zip"


def test_fetch_problem_statement_visualizer_link_local(mock_requests_get):
    html_content = HTML_TEMPLATE_TASK_STATEMENT.format(problem_html=HTML_VISUALIZER_LINK_LOCAL)
    mock_requests_get.return_value = create_mock_response(html_content)

    md_content, filename_suggestion, visualizer_zip_url = fetch_problem_statement("https://atcoder.jp/contests/ahc002/tasks/ahc002_a")

    assert md_content is not None
    assert filename_suggestion == "ahc002_a_problem.md"
    assert visualizer_zip_url == "https://img.atcoder.jp/ahc002/tools_local.zip"


def test_fetch_problem_statement_visualizer_link_local_version(mock_requests_get):
    html_content = HTML_TEMPLATE_TASK_STATEMENT.format(problem_html=HTML_VISUALIZER_LINK_LOCAL_VERSION)
    mock_requests_get.return_value = create_mock_response(html_content)

    md_content, filename_suggestion, visualizer_zip_url = fetch_problem_statement("https://atcoder.jp/contests/ahc003/tasks/ahc003_a")

    assert md_content is not None
    assert filename_suggestion == "ahc003_a_problem.md"
    assert visualizer_zip_url == "https://img.atcoder.jp/ahc003/tools_local_version.zip"


def test_fetch_problem_statement_no_visualizer_link(mock_requests_get):
    html_content = HTML_TEMPLATE_TASK_STATEMENT.format(problem_html=HTML_NO_VISUALIZER_LINK)
    mock_requests_get.return_value = create_mock_response(html_content)

    md_content, filename_suggestion, visualizer_zip_url = fetch_problem_statement("https://atcoder.jp/contests/ahc004/tasks/ahc004_a")

    assert md_content is not None
    assert filename_suggestion == "ahc004_a_problem.md"
    assert visualizer_zip_url is None


def test_fetch_problem_statement_request_error(mock_requests_get):
    mock_requests_get.return_value = create_mock_response("", status_code=404)

    md_content, filename_suggestion, visualizer_zip_url = fetch_problem_statement("https://atcoder.jp/contests/ahc005/tasks/ahc005_a")

    assert md_content is None
    assert filename_suggestion is None
    assert visualizer_zip_url is None


def test_fetch_problem_statement_no_task_statement(mock_requests_get):
    html_content = "<html><body><p>No task statement here.</p></body></html>"
    mock_requests_get.return_value = create_mock_response(html_content)

    md_content, filename_suggestion, visualizer_zip_url = fetch_problem_statement("https://atcoder.jp/contests/ahc006/tasks/ahc006_a")

    assert md_content is None
    assert filename_suggestion is None
    assert visualizer_zip_url is None


HTML_VISUALIZER_LINK_PATTERN_SINGLE = """
<h3>ツール</h3>
<p><a href="https://img.atcoder.jp/ahc007/tools.zip">Download Tools</a></p>
"""

HTML_VISUALIZER_LINK_PATTERN_MULTIPLE_KEYWORD = """
<h3>ツール</h3>
<p><a href="https://img.atcoder.jp/ahc008/another.zip">Another Zip</a></p>
<p><a href="https://img.atcoder.jp/ahc008/tools_local.zip">ローカル版ツール</a></p>
<p><a href="https://img.atcoder.jp/ahc008/onemore.zip">One More Zip</a></p>
"""

HTML_VISUALIZER_LINK_PATTERN_MULTIPLE_NO_KEYWORD = """
<h3>ツール</h3>
<p><a href="https://img.atcoder.jp/ahc009/first.zip">Download 1</a></p>
<p><a href="https://img.atcoder.jp/ahc009/second.zip">Download 2</a></p>
"""

HTML_VISUALIZER_LINK_PATTERN_FALLBACK_HERE = HTML_VISUALIZER_LINK_HERE  # Reuse existing

HTML_NO_CONTEST_ID_IN_URL_BUT_KEYWORD = """ # For testing fallback when contest_id cannot be derived
<html>
<body>
  <div id='task-statement'>
    <h3>ツール</h3>
    <p>入力ジェネレータとビジュアライザは
    <a href="https://otherdomain.com/ahc010/tools.zip">ここ</a>
    からダウンロード出来ます。</p>
  </div>
</body>
</html>
"""


def test_fetch_problem_statement_visualizer_link_pattern_single(mock_requests_get):
    html_content = HTML_TEMPLATE_TASK_STATEMENT.format(problem_html=HTML_VISUALIZER_LINK_PATTERN_SINGLE)
    mock_requests_get.return_value = create_mock_response(html_content)
    md_content, _, visualizer_zip_url = fetch_problem_statement("https://atcoder.jp/contests/ahc007/tasks/ahc007_a")
    assert visualizer_zip_url == "https://img.atcoder.jp/ahc007/tools.zip"


def test_fetch_problem_statement_visualizer_link_pattern_multiple_keyword(mock_requests_get):
    html_content = HTML_TEMPLATE_TASK_STATEMENT.format(problem_html=HTML_VISUALIZER_LINK_PATTERN_MULTIPLE_KEYWORD)
    mock_requests_get.return_value = create_mock_response(html_content)
    md_content, _, visualizer_zip_url = fetch_problem_statement("https://atcoder.jp/contests/ahc008/tasks/ahc008_a")
    assert visualizer_zip_url == "https://img.atcoder.jp/ahc008/tools_local.zip"


def test_fetch_problem_statement_visualizer_link_pattern_multiple_no_keyword(mock_requests_get):
    html_content = HTML_TEMPLATE_TASK_STATEMENT.format(problem_html=HTML_VISUALIZER_LINK_PATTERN_MULTIPLE_NO_KEYWORD)
    mock_requests_get.return_value = create_mock_response(html_content)
    md_content, _, visualizer_zip_url = fetch_problem_statement("https://atcoder.jp/contests/ahc009/tasks/ahc009_a")
    assert visualizer_zip_url == "https://img.atcoder.jp/ahc009/first.zip"  # Should pick the first one


def test_fetch_problem_statement_visualizer_link_pattern_fallback_to_here(mock_requests_get):
    # This HTML does not contain a link matching img.atcoder.jp/{contest_id}/*.zip pattern
    # So it should fallback to 'ここ' keyword search.
    problem_html_no_pattern_match = """
    <h3>ツール</h3>
    <p>ビジュアライザは <a href="https://someotherplace.com/tool.zip">ここ</a> から。</p>
    """
    html_content = HTML_TEMPLATE_TASK_STATEMENT.format(problem_html=problem_html_no_pattern_match)
    mock_requests_get.return_value = create_mock_response(html_content)
    md_content, _, visualizer_zip_url = fetch_problem_statement("https://atcoder.jp/contests/ahc011/tasks/ahc011_a")
    assert visualizer_zip_url == "https://someotherplace.com/tool.zip"


def test_fetch_problem_statement_no_contest_id_fallback_to_keyword(mock_requests_get):
    # Use a URL from which contest_id cannot be reliably derived by current logic
    # This forces the URL pattern part to be skipped, testing keyword fallback directly.
    mock_requests_get.return_value = create_mock_response(HTML_NO_CONTEST_ID_IN_URL_BUT_KEYWORD)
    # Malformed URL for testing
    md_content, _, visualizer_zip_url = fetch_problem_statement("https://atcoder.jp/contests/tasks/ahc010_a")
    assert visualizer_zip_url == "https://otherdomain.com/ahc010/tools.zip"
