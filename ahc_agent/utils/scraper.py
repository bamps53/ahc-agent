import os
import re
import shutil
from typing import Optional
from urllib.parse import urljoin
import zipfile

from bs4 import BeautifulSoup
import html2text
import requests


def fetch_problem_statement(url: Optional[str] = None, html_content: Optional[str] = None):  # urlをOptionalに変更し、html_content引数を追加
    """Fetches and parses the problem statement from an AtCoder task URL.

    Args:
        url (str, optional): The URL of the AtCoder task page.
        html_content (str, optional): The HTML content of the problem statement.
                                      If provided, `url` is used for context (like visualizer URL) but not fetched.
    """
    soup = None
    if html_content:
        soup = BeautifulSoup(html_content, "html.parser")
    elif url:
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "html.parser")
        except requests.exceptions.RequestException as e:
            print(f"Error fetching URL {url}: {e}")
            return None, None
    else:
        print("Error: Either url or html_content must be provided.")
        return None, None

    if not soup:  # soupがNoneのままの場合 (エラーケース)
        return None, None
    task_statement_div = soup.find(id="task-statement")

    if not task_statement_div:
        # ローカルHTMLの場合、task-statement divがない可能性も考慮
        # 代替としてbody全体を使うか、エラーとするか。ここではエラーとする。
        print("Task statement section (div with id='task-statement') not found.")
        return None, None

    html_for_md = str(task_statement_div)
    md_content = html2text.html2text(html_for_md)

    contest_id_from_url = None  # URLがない場合はNoneのまま

    if url:  # URLがある場合のみビジュアライザURL抽出を試みる
        try:
            parts = url.split("/")
            if len(parts) >= 5 and parts[-2] == "tasks":
                contest_id_from_url = parts[-3]
            elif len(parts) >= 4:
                if parts[-2] == "contests":
                    contest_id_from_url = parts[-1]
                elif len(parts) >= 3 and parts[-3] == "contests":
                    contest_id_from_url = parts[-2]
        except IndexError:
            print(f"Warning: Could not determine contest ID from URL for visualizer search: {url}.")

    visualizer_zip_url = None
    # URLがあり、かつ contest_id_from_url が特定できた場合のみビジュアライザを探す
    if url and contest_id_from_url:
        pattern_base_url = f"https://img.atcoder.jp/{contest_id_from_url}/"
        zip_links = soup.find_all("a", href=re.compile(rf"^{re.escape(pattern_base_url)}.*\.zip$"))

        if zip_links:
            if len(zip_links) == 1:
                visualizer_zip_url = zip_links[0]["href"]
            else:
                keywords = ["ローカル", "local", "ツール", "tool", "visualizer", "ビジュアライザ"]
                best_link = None
                for link in zip_links:
                    link_text = link.get_text(separator=" ", strip=True).lower()
                    context_text = "".join(link.find_parent().stripped_strings).lower() if link.find_parent() else ""
                    if any(keyword in link_text or keyword in context_text for keyword in keywords):
                        best_link = link["href"]
                        break
                visualizer_zip_url = best_link if best_link else zip_links[0]["href"]

        # 相対パスのビジュアライザリンクも探す (例: tools.zip, visualizer.zip)
        # HTMLコンテンツから直接取得する場合、ベースURLが必要になる
        if not visualizer_zip_url and url:  # urlが存在する場合のみurljoinを試みる
            relative_zip_links = soup.find_all("a", href=re.compile(r".*\.zip$"))
            for link in relative_zip_links:
                href = link["href"]
                link_text = link.get_text(separator=" ", strip=True).lower()
                context_text = "".join(link.find_parent().stripped_strings).lower() if link.find_parent() else ""
                keywords = ["tool", "visualizer", "local", "ツール", "ビジュアライザ", "ローカル"]
                if any(keyword in link_text or keyword in context_text for keyword in keywords):
                    # urljoinで絶対URLに変換
                    visualizer_zip_url = urljoin(url, href)
                    break

    # 2. Fallback to keyword-based search if URL pattern search fails
    if not visualizer_zip_url:
        # Try to find the visualizer link with "ここ" (here)
        visualizer_link_keyword = "ここ"
        local_tool_link = soup.find("a", string=visualizer_link_keyword)

        # Fallback to "ローカル版" or "Local version" if "ここ" is not found
        if not local_tool_link:
            local_tool_link = soup.find("a", string=re.compile(r"ローカル版|Local version"))

        if local_tool_link and local_tool_link.has_attr("href"):
            raw_zip_url = local_tool_link["href"]
            # Ensure the URL is absolute
            visualizer_zip_url = urljoin(url, raw_zip_url) if not raw_zip_url.startswith("http") else raw_zip_url

    if visualizer_zip_url:
        print(f"Found visualizer download link: {visualizer_zip_url}")
    else:
        print("Could not find visualizer download link.")

    return md_content, visualizer_zip_url


def download_and_extract_visualizer(zip_url, target_tools_dir):
    """Downloads and extracts the visualizer zip file to target_tools_dir."""
    if not zip_url:  # zip_urlがNoneなら何もしない
        print("No visualizer URL provided, skipping download.")
        return False
    print(f"Attempting to download visualizer from: {zip_url}")
    try:
        print(f"Downloading visualizer from {zip_url}...")
        response = requests.get(zip_url, stream=True, timeout=30)
        response.raise_for_status()

        temp_zip_path = "temp_visualizer.zip"
        with open(temp_zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download complete.")

        if os.path.exists(target_tools_dir):
            print(f"Removing existing tools directory: {target_tools_dir}")
            shutil.rmtree(target_tools_dir)
        os.makedirs(target_tools_dir, exist_ok=True)

        print(f"Extracting {temp_zip_path} to {target_tools_dir}/")
        with zipfile.ZipFile(temp_zip_path, "r") as zip_ref:
            top_level_dirs = {item.split("/")[0] for item in zip_ref.namelist() if "/" in item and item.split("/")[0] != ""}

            if len(top_level_dirs) == 1:
                single_top_dir = top_level_dirs.pop()
                is_single_dir_archive = True
                for name in zip_ref.namelist():
                    if not name.startswith(single_top_dir + "/") and name != single_top_dir + "/":
                        if name.startswith(".") and single_top_dir in zip_ref.namelist():
                            continue
                        if name == single_top_dir:
                            continue
                        is_single_dir_archive = False
                        break

                if is_single_dir_archive:
                    temp_extract_dir = "temp_extract_dir_for_visualizer_stripping"
                    if os.path.exists(temp_extract_dir):
                        shutil.rmtree(temp_extract_dir)
                    os.makedirs(temp_extract_dir)
                    zip_ref.extractall(temp_extract_dir)

                    extracted_folder_path = os.path.join(temp_extract_dir, single_top_dir)
                    if os.path.isdir(extracted_folder_path):
                        for item_name in os.listdir(extracted_folder_path):
                            s = os.path.join(extracted_folder_path, item_name)
                            d = os.path.join(target_tools_dir, item_name)
                            if os.path.isdir(s):
                                shutil.copytree(s, d, dirs_exist_ok=True)
                            else:
                                shutil.copy2(s, d)
                        print(f"Successfully extracted contents of '{single_top_dir}' to {target_tools_dir}/")
                    else:
                        zip_ref.extractall(target_tools_dir)
                        print(f"Successfully extracted to {target_tools_dir}/ (single top item was not a directory)")
                    shutil.rmtree(temp_extract_dir)
                else:
                    zip_ref.extractall(target_tools_dir)
                    print(f"Successfully extracted to {target_tools_dir}/ (archive has multiple items at root or complex structure)")
            else:
                zip_ref.extractall(target_tools_dir)
                print(f"Successfully extracted to {target_tools_dir}/ (no single top-level directory or multiple items at root)")

        os.remove(temp_zip_path)
        print(f"Visualizer setup complete in {target_tools_dir}.")
        return True

    except requests.exceptions.RequestException as e:
        print(f"Error downloading visualizer: {e}")
    except zipfile.BadZipFile:
        print("Error: Downloaded file is not a valid zip file.")
    except OSError as e:
        print(f"Error during file operations: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during visualizer setup: {e}")
    return False


def scrape_and_setup_problem(
    url: Optional[str] = None, base_output_dir=".", html_file_path: Optional[str] = None, contest_id_for_filename: Optional[str] = None
):
    """Fetches problem (from URL or local file), downloads visualizer, and saves them to structured directories."""
    md_content, filename, visualizer_zip_url = None, "problem.md", None

    actual_url_for_visualizer_context = url  # ビジュアライザの相対パス解決用

    if html_file_path:
        try:
            with open(html_file_path, encoding="utf-8") as f:
                html_content = f.read()
            # ローカルHTMLの場合、URLはオプションだがビジュアライザの相対パス解決のためにあると良い
            md_content, visualizer_zip_url_from_html = fetch_problem_statement(url=actual_url_for_visualizer_context, html_content=html_content)
            if visualizer_zip_url_from_html:  # HTML内から見つかったものを優先
                visualizer_zip_url = visualizer_zip_url_from_html
            # filename_suggestion は "problem.md" で固定

        except FileNotFoundError:
            print(f"Error: HTML file not found at {html_file_path}")
            return False
        except Exception as e:
            print(f"Error reading or parsing HTML file {html_file_path}: {e}")
            return False
    elif url:
        md_content, visualizer_zip_url = fetch_problem_statement(url=url)
    else:
        print("Error: Either url or html_file_path must be provided to scrape_and_setup_problem.")
        return False

    if md_content is None:
        return False

    problem_specific_dir = base_output_dir  # initコマンドではbase_output_dirがプロジェクトルートになる想定

    try:
        os.makedirs(problem_specific_dir, exist_ok=True)
        output_path = os.path.join(problem_specific_dir, filename)  # 提案されたファイル名を使用
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(md_content)
        print(f"Problem statement saved to {output_path}")

        target_tools_dir = os.path.join(problem_specific_dir, "tools")
        os.makedirs(target_tools_dir, exist_ok=True)  # toolsディレクトリは常に作成

        if visualizer_zip_url:
            resolved_visualizer_zip_url = visualizer_zip_url
            # URLが相対パスであり、かつ http/https で始まらない場合
            if not visualizer_zip_url.startswith(("http://", "https://")):
                if actual_url_for_visualizer_context:  # ベースURLがある場合
                    resolved_visualizer_zip_url = urljoin(actual_url_for_visualizer_context, visualizer_zip_url)
                elif contest_id_for_filename:  # ベースURLがなく、コンテストIDがある場合
                    img_base_url = f"https://img.atcoder.jp/{contest_id_for_filename}/"
                    resolved_visualizer_zip_url = urljoin(img_base_url, visualizer_zip_url)
                else:
                    print(f"Warning: Could not fully resolve relative visualizer URL: {visualizer_zip_url}. No base URL or contest ID provided.")

            if not download_and_extract_visualizer(resolved_visualizer_zip_url, target_tools_dir):
                print("Visualizer download/extraction failed or skipped. Continuing without visualizer.")
        else:
            print("No visualizer URL found or provided. Skipping visualizer download.")

        target_tools_in_dir = os.path.join(target_tools_dir, "in")
        os.makedirs(target_tools_in_dir, exist_ok=True)
        print(f"Created directory for test cases: {target_tools_in_dir}")

        return True

    except OSError as e:
        print(f"Error writing file or creating directories: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during setup: {e}")
    return False
