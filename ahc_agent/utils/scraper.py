import argparse
import os
import re
import shutil
from urllib.parse import urljoin
import zipfile

from bs4 import BeautifulSoup
import html2text
import requests


def fetch_problem_statement(url):
    """Fetches and parses the problem statement from an AtCoder task URL.

    Args:
        url (str): The URL of the AtCoder task page.

    Returns:
        tuple: A tuple containing md_content (str), filename_suggestion (str),
               visualizer_zip_url (str or None), or (None, None, None) if an error occurs.
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL {url}: {e}")
        return None, None, None

    soup = BeautifulSoup(response.content, "html.parser")
    task_statement_div = soup.find(id="task-statement")

    if not task_statement_div:
        print(f"Task statement section not found on page: {url}")
        return None, None, None

    html_content = str(task_statement_div)
    md_content = html2text.html2text(html_content)

    filename_suggestion = "problem_statement.md"
    contest_id_from_url = None
    try:
        parts = url.split("/")
        if len(parts) >= 5 and parts[-2] == "tasks":
            contest_id_from_url = parts[-3]
            task_id_full = parts[-1]
            task_id = task_id_full.split("_")[-1] if "_" in task_id_full else task_id_full
            filename_suggestion = f"{contest_id_from_url}_{task_id}_problem.md"
        elif len(parts) >= 4:
            # Fallback for URLs not strictly matching /contests/.../tasks/...
            # This might occur for direct task links or other formats.
            # Try to infer contest_id if possible, otherwise filename_suggestion remains default.
            if parts[-2] == "contests":  # e.g. /contests/ahc001
                contest_id_from_url = parts[-1]
                filename_suggestion = f"{contest_id_from_url}_problem.md"
            elif len(parts) >= 3 and parts[-3] == "contests":  # e.g. /contests/ahc001/tasks
                contest_id_from_url = parts[-2]
                filename_suggestion = f"{contest_id_from_url}_problem.md"
            # else: filename_suggestion remains default

    except IndexError:
        print(f"Warning: Could not determine contest/task ID from URL for filename: {url}.")
        # Keep default filename_suggestion if pattern doesn't match

    visualizer_zip_url = None

    # 1. Try to find by URL pattern: https://img.atcoder.jp/{contest_id}/*.zip
    if contest_id_from_url:
        pattern_base_url = f"https://img.atcoder.jp/{contest_id_from_url}/"
        zip_links = soup.find_all("a", href=re.compile(rf"^{re.escape(pattern_base_url)}.*\.zip$"))

        if zip_links:
            if len(zip_links) == 1:
                visualizer_zip_url = zip_links[0]["href"]
            else:
                # Prioritize links with keywords if multiple zip files are found
                keywords = ["ローカル", "local", "ツール", "tool", "visualizer", "ビジュアライザ"]
                best_link = None
                for link in zip_links:
                    link_text = link.get_text(separator=" ", strip=True).lower()
                    # Also check parent or surrounding text for context if link text is generic (e.g. "here")
                    # This is a simple check, could be made more sophisticated
                    context_text = "".join(link.find_parent().stripped_strings).lower() if link.find_parent() else ""

                    if any(keyword in link_text or keyword in context_text for keyword in keywords):
                        best_link = link["href"]
                        break  # Take the first keyword match
                if best_link:
                    visualizer_zip_url = best_link
                else:
                    # For now, just take the first one found by the pattern.
                    visualizer_zip_url = zip_links[0]["href"]
                    print(f"Multiple .zip links for {contest_id_from_url} found. Selected: {visualizer_zip_url}")

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

    return md_content, filename_suggestion, visualizer_zip_url


def download_and_extract_visualizer(zip_url, target_tools_dir):
    """Downloads and extracts the visualizer zip file to target_tools_dir."""
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
                    print(
                        f"Successfully extracted to {target_tools_dir}/ "
                        "(archive has multiple items at root or complex structure)"
                    )
            else:
                zip_ref.extractall(target_tools_dir)
                print(
                    f"Successfully extracted to {target_tools_dir}/ (no single top-level directory or multiple items at root)"
                )

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


def scrape_and_setup_problem(url, base_output_dir="."):
    """Fetches problem, downloads visualizer, and saves them to structured directories."""
    md_content, _, visualizer_zip_url = fetch_problem_statement(url)

    if md_content is None:
        return False  # Error in fetching problem statement

    # Determine the problem-specific directory based on the URL
    # For init command, base_output_dir will be the workspace_dir itself.
    problem_specific_dir = base_output_dir

    try:
        os.makedirs(problem_specific_dir, exist_ok=True)
        output_path = os.path.join(problem_specific_dir, "problem.md")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(md_content)
        print(f"Problem statement saved to {output_path}")

        if visualizer_zip_url:
            target_tools_dir = os.path.join(problem_specific_dir, "tools")
            if not download_and_extract_visualizer(visualizer_zip_url, target_tools_dir):
                print("Visualizer download/extraction failed. Continuing without visualizer.")

        return True

    except OSError as e:
        print(f"Error writing file or creating directories: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during setup: {e}")
    return False


def main():
    parser = argparse.ArgumentParser(description="Scrape task statement and download visualizer from an AtCoder contest page.")
    parser.add_argument("url", help="The URL of the AtCoder task page")
    args = parser.parse_args()

    if scrape_and_setup_problem(args.url, "."):
        print("Problem setup completed successfully.")
    else:
        print("Problem setup failed.")


if __name__ == "__main__":
    main()
