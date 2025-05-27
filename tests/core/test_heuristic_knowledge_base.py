import os
import pytest
from pathlib import Path

from ahc_agent.core.heuristic_knowledge_base import HeuristicKnowledgeBase, AlgorithmInfo, PastContestSolutionInfo
from ahc_agent.utils.file_io import read_json, read_file # For verification

@pytest.fixture
def hkb_base_dir(tmp_path: Path) -> Path:
    kb_dir = tmp_path / "heuristic_kb"
    kb_dir.mkdir()
    return kb_dir

@pytest.fixture
def hkb(hkb_base_dir: Path) -> HeuristicKnowledgeBase:
    return HeuristicKnowledgeBase(str(hkb_base_dir))

# --- Algorithm Library Tests ---
def test_hkb_add_get_algorithm(hkb: HeuristicKnowledgeBase):
    code_content = "void dijkstra() {}"
    meta_content = {
        "name": "Dijkstra Algorithm",
        "description": "Shortest path",
        "tags": ["graph", "shortest_path"],
        "filename": "dijkstra.cpp" # Recommended to include for clarity
    }
    
    add_success = hkb.add_algorithm("dijkstra.cpp", "dijkstra.meta.json", code_content, meta_content)
    assert add_success
    
    algo_info = hkb.get_algorithm("dijkstra") # Search by basename
    assert algo_info is not None
    assert isinstance(algo_info, AlgorithmInfo)
    assert algo_info.name == "Dijkstra Algorithm"
    assert algo_info.description == "Shortest path"
    assert algo_info.code == code_content
    assert algo_info.metadata["tags"] == ["graph", "shortest_path"]
    assert (Path(hkb.library_dir) / "dijkstra.cpp").exists()
    assert (Path(hkb.library_dir) / "dijkstra.meta.json").exists()

def test_hkb_list_search_algorithms(hkb: HeuristicKnowledgeBase):
    hkb.add_algorithm("algo1.cpp", "algo1.meta.json", "code1", {"name": "Algo One", "description": "First algorithm", "tags": ["tagA", "common"]})
    hkb.add_algorithm("algo2.py", "algo2.meta.json", "code2", {"name": "Algo Two", "description": "Second algorithm", "tags": ["tagB", "common"]})
    hkb.add_algorithm("complex_algo.cpp", "complex_algo.meta.json", "code3", {"name": "Complex Search", "description": "A complex item for search", "tags": ["tagA"]})

    all_algos = hkb.list_algorithms()
    assert len(all_algos) == 3
    
    common_tag_algos = hkb.list_algorithms(tag="common")
    assert len(common_tag_algos) == 2
    retrieved_names = sorted([a.name for a in common_tag_algos])
    assert retrieved_names == ["Algo One", "Algo Two"]

    tag_a_algos = hkb.list_algorithms(tag="tagA")
    assert len(tag_a_algos) == 2
    
    search_one = hkb.search_algorithms("One")
    assert len(search_one) == 1
    assert search_one[0].name == "Algo One"
    
    search_algo = hkb.search_algorithms("algorithm") # Matches description
    assert len(search_algo) == 2
    
    search_complex = hkb.search_algorithms("complex") # Matches name and description
    assert len(search_complex) == 1
    assert search_complex[0].name == "Complex Search"

    search_tag_b = hkb.search_algorithms("tagB") # Matches tag
    assert len(search_tag_b) == 1
    assert search_tag_b[0].name == "Algo Two"


# --- Past Contest DB Tests ---
def test_hkb_add_get_past_contest(hkb: HeuristicKnowledgeBase):
    contest_id = "ahc001"
    data = {
        "title": "AHC001 Title", 
        "problem_summary": "Summary of AHC001", 
        "tags": ["optimization", "annealing"],
        "solution_approaches": [{"algorithms_used": ["Simulated Annealing"]}]
    }
    add_success = hkb.add_past_contest_solution(contest_id, data)
    assert add_success
    
    contest_info = hkb.get_past_contest_solution(contest_id)
    assert contest_info is not None
    assert isinstance(contest_info, PastContestSolutionInfo)
    assert contest_info.title == "AHC001 Title"
    assert contest_info.tags == ["optimization", "annealing"]
    assert (Path(hkb.past_contests_db_dir) / f"{contest_id}.json").exists()

def test_hkb_list_search_past_contests(hkb: HeuristicKnowledgeBase):
    hkb.add_past_contest_solution("ahc001", {"title": "Contest Alpha", "tags": ["tagX", "common_contest"], "solution_approaches": [{"algorithms_used": ["Greedy"], "key_ideas": ["fast sort"]}]})
    hkb.add_past_contest_solution("ahc002", {"title": "Contest Beta", "tags": ["tagY", "common_contest"], "solution_approaches": [{"algorithms_used": ["Beam Search"], "key_ideas": ["pruning strategy"]}]})
    hkb.add_past_contest_solution("xyz003", {"title": "Contest Gamma", "tags": ["tagX"], "solution_approaches": [{"algorithms_used": ["DP", "Greedy"], "key_ideas": ["state compression"]}]})

    all_contests = hkb.list_past_contest_solutions()
    assert len(all_contests) == 3
    
    common_tag_contests = hkb.list_past_contest_solutions(tag="common_contest")
    assert len(common_tag_contests) == 2
    
    search_alpha = hkb.search_past_contests(keyword="Alpha")
    assert len(search_alpha) == 1
    assert search_alpha[0].contest_id == "ahc001"

    search_greedy_algo = hkb.search_past_contests(algorithm_used="Greedy")
    assert len(search_greedy_algo) == 2 # ahc001 and xyz003
    
    search_tag_y = hkb.search_past_contests(tag="tagY")
    assert len(search_tag_y) == 1
    assert search_tag_y[0].contest_id == "ahc002"

    search_state_idea = hkb.search_past_contests(keyword="state compression")
    assert len(search_state_idea) == 1
    assert search_state_idea[0].contest_id == "xyz003"

# --- Generic Data Tests ---
def test_hkb_generic_json_data(hkb: HeuristicKnowledgeBase):
    key = "test_data.json"
    data = {"value": 123, "message": "hello"}
    
    assert hkb.get_generic_json_data(key) is None
    save_success = hkb.save_generic_json_data(key, data)
    assert save_success
    
    retrieved_data = hkb.get_generic_json_data(key)
    assert retrieved_data == data
    assert (Path(hkb.base_dir) / key).exists()

def test_hkb_generic_file_data(hkb: HeuristicKnowledgeBase):
    filename = "notes.txt"
    content = "This is a note."
    
    assert hkb.get_generic_file_data(filename) is None
    save_success = hkb.save_generic_file_data(filename, content)
    assert save_success
    
    retrieved_content = hkb.get_generic_file_data(filename)
    assert retrieved_content == content
    assert (Path(hkb.base_dir) / filename).exists()

# --- Problem Level Data Tests ---
def test_hkb_problem_level_data(hkb: HeuristicKnowledgeBase):
    subdir = "problem_experiments"
    data_id = "exp001"
    data = {"param": 0.5, "score": 9876}

    assert hkb.get_problem_level_data(subdir, data_id) is None
    assert len(hkb.list_problem_level_data_ids(subdir)) == 0
    
    save_success = hkb.save_problem_level_data(subdir, data_id, data)
    assert save_success
    
    retrieved_data = hkb.get_problem_level_data(subdir, data_id)
    assert retrieved_data == data
    assert (Path(hkb.base_dir) / subdir / f"{data_id}.json").exists()

    data_id2 = "exp002"
    data2 = {"param": 0.9, "score": 12345}
    hkb.save_problem_level_data(subdir, data_id2, data2)

    ids = hkb.list_problem_level_data_ids(subdir)
    assert len(ids) == 2
    assert sorted(ids) == sorted([data_id, data_id2])
