import os
import time
import pytest
from pathlib import Path

from ahc_agent.core.session_store import SessionManager, SessionStore
from ahc_agent.utils.file_io import read_json # For verification

# Helper function to create a dummy workspace structure
def create_workspace_problem_dir(tmp_path: Path, problem_id: str) -> Path:
    problem_dir = tmp_path / problem_id
    problem_dir.mkdir(parents=True, exist_ok=True)
    # SessionManager expects workspace_dir (parent of problem_dir) and problem_id
    return problem_dir 

@pytest.fixture
def workspace_root(tmp_path: Path) -> Path:
    # This is the directory containing multiple problem directories
    ws_root = tmp_path / "test_workspace"
    ws_root.mkdir(exist_ok=True)
    return ws_root

@pytest.fixture
def problem_id() -> str:
    return "ahc999"

@pytest.fixture
def session_manager(workspace_root: Path, problem_id: str) -> SessionManager:
    # Ensure the problem directory itself exists for context, though SessionManager uses parent
    (workspace_root / problem_id).mkdir(exist_ok=True)
    return SessionManager(str(workspace_root), problem_id)

def test_session_manager_create_session(session_manager: SessionManager, workspace_root: Path, problem_id: str):
    initial_meta = {"user": "test_user"}
    session_store = session_manager.create_session(initial_metadata=initial_meta)
    assert session_store is not None
    assert isinstance(session_store, SessionStore)
    
    session_id = session_store.session_id
    assert session_id is not None
    
    expected_session_dir = workspace_root / problem_id / "knowledge" / "sessions" / session_id
    assert expected_session_dir.exists()
    
    metadata_path = expected_session_dir / "metadata.json"
    assert metadata_path.exists()
    
    metadata = read_json(str(metadata_path))
    assert metadata["session_id"] == session_id
    assert metadata["problem_id"] == problem_id
    assert metadata["user"] == "test_user"
    assert "created_at" in metadata
    assert metadata["status"] == "created" # Default status from SessionStore.create_session_metadata

def test_session_manager_get_existing_session(session_manager: SessionManager):
    session_store_created = session_manager.create_session()
    session_id_created = session_store_created.session_id
    
    session_store_retrieved = session_manager.get_session_store(session_id_created)
    assert session_store_retrieved is not None
    assert session_store_retrieved.session_id == session_id_created
    assert session_store_retrieved.problem_id == session_manager.problem_id

def test_session_manager_get_non_existent_session(session_manager: SessionManager):
    store = session_manager.get_session_store("non_existent_session_id")
    assert store is None

def test_session_manager_list_sessions(session_manager: SessionManager):
    # Listing sessions relies on subdirectories being present.
    # If create_session doesn't create its dir immediately or if metadata.json is key,
    # this test might need adjustment based on SessionManager's list_sessions implementation.
    # Assuming list_sessions finds sessions with valid metadata.json.
    
    # Clean up any pre-existing sessions in the test problem dir to ensure clean test
    sessions_base_dir = Path(session_manager.sessions_base_dir)
    if sessions_base_dir.exists():
        import shutil
        for item in sessions_base_dir.iterdir():
            shutil.rmtree(item)

    sessions_before = session_manager.list_sessions()
    assert len(sessions_before) == 0
    
    s1_store = session_manager.create_session({"name": "session1", "custom_field": "s1"})
    time.sleep(0.01) # Ensure different timestamps for creation order if not sorting by name
    s2_store = session_manager.create_session({"name": "session2", "custom_field": "s2"})
    
    sessions_after = session_manager.list_sessions()
    assert len(sessions_after) == 2
    
    retrieved_ids = sorted([s["session_id"] for s in sessions_after])
    expected_ids = sorted([s1_store.session_id, s2_store.session_id])
    assert retrieved_ids == expected_ids
    
    # Check if custom fields from initial_metadata are in listed metadata
    custom_fields = {}
    for s_meta in sessions_after:
        if "name" in s_meta: # Ensure name is present before trying to access it
             custom_fields[s_meta["name"]] = s_meta.get("custom_field")

    assert custom_fields.get("session1") == "s1"
    assert custom_fields.get("session2") == "s2"


@pytest.fixture
def active_session_store(session_manager: SessionManager) -> SessionStore:
    # Create a session and return its store for further tests
    # Pass initial metadata directly to create_session.
    # create_session_metadata is called internally by SessionManager.create_session.
    return session_manager.create_session(initial_metadata={"status_fixture": "fixture_created_in_manager"})


def test_session_store_get_update_metadata(active_session_store: SessionStore):
    metadata = active_session_store.get_session_metadata()
    assert metadata is not None
    # Check the initial status set by the fixture via SessionManager.create_session
    assert metadata["status_fixture"] == "fixture_created_in_manager" 
    
    update_success = active_session_store.update_session_metadata({"status": "running", "progress": 50})
    assert update_success
    
    updated_metadata = active_session_store.get_session_metadata()
    assert updated_metadata["status"] == "running"
    assert updated_metadata["progress"] == 50
    assert updated_metadata["status_fixture"] == "fixture_created_in_manager" # Original field should persist
    assert updated_metadata["updated_at"] > metadata["updated_at"]

def test_session_store_problem_analysis(active_session_store: SessionStore):
    analysis_data = {"type": "graph", "constraints": "N < 100"}
    assert active_session_store.get_problem_analysis() is None # Initially
    
    save_success = active_session_store.save_problem_analysis(analysis_data)
    assert save_success
    
    retrieved_analysis = active_session_store.get_problem_analysis()
    assert retrieved_analysis == analysis_data
    
    metadata = active_session_store.get_session_metadata()
    assert metadata["has_problem_analysis"] is True
    assert metadata["status"] == "analysis_complete"

def test_session_store_solution_strategy(active_session_store: SessionStore):
    strategy_data = {"approach": "greedy", "algorithm": "dijkstra"}
    assert active_session_store.get_solution_strategy() is None # Initially

    save_success = active_session_store.save_solution_strategy(strategy_data)
    assert save_success
    
    retrieved_strategy = active_session_store.get_solution_strategy()
    assert retrieved_strategy == strategy_data

    metadata = active_session_store.get_session_metadata()
    assert metadata["has_solution_strategy"] is True
    assert metadata["status"] == "strategy_complete"

def test_session_store_solutions(active_session_store: SessionStore):
    assert len(active_session_store.list_solutions()) == 0
    assert active_session_store.get_best_solution() is None

    sol1_data = {"solution_id": "sol1", "score": 100, "generation": 1} # Changed "id" to "solution_id" for consistency
    sol1_code = "int main() { return 1; }"
    save_s1 = active_session_store.save_solution("sol1", sol1_data, sol1_code)
    assert save_s1

    retrieved_s1 = active_session_store.get_solution("sol1")
    assert retrieved_s1 is not None
    assert retrieved_s1["score"] == 100
    assert retrieved_s1["code"] == sol1_code
    assert "code_filename" in retrieved_s1 

    sol2_data_no_code = {"solution_id": "sol2", "score": 200, "generation": 2, "code": "direct code in json"}
    save_s2 = active_session_store.save_solution("sol2", sol2_data_no_code) 
    assert save_s2
    
    retrieved_s2 = active_session_store.get_solution("sol2")
    assert retrieved_s2 is not None
    assert retrieved_s2["score"] == 200
    assert retrieved_s2["code"] == "direct code in json" 

    sol3_data_no_score = {"solution_id": "sol3", "generation": 3}
    save_s3 = active_session_store.save_solution("sol3", sol3_data_no_score, "code for sol3")
    assert save_s3

    all_solutions = active_session_store.list_solutions()
    assert len(all_solutions) == 3
    
    best_solution = active_session_store.get_best_solution()
    assert best_solution is not None
    # Ensure the "solution_id" field from the original data is preserved, not the key used in save_solution if different
    assert best_solution["solution_id"] == "sol2" 

    assert active_session_store.get_solution("non_existent_sol") is None
    
    metadata = active_session_store.get_session_metadata()
    assert metadata["last_solution_id"] == "sol3" 
    assert metadata["status"] == "solution_saved"


def test_session_store_evolution_log(active_session_store: SessionStore):
    log_data = {"generations": 5, "best_score_history": [10, 20, 30, 30, 40]}
    assert active_session_store.get_evolution_log() is None 

    save_success = active_session_store.save_evolution_log(log_data)
    assert save_success
    
    retrieved_log = active_session_store.get_evolution_log()
    assert retrieved_log == log_data

    metadata = active_session_store.get_session_metadata()
    assert metadata["has_evolution_log"] is True
    assert metadata["status"] == "evolution_logged"

def test_session_store_llm_interactions(active_session_store: SessionStore):
    assert len(active_session_store.list_llm_interactions()) == 0

    interaction1_data = {"interaction_id": "interaction1", "prompt": "Analyze this", "response": "It is complex."}
    save_i1 = active_session_store.save_llm_interaction("interaction1", interaction1_data)
    assert save_i1

    retrieved_i1 = active_session_store.get_llm_interaction("interaction1")
    assert retrieved_i1 == interaction1_data

    interaction2_data = {"interaction_id": "interaction2", "prompt": "Summarize that", "response": "It is short."}
    save_i2 = active_session_store.save_llm_interaction("interaction2", interaction2_data)
    assert save_i2
    
    all_interactions = active_session_store.list_llm_interactions()
    assert len(all_interactions) == 2
    
    # Check if interaction_id from data is preserved.
    retrieved_interaction_ids_from_data = sorted([d.get("interaction_id") for d in all_interactions])
    expected_interaction_ids = sorted([interaction1_data["interaction_id"], interaction2_data["interaction_id"]])
    assert retrieved_interaction_ids_from_data == expected_interaction_ids

    assert active_session_store.get_llm_interaction("non_existent_interaction") is None
