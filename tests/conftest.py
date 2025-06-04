import os

import pytest

from ahc_agent.utils.docker_manager import DockerManager


@pytest.fixture
def docker_manager():
    """Provides a DockerManager instance for tests."""
    # DockerManagerの初期化パラメータはテストの要件に応じて調整してください
    config = {
        "image": "test-image",
        "enabled": True,
        "timeout": 30,
        "build_timeout": 120,
        # 他に必要な設定があればここに追加
    }
    return DockerManager(config=config)


@pytest.fixture
def temp_workspace(tmp_path):
    """Provides a temporary workspace directory for tests using pytest's tmp_path."""
    # tmp_path is a pytest fixture that provides a Path object to a temporary directory
    return str(tmp_path)


@pytest.fixture(scope="session", autouse=True)
def set_test_mode_temp_workspace(tmp_path_factory):
    print("\n[CONTEST_SETUP] Running set_test_mode_temp_workspace fixture setup...")  # DEBUG PRINT
    """
    Sets the AHCAGENT_TEST_MODE_TEMP_WORKSPACE environment variable for the entire test session
    if AHCAGENT_LLM_LOGGING_DISABLED is not 'true'.
    This directory will be used by LLMClient for logs if no other workspace is specified
    and logging is not disabled.
    """
    original_value = os.getenv("AHCAGENT_TEST_MODE_TEMP_WORKSPACE")
    original_logging_disabled_env = os.getenv("AHCAGENT_LLM_LOGGING_DISABLED")
    print(f"[CONTEST_SETUP] Initial AHCAGENT_TEST_MODE_TEMP_WORKSPACE: {original_value}")  # DEBUG PRINT
    print(f"[CONTEST_SETUP] Initial AHCAGENT_LLM_LOGGING_DISABLED: {original_logging_disabled_env}")  # DEBUG PRINT

    # Determine if logging is effectively disabled for the session based on env var
    # This check is done once at the start of the session.
    logging_globally_disabled = os.getenv("AHCAGENT_LLM_LOGGING_DISABLED", "false").lower() == "true"
    print(f"[CONTEST_SETUP] Evaluated logging_globally_disabled: {logging_globally_disabled}")  # DEBUG PRINT

    temp_dir_path_str = None
    if not logging_globally_disabled:
        # Create a session-scoped temporary directory
        session_tmp_dir = tmp_path_factory.mktemp("llm_logs_session_")
        temp_dir_path_str = str(session_tmp_dir)
        os.environ["AHCAGENT_TEST_MODE_TEMP_WORKSPACE"] = temp_dir_path_str
        print(f"[CONTEST_SETUP] SET AHCAGENT_TEST_MODE_TEMP_WORKSPACE to: {temp_dir_path_str}")  # DEBUG PRINT
    else:
        # If logging is globally disabled, ensure AHCAGENT_TEST_MODE_TEMP_WORKSPACE is not set,
        # as it's not needed and could be misleading.
        if "AHCAGENT_TEST_MODE_TEMP_WORKSPACE" in os.environ:
            del os.environ["AHCAGENT_TEST_MODE_TEMP_WORKSPACE"]
            print("[CONTEST_SETUP] Logging globally disabled. UNSET AHCAGENT_TEST_MODE_TEMP_WORKSPACE (was set).")  # DEBUG PRINT
        else:
            print("[CONTEST_SETUP] Logging globally disabled. AHCAGENT_TEST_MODE_TEMP_WORKSPACE was not set (correct).")  # DEBUG PRINT

    yield
    print("\n[CONTEST_TEARDOWN] Running set_test_mode_temp_workspace fixture teardown...")  # DEBUG PRINT

    # Teardown: restore original environment variable state for AHCAGENT_TEST_MODE_TEMP_WORKSPACE
    if original_value is None:
        if "AHCAGENT_TEST_MODE_TEMP_WORKSPACE" in os.environ:
            del os.environ["AHCAGENT_TEST_MODE_TEMP_WORKSPACE"]
            print("[CONTEST_TEARDOWN] UNSET AHCAGENT_TEST_MODE_TEMP_WORKSPACE. Original was None.")  # DEBUG PRINT
    else:
        os.environ["AHCAGENT_TEST_MODE_TEMP_WORKSPACE"] = original_value
        print(f"[CONTEST_TEARDOWN] RESTORED AHCAGENT_TEST_MODE_TEMP_WORKSPACE to: {original_value}")  # DEBUG PRINT

    # Restore AHCAGENT_LLM_LOGGING_DISABLED if it was touched by test setup (though this fixture doesn't change it)
    if original_logging_disabled_env is None:
        if "AHCAGENT_LLM_LOGGING_DISABLED" in os.environ:
            # This case implies a test might have set it; this fixture doesn't, but good to be robust
            current_llm_disabled_env = os.getenv("AHCAGENT_LLM_LOGGING_DISABLED")
            print(f"[CONTEST_TEARDOWN] AHCAGENT_LLM_LOGGING_DISABLED is '{current_llm_disabled_env}'. Original was None.")  # DEBUG PRINT
            print("    It will be left as is or managed by the specific test.")  # DEBUG PRINT
    else:
        os.environ["AHCAGENT_LLM_LOGGING_DISABLED"] = original_logging_disabled_env
        print(f"[CONTEST_TEARDOWN] RESTORED AHCAGENT_LLM_LOGGING_DISABLED to: {original_logging_disabled_env}")  # DEBUG PRINT
    print("[CONTEST_TEARDOWN] Teardown complete.")  # DEBUG PRINT

    # For debugging:
    # if temp_dir_path_str:
    #     print(f"INFO: AHCAGENT_TEST_MODE_TEMP_WORKSPACE teardown. Original was: {original_value}")
    # elif logging_globally_disabled:
    #     print("INFO: LLM Logging was globally disabled for the session.")
