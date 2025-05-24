import pytest

from ahc_agent_cli.utils.docker_manager import DockerManager


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
