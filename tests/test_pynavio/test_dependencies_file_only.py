from pathlib import Path
from pynavio.dependencies import infer_external_dependencies


def test_infer_external_dependencies_file_only():
    pip_requirements = infer_external_dependencies(Path(__file__), file_only=False)

    pip_requirements_file_only = infer_external_dependencies(Path(__file__))
    assert pip_requirements_file_only == []
    assert any('pytest' in item for item in pip_requirements)

