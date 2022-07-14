from pathlib import Path

import pynavio
from pynavio.dependencies import infer_external_dependencies


def test_infer_external_dependencies_file_only():
    pip_requirements = infer_external_dependencies(Path(__file__),
                                                   file_only=False)
    pip_requirements_dir = infer_external_dependencies(Path(__file__).parent)
    pip_requirements_file_only = infer_external_dependencies(Path(__file__))

    # running make test will first install pynavio, so the module will
    # appear in pip requirements. Running pytest without installing will
    # cause pigar to ignore the module, so requirements will be empty
    expected = [[], [f'{pynavio.__name__}=={pynavio.__version__}']]
    assert pip_requirements_file_only in expected
    assert any('pytest' in item for item in pip_requirements)
    assert any('pytest' in item for item in pip_requirements_dir)
