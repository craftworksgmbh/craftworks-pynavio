from pathlib import Path

from pynavio.utils.infer_dependencies import (infer_external_dependencies,
                                              read_requirements_txt)


def test_read_requirements_txt(tmp_path):
    fixture_text = '''### fixture requirements.txt

                      pandas==1.2.4
                      numpy
    '''
    file_path = tmp_path / 'requirements.txt'
    with open(file_path, 'w') as f:
        f.write(fixture_text)

    assert read_requirements_txt(file_path) == ['pandas==1.2.4', 'numpy']


def test_infer_external_dependencies():
    # import packages to generate requirements for this file
    # and make sure those are correctly identified as dependencies

    import mlflow  # noqa: F401
    import numpy  # noqa: F401

    # this is to demonstrate, that even if the import statement will never be executed, it will still be in the output
    if False:
        import pandas as pd  # noqa: F401
        exec('import sklearn')

    pip_requirements = infer_external_dependencies(Path(__file__).parent)

    assert any('pandas' in item for item in pip_requirements)
    assert any('numpy' in item for item in pip_requirements)
    assert any('mlflow' in item for item in pip_requirements)
    assert any('scikit_learn' in item for item in pip_requirements)
