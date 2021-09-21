from pynavio.utils.infer_dependencies import (infer_external_dependencies,
                                              read_requirements_txt)


def test_read_requirements_txt(tmp_path):
    fixture_text = '''### fixture requirements.txt

                      pandas==1.2.4
                      numpy
    '''
    file_path = tmp_path/'requirements.txt'
    with open(file_path, 'w') as f:
        f.write(fixture_text)

    assert read_requirements_txt(file_path) == ['pandas==1.2.4', 'numpy']


def test_infer_external_dependencies():
    import numpy
    import pandas as pd
    if False:
        import mlflow
        exec("exec('import sklearn')")

    pip_requirements = infer_external_dependencies('.')

    assert any('pandas' in item for item in pip_requirements)
    assert any('numpy' in item for item in pip_requirements)
    assert any('mlflow' in item for item in pip_requirements)
    assert any('scikit_learn' in item for item in pip_requirements)
