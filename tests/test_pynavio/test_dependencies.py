from pathlib import Path

from pynavio.dependencies import (_generate_ignore_dirs_args,
                                  infer_external_dependencies,
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


def test_generate_ignore_dirs_args(tmp_path):
    folder_name_containing_venv = 'pyvenv1'
    (tmp_path / folder_name_containing_venv).mkdir()
    ignore_dirs_args = _generate_ignore_dirs_args(tmp_path, None)
    assert len(ignore_dirs_args) == 2
    assert f'{ignore_dirs_args[-1]}'.endswith(folder_name_containing_venv)


def test_generate_ignore_dirs_args_with_to_ignore_dirs(tmp_path):
    # checks that if to_ignore_dirs is specified, it will no longer ignore
    # the default ones (e.g. *venv* in this case)
    folder_name_containing_venv = 'pyvenv1'
    other_path = 'other'
    (tmp_path / folder_name_containing_venv).mkdir()
    (tmp_path / other_path).mkdir()
    ignore_dirs_args = _generate_ignore_dirs_args(tmp_path,
                                                  [tmp_path / other_path])
    assert len(ignore_dirs_args) == 2
    assert f'{ignore_dirs_args[-1]}'.endswith(other_path)


def test_infer_external_dependencies():
    # import packages to generate requirements for this file
    # and make sure those are correctly identified as dependencies

    import mlflow  # noqa: F401
    import numpy  # noqa: F401

    # this is to demonstrate, that even if the import statement will never
    # be executed, it will still be in the output
    if False:
        import pandas as pd  # noqa: F401
        exec('import sklearn')

    pip_requirements = infer_external_dependencies(Path(__file__).parent)
    pip_requirements = ' '.join(pip_requirements)

    for item in ['pandas', 'numpy', 'mlflow', 'scikit_learn']:
        assert item in pip_requirements
