from pathlib import Path

import pytest


@pytest.fixture
def rootpath():
    return Path(__file__).parents[1]

@pytest.fixture
def envpath():
    return '<path_to_conda_file>'
