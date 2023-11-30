from pathlib import Path

import pytest


@pytest.fixture
def rootpath():
    return Path(__file__).parents[1]


@pytest.fixture()
def fixtures_path(rootpath):
    return Path(rootpath, 'tests', 'test_pynavio', 'fixtures')
