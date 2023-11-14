from pathlib import Path

import pytest


@pytest.fixture
def rootpath():
    return Path(__file__).parents[1]
