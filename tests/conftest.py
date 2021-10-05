from pathlib import Path

import pytest


@pytest.fixture
def tests_rootpath():
    return Path(__file__).parent
