from __future__ import absolute_import
from __future__ import print_function
import sys
import pytest

REQUIRED_PYTHON = "python"


@pytest.mark.run(order=0)
def test_import():
    import behav

if __name__ == '__main__':
    test_import()
