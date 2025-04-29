"""
    Dummy conftest.py for greencurve.

    If you don't know what this is for, just leave it empty.
    Read more about conftest.py under:
    - https://docs.pytest.org/en/stable/fixture.html
    - https://docs.pytest.org/en/stable/writing_plugins.html
"""

import pytest
import warnings
from holidays.deprecations.v1_incompatibility import FutureIncompatibilityWarning

warnings.filterwarnings("ignore", category=FutureIncompatibilityWarning)
