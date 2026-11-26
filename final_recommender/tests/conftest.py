"""Shared fixtures. Tests may be run from anywhere:

    pytest final_recommender/tests          # from the repo root
    pytest tests                            # from inside final_recommender

Tests that need RDKit are skipped automatically when it is not installed, so the
output-consistency tests still run in a bare environment.
"""
import os
import sys

import pytest

REC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # .../final_recommender
if REC not in sys.path:
    sys.path.insert(0, REC)

try:
    import rdkit  # noqa: F401
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False

needs_rdkit = pytest.mark.skipif(not HAS_RDKIT, reason='RDKit not installed')


@pytest.fixture(scope='session')
def rec_dir():
    """Absolute path of final_recommender/, where every CSV lives."""
    return REC


@pytest.fixture(scope='session')
def F(rec_dir):
    """The loaded feature context. One build for the whole session."""
    pytest.importorskip('rdkit')
    import features
    return features.load(rec_dir)
