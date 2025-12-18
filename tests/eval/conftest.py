"""
Pytest configuration for eval tests.
"""

import pytest


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "e2e: mark test as end-to-end (makes real API calls, costs money)"
    )


def pytest_collection_modifyitems(config, items):
    """Skip e2e tests by default unless explicitly requested."""
    if config.getoption("-m") and "e2e" in config.getoption("-m"):
        # e2e tests explicitly requested, don't skip
        return
    
    skip_e2e = pytest.mark.skip(reason="e2e tests skipped by default (use -m e2e to run)")
    for item in items:
        if "e2e" in item.keywords:
            item.add_marker(skip_e2e)
