# Skip this test if seaborn is unavailable.

import pytest

try:
    import seaborn
    seaborn_imported = True
except ImportError:
    seaborn_imported = False

@pytest.mark.skipif(not seaborn_imported, reason="seaborn is not installed")
def test_analysis_dependency_smoke():
    """Smoke-test that optional analysis plotting dependencies can be imported."""
    pass
