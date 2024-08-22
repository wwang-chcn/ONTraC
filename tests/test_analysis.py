# test if seaborn is installed, otherwise skip the this test

import pytest

try:
    import seaborn
    seaborn_imported = True
except ImportError:
    seaborn_imported = False

@pytest.mark.skipif(not seaborn_imported, reason="seaborn is not installed")
def test_XXX():
    pass