import pytest

from business.catalog import reload_business_data


@pytest.fixture(autouse=True)
def isolated_price_overrides(tmp_path, monkeypatch):
    """每个测试用自己的空覆盖文件，不读写 /app/data，也不互相影响。"""
    monkeypatch.setenv("GOEUROOPS_PRICE_OVERRIDES", str(tmp_path / "price_overrides.json"))
    reload_business_data()
    yield
    reload_business_data()
