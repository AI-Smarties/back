import db

def test_db_module_exports_expected_symbols():
    assert hasattr(db, "engine")
    assert hasattr(db, "Session")
    assert hasattr(db, "Base")

    url_str = str(db.engine.url)
    assert isinstance(url_str, str)
    assert url_str != ""
