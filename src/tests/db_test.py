import db


def test_db_module_exports_expected_symbols():
    assert hasattr(db, "engine")
    assert hasattr(db, "sessionlocal")
    assert hasattr(db, "Base")
    assert hasattr(db, "create_tables")
    assert hasattr(db, "drop_tables")

    url_str = str(db.engine.url)
    assert isinstance(url_str, str)
    assert url_str != ""
