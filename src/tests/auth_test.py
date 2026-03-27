from fastapi import HTTPException
from fastapi.testclient import TestClient

from main import app
from auth import get_current_user

client = TestClient(app)


def test_users_me_returns_user():
    app.dependency_overrides[get_current_user] = lambda: {
        "user_id": "test-user",
        "email": "test@test.com",
    }

    response = client.get("/users/me")
    assert response.status_code == 200
    data = response.json()
    assert "user_id" in data
    assert "email" in data

    app.dependency_overrides = {}


def test_user_isolation_between_users():
    # --- user A ---
    app.dependency_overrides[get_current_user] = lambda: {
        "user_id": "user-A",
        "email": "a@test.com",
    }

    client.post("/drop/tables")
    client.post("/create/tables")

    client.post("/create/category?name=A_cat")
    client.post("/create/conversation?name=A_conv")

    res = client.get("/get/conversations")
    assert len(res.json()) == 1

    # --- user B ---
    app.dependency_overrides[get_current_user] = lambda: {
        "user_id": "user-B",
        "email": "b@test.com",
    }

    res = client.get("/get/conversations")
    assert res.json() == []

    res = client.post("/update/conversation/category?conv_id=1&cat_id=1")
    assert res.status_code == 404

    app.dependency_overrides = {}


def test_users_me_requires_auth():
    app.dependency_overrides = {}

    response = client.get("/users/me")
    assert response.status_code == 401


def mock_auth_failure():
    raise HTTPException(status_code=401, detail="Invalid token")


def test_invalid_token_returns_401():
    app.dependency_overrides[get_current_user] = mock_auth_failure
    response = client.get("/users/me")
    assert response.status_code == 401
    app.dependency_overrides = {}
