import firebase_admin
from firebase_admin import credentials, auth
from fastapi import HTTPException, Header

cred = credentials.ApplicationDefault()
firebase_admin.initialize_app(cred)


def verify_token(auth_header: str):
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(401, "Missing token")

    token = auth_header.split(" ")[1]

    try:
        decoded = auth.verify_id_token(token)
        return decoded
    except Exception:
        raise HTTPException(401, "Invalid token")


def get_current_user(authorization: str = Header(None)):
    decoded = verify_token(authorization)
    return {
        "user_id": decoded["uid"],
        "email": decoded.get("email"),
    }
