import os

import firebase_admin
from firebase_admin import credentials, auth
from fastapi import HTTPException, Header

FIREBASE_APP = None


def _init_firebase():
    global FIREBASE_APP # pylint: disable=global-statement
    if FIREBASE_APP is None:
        cred = credentials.ApplicationDefault()
        project_id = os.environ["FIREBASE_PROJECT_ID"]
        FIREBASE_APP = firebase_admin.initialize_app(cred, {"projectId": project_id})


def verify_token(auth_header: str):
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(401, "Missing token")

    token = auth_header.split(" ")[1]

    try:
        _init_firebase()
        decoded = auth.verify_id_token(token)
        return decoded
    except Exception as e: # pylint: disable=broad-exception-caught
        raise HTTPException(401, "Invalid token") from e


def get_current_user(authorization: str = Header(None)):
    decoded = verify_token(authorization)
    return {
        "user_id": decoded["uid"],
        "email": decoded.get("email"),
    }
