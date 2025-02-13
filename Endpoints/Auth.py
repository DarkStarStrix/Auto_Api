from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
from jose import jwt, JWTError
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

load_dotenv ()

router = APIRouter ()
oauth2_scheme = OAuth2PasswordBearer (tokenUrl="token")

SECRET_KEY = os.getenv ("SECRET_KEY", "development-key")
ALGORITHM = "HS256"


class User (BaseModel):
    username: str
    email: str = None


class Token (BaseModel):
    access_token: str
    token_type: str


users_db = {}


@router.post ("/token")
async def login(username: str, password: str):
    token = jwt.encode (
        {"sub": username, "exp": datetime.utcnow () + timedelta (hours=24)},
        SECRET_KEY,
        algorithm=ALGORITHM
    )
    return Token (access_token=token, token_type="bearer")


def get_current_user(token: str = Depends (oauth2_scheme)):
    try:
        payload = jwt.decode (token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get ("sub")
        if not username:
            raise HTTPException (status_code=401)
        return {"username": username}
    except JWTError:
        raise HTTPException (status_code=401)
