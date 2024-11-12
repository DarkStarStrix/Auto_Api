# app.py
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from python_jose import JWTError, jwt  # Changed from PyJWT to python-jose
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv ()

# JWT Configuration
SECRET_KEY = os.getenv ("JWT_SECRET_KEY", "your-secret-key-change-this-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30 * 24 * 60  # 30 days

# Security
security = HTTPBearer ()


def create_api_token(user_id: str) -> str:
    """Create a new JWT token"""
    expires_delta = timedelta (minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    expire = datetime.utcnow () + expires_delta

    to_encode = {
        "sub": str (user_id),
        "exp": expire,
        "iat": datetime.utcnow ()
    }

    try:
        encoded_jwt = jwt.encode (to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt
    except Exception as e:
        raise HTTPException (
            status_code=500,
            detail=f"Error creating token: {str (e)}"
        )


def verify_token(credentials: HTTPAuthorizationCredentials = Depends (security)):
    """Verify JWT token"""
    try:
        token = credentials.credentials
        payload = jwt.decode (token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get ("sub")
        if user_id is None:
            raise HTTPException (
                status_code=401,
                detail="Invalid authentication credentials"
            )
        return user_id
    except JWTError as e:
        raise HTTPException (
            status_code=401,
            detail=f"Invalid token: {str (e)}"
        )
    except Exception as e:
        raise HTTPException (
            status_code=500,
            detail=f"Error verifying token: {str (e)}"
        )
