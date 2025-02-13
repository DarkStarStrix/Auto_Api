from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
from Auth import router as auth_router
from Dashboard import router as dashboard_router
import os

app = FastAPI ()

app.add_middleware (
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs ("static", exist_ok=True)

app.mount ("/static", StaticFiles (directory="static"), name="static")
app.include_router (auth_router, prefix="/api/v1", tags=["auth"])
app.include_router (dashboard_router, prefix="/api/v1", tags=["dashboard"])


@app.get ("/")
async def root():
    return FileResponse ('static/index.html')


if __name__ == "__main__":
    uvicorn.run ("Main:app", host="0.0.0.0", port=8000, reload=True)
