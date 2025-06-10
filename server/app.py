# server/app.py
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def root():
    return {"status": "ok", "msg": "Basketball foul detector API placeholder"}
