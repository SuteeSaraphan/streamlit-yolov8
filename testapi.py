from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI()

@app.get("/")
def index():
    return {"message": "Hello World"}

#recive json data
@app.post("/recive_json")
def recive_json(data: dict):
    print(data)