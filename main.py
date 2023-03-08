from typing import List, Dict

from fastapi import FastAPI
from classify import run_classifier

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/classify")
def classify(examples: List[Dict[str, str]], input: str) -> List[str]:
    run_classifier(examples, input)
