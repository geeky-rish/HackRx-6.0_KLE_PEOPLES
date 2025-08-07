
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from core import process_pdf_and_answer_questions
import os
from pathlib import Path

env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)

app = FastAPI(title="Policy QA API", version="1.0.0")

class QARequest(BaseModel):
    documents: str
    questions: list[str]

@app.post("/hackrx/run")
async def run_qa(qa: QARequest):
    try:
        answers = process_pdf_and_answer_questions(qa.documents, qa.questions)
        return JSONResponse(content={"answers": answers})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
