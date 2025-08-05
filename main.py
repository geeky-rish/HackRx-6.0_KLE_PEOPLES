from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from core import process_pdf_and_answer_questions
import os
from pathlib import Path

# Load .env variables
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)

# Initialize FastAPI app
app = FastAPI(title="Policy QA API", version="1.0.0")

# Request body schema
class QARequest(BaseModel):
    documents: str
    questions: list[str]

# Main public endpoint (no auth)
@app.post("/hackrx/run")
async def run_qa(qa: QARequest):
    try:
        answers = process_pdf_and_answer_questions(qa.documents, qa.questions)
        return JSONResponse(content={"answers": answers})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run app locally
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
