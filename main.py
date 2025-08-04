from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from dotenv import load_dotenv
from core import process_pdf_and_answer_questions
from fastapi.openapi.utils import get_openapi
import os
from pathlib import Path
from dotenv import load_dotenv

from dotenv import load_dotenv
from pathlib import Path
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)
API_KEY = os.getenv("API_KEY")



# Initialize FastAPI app and auth scheme
app = FastAPI(title="Policy QA API", version="1.0.0")
security = HTTPBearer()

# Dependency to verify token
def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")

# Request body schema
class QARequest(BaseModel):
    documents: str
    questions: list[str]

# Main endpoint
@app.post("/hackrx/run")
async def run_qa(qa: QARequest, creds: HTTPAuthorizationCredentials = Depends(verify_token)):
    try:
        answers = process_pdf_and_answer_questions(qa.documents, qa.questions)
        return JSONResponse(content={"answers": answers})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Custom OpenAPI schema to show Bearer auth in Swagger UI
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="Policy QA API",
        version="1.0.0",
        description="Answer questions about policy PDFs using Gemini Pro.",
        routes=app.routes,
    )
    openapi_schema["components"]["securitySchemes"] = {
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT"
        }
    }
    for path in openapi_schema["paths"].values():
        for method in path.values():
            method["security"] = [{"BearerAuth": []}]
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

# Run app locally
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
