"""DayOne API: photo of a registry page -> reviewed, patient-linked, synced record.

Run from this folder:   uvicorn main:app --reload
Then open:              http://localhost:8000        (the chat)
                        http://localhost:8000/docs   (every endpoint, clickable)
"""
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from core import config
from api import images, patients, records, sync
from offline import IllegalTransition
from services.ai_service import active_extractor
from services.record_service import ApiError

app = FastAPI(title="DayOne API", description="Backend for maternal registry digitization", version="0.2.0")

# The chat page may also be opened straight from disk (file://), so allow any origin.
# Fine for a local demo; restrict before deploying anywhere.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

app.include_router(records.router)
app.include_router(images.router)
app.include_router(records.demo_router)
app.include_router(patients.router)
app.include_router(sync.router)


@app.exception_handler(ApiError)
def api_error(_: Request, e: ApiError):
    return JSONResponse({"detail": e.message, **e.extra}, status_code=e.status_code)


@app.exception_handler(IllegalTransition)
def illegal(_: Request, e: IllegalTransition):
    return JSONResponse({"detail": str(e)}, status_code=409)


@app.exception_handler(ValueError)
def bad_value(_: Request, e: ValueError):
    return JSONResponse({"detail": str(e)}, status_code=422)


@app.get("/", include_in_schema=False)
def chat_page():
    return FileResponse(config.FRONTEND_FILE, media_type="text/html")


@app.get("/health")
def health():
    return {"status": "healthy", "extractor": active_extractor()}
