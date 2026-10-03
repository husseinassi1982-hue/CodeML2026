from fastapi import FastAPI

from api.test_route import router as test_router

app = FastAPI(
    title= "CodeML DayOne API",
    description="Backend for maternal registry digitization",
    version="0.1.0"
)

app.include_router(test_router)

@app.get("/")
def root():
    return {
        "status" : "ok",
        "message": "CodeML backend is running"
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy"

    }