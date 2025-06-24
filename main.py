import os
import logging
from dotenv import load_dotenv

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes.detection import router as document_router
from utils.database import MongoDB

# Load environment variables from .env file
load_dotenv()
PORT = int(os.getenv("PORT", 8000))  # default to 8000 if not set

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Test database connection on startup
@app.on_event("startup")
async def startup_db_client():
    try:
        mongo = MongoDB()
        db = mongo.get_database()
        db.command('ping')
        logger.info("Successfully connected to MongoDB!")
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        raise

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Include routers
app.include_router(document_router, prefix="/api/document")

@app.get("/")
def root():
    return {"message": "Welcome to FastAPI with MongoDB"}

@app.get("/health")
def health_check():
    try:
        mongo = MongoDB()
        db = mongo.get_database()
        db.command('ping')
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {"status": "unhealthy", "database": "disconnected", "error": str(e)}

# Run app programmatically using port from .env
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=True)
