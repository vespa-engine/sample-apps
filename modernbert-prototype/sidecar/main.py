from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import uvicorn
import numpy as np
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(debug=True)


# Middleware to log all requests
@app.middleware("http")
async def log_requests(request: Request, call_next):
    body = await request.body()
    logger.debug(f"Request: {request.method} {request.url}")
    logger.debug(f"Headers: {request.headers}")
    logger.debug(f"Body: {body.decode()}")

    response = await call_next(request)
    return response


# Simple in-memory "state"
status = "not_ready"
model_id = None


class InitializeRequest(BaseModel):
    modelId: str


@app.post("/initialize")
async def initialize(request: Request, request_data: InitializeRequest):
    body = await request.body()
    logger.debug(f"Raw initialize request body: {body.decode()}")
    logger.debug(f"Parsed request data: {request_data}")

    global status, model_id
    model_id = request_data.modelId
    status = "ready"
    return {"message": f"Model initialized successfully with ID: {model_id}"}


@app.get("/status")
def get_status():
    """
    Endpoint to return whether the model is ready.
    The Java code checks if it receives {"status":"ready"}.
    """
    return {"status": status}


class EmbedRequest(BaseModel):
    modelId: str
    text: str


@app.post("/embed")
def embed(request_data: EmbedRequest, num_dims: int):
    """
    Endpoint to embed text using the specified modelId.
    For demonstration, this just returns a fixed embedding.
    Replace with actual embedding logic as needed.
    """
    global status, model_id
    if status != "ready" or request_data.modelId != model_id:
        # Model is not ready or incorrect model ID
        raise HTTPException(
            status_code=400, detail="Model is not ready or invalid modelId"
        )

    # Your embedding logic here. For example:
    # embedding_vector = real_embed_function(request_data.text)
    # Return some mock integer list for demonstration:
    embedding_vector = np.random.randint(0, 100, num_dims).tolist()

    return {"embedding": embedding_vector}


# Create health check endpoint
@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=1337, log_level="debug")
