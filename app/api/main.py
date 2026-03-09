from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import ingest, search, register

app = FastAPI(title="Scalable FAISS Store")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],           # This allows all origins
    allow_credentials=True,
    allow_methods=["*"],           # Allows all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],           # Allows all headers
)

app.include_router(ingest.router)
app.include_router(search.router)
app.include_router(register.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="127.0.0.1", port=8068, reload=True)