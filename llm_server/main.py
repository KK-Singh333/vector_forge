from fastapi import FastAPI, UploadFile, File, Form
from schemas import RegisterRequest, ChatRequest
from services.chat_service import ChatService
from services.knowledge_client import KnowledgeClient
from services.user_service import UserService
from services.llm_client import LLMClient
from dotenv import load_dotenv
load_dotenv()
app = FastAPI(title="Chat Service")

knowledge_client = KnowledgeClient(
    base_url="http://127.0.0.1:8068"
)

user_service = UserService("http://127.0.0.1:8068")
llm_client = LLMClient()

chat_service = ChatService(
    knowledge_client=knowledge_client,
    llm_client=llm_client
)

@app.post("/register")
async def register(request: RegisterRequest):

    user =await user_service.register(request.username)

    return {
        "user_id": user["user_id"],
        "username": user["username"]
    }


@app.post("/ingest")
async def upload(
    user_id: int = Form(...),
    pdf_id: str = Form(...),
    file: UploadFile = File(...)
):

    if not user_service.exists(user_id):
        return {"error": "User does not exist"}

    result = await knowledge_client.upload_pdf(
        user_id=user_id,
        pdf_id=pdf_id,
        file=file
    )

    return result


@app.post("/chat")
async def chat(request: ChatRequest):

    if not user_service.exists(request.user_id):
        return {"error": "User does not exist"}

    return await chat_service.ask(
        user_id=request.user_id,
        query=request.query,
        k=request.k
    )
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8069, reload=True)