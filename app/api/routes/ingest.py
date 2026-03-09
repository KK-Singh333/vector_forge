from fastapi import APIRouter, UploadFile, File, Form
import os
import uuid
router = APIRouter(prefix="/ingest", tags=["Ingestion"])
from api.container import ingestion_service, cached_search_service

UPLOAD_DIR = "uploaded_pdfs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/")
async def ingest(
    user_id: int = Form(...),
    pdf_id: str = Form(...),
    file: UploadFile = File(...)
):
    if not file.filename.endswith(".pdf"):
        return {"status": "failed", "reason": "Only PDF allowed"}

    unique_name = f"{uuid.uuid4()}_{file.filename}"
    file_path = os.path.join(UPLOAD_DIR, unique_name)

    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    success = ingestion_service.ingest_pdf(
        file_path=file_path,
        user_id=int(user_id),
        pdf_id=int(pdf_id)
    )

    if success:
        cached_search_service.invalidate_user(user_id)
        return {"status": "success"}

    return {"status": "failed"}
