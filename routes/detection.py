from fastapi import APIRouter, UploadFile, File, Form, Depends
from controllers.document_controller import process_document

router = APIRouter()

@router.post("/verify-document")
async def verify_document(
    card_type: str = Form(...), 
    file: UploadFile = File(...) 
):
    return await process_document(card_type, file)
