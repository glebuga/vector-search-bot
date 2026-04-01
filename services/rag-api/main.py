import os
import io
import re
import json
import hashlib
import datetime
import html
import logging
from typing import Any, Dict, List, Optional
from contextlib import asynccontextmanager

import httpx
import pdfplumber
from pydantic import BaseModel
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from databases import Database
from sentence_transformers import SentenceTransformer
import yake
from typograf import RemoteTypograf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("rag-api")

load_dotenv()

DB_URL = os.getenv("DATABASE_URL", "")
EMBEDDINGS_MODEL_NAME = os.getenv(
    "EMBEDDINGS_MODEL_NAME",
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
)
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "768"))
LLM_SERVICE_URL = os.getenv("LLM_SERVICE_URL", "http://llm-service:8000/api/llm/answer")

TYPOGRAPH_REMOTE_ENABLED = os.getenv("TYPOGRAPH_REMOTE_ENABLED", "false").lower() in ("1", "true", "yes")

DOCS_ALLOWED_EXTENSIONS = {"pdf"}

database = Database(DB_URL)

embedding_model_instance: Optional[SentenceTransformer] = None
kw_extractor = yake.KeywordExtractor(lan="ru", n=1, dedupLim=0.9, dedupFunc="seqm", windowsSize=1, top=10)
rt: Optional[RemoteTypograf] = RemoteTypograf(p=False, br=False) if TYPOGRAPH_REMOTE_ENABLED else None


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in DOCS_ALLOWED_EXTENSIONS


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not DB_URL:
        raise RuntimeError("DATABASE_URL is not set")
    if not LLM_SERVICE_URL:
        raise RuntimeError("LLM_SERVICE_URL is not set")

    await database.connect()
    logger.info("Connected to PostgreSQL.")

    global embedding_model_instance
    logger.info(f"Loading embeddings model: {EMBEDDINGS_MODEL_NAME}")
    embedding_model_instance = SentenceTransformer(EMBEDDINGS_MODEL_NAME)
    logger.info("Embeddings model loaded.")

    yield

    await database.disconnect()
    logger.info("Disconnected from PostgreSQL.")


app = FastAPI(title="RAG API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class UploadDocResponse(BaseModel):
    document_id: int
    chunks_count: int


class AskRequest(BaseModel):
    telegram_user_id: int
    question: str
    top_k: int = 5


class AskResponse(BaseModel):
    answer: str


def strip_html_tags_and_unescape(html_text: Optional[str]) -> str:
    if not html_text:
        return ""
    return html.unescape(re.sub(re.compile("<.*?>"), "", html_text))


def advanced_text_processing(text: Optional[str]) -> str:
    """
    Небольшая "чистка" и реконструкция абзацев из PDF-текста.
    """
    if not text:
        return ""
    lines_initial = [line.strip() for line in text.splitlines()]

    corrected_lines_stage1: List[str] = []
    for line in lines_initial:
        line = re.sub(r"(\d+)\s*\.\s*(\d+)\s*\.\s*(\d+)", r"\1.\2.\3", line)
        line = re.sub(r"(\d+)\s*\.\s*(\d+)", r"\1.\2", line)
        line = re.sub(r"(\d)\s*-\s*(\d)", r"\1-\2", line)
        corrected_lines_stage1.append(line)

    lines_for_paragraph_joining = corrected_lines_stage1
    reconstructed_paragraphs: List[str] = []
    current_paragraph_parts: List[str] = []
    sentence_enders = [".", "!", "?", "…", ":", ";"]
    list_markers = re.compile(r"^\s*(?:\d+[\.\)]|[a-zA-Z][\.\)]|[-*•–—IVXLCDMivxlcdm]+[\.\)]?)\s+")

    for i, current_line in enumerate(lines_for_paragraph_joining):
        if not current_line:
            if current_paragraph_parts:
                reconstructed_paragraphs.append(" ".join(current_paragraph_parts))
                current_paragraph_parts = []
            reconstructed_paragraphs.append("")
            continue

        current_paragraph_parts.append(current_line)
        current_line_ends_with_sentence_ender = any(current_line.endswith(ender) for ender in sentence_enders)

        next_line_forces_break = False
        if i + 1 < len(lines_for_paragraph_joining):
            next_line_stripped = lines_for_paragraph_joining[i + 1].strip()
            if next_line_stripped:
                condition_next_uppercase = next_line_stripped[0].isupper() and not current_line.endswith(",")
                condition_next_list_marker = list_markers.match(next_line_stripped)
                condition_next_short_header = (
                    current_line_ends_with_sentence_ender
                    and (len(next_line_stripped.split()) < 4 and len(next_line_stripped) < 50)
                    and not condition_next_list_marker
                )
                if condition_next_uppercase or condition_next_list_marker or condition_next_short_header:
                    next_line_forces_break = True

        current_line_is_short_header_or_list_item = (len(current_line.split()) < 5 and len(current_line) < 60) or (
            list_markers.match(current_line)
        )
        if (
            current_line_ends_with_sentence_ender
            or next_line_forces_break
            or (current_line_is_short_header_or_list_item and i > 0 and len(current_paragraph_parts) == 1)
        ):
            if current_paragraph_parts:
                reconstructed_paragraphs.append(" ".join(current_paragraph_parts))
                current_paragraph_parts = []

    if current_paragraph_parts:
        reconstructed_paragraphs.append(" ".join(current_paragraph_parts))

    final_text = "\n".join(reconstructed_paragraphs)
    final_text = re.sub(r"(\n\s*){2,}", "\n\n", final_text)
    final_text = re.sub(r" +", " ", final_text)
    return final_text.strip()


def clean_text_final_pass(text: Optional[str]) -> str:
    if not text:
        return ""
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    cleaned_lines = [line for line in lines if line]
    final_cleaned_text = "\n".join(cleaned_lines)
    return final_cleaned_text.strip()


def extract_text_from_pdf_bytes(
    pdf_bytes: bytes,
    filename: str,
    header_margin_percent: int = 5,
    footer_margin_percent: int = 4,
    x_tolerance_val: int = 4,
    y_tolerance_val: int = 4,
) -> str:
    logger.info(f"Extract text from PDF: {filename}")
    all_pages_text_parts: List[str] = []

    try:
        with io.BytesIO(pdf_bytes) as f_bytes:
            with pdfplumber.open(f_bytes) as pdf:
                logger.info(f"Pages found: {len(pdf.pages)}")
                for page in pdf.pages:
                    page_height = page.height
                    header_cutoff = page_height * (header_margin_percent / 100.0)
                    footer_cutoff = page_height * (1 - footer_margin_percent / 100.0)
                    main_content_area = page.crop((0, header_cutoff, page.width, footer_cutoff))

                    raw_text_from_page = main_content_area.extract_text(
                        use_text_flow=True, x_tolerance=x_tolerance_val, y_tolerance=y_tolerance_val
                    )
                    if raw_text_from_page:
                        all_pages_text_parts.append(raw_text_from_page)

        monolithic_text = "\n".join(all_pages_text_parts)
        processed_text = advanced_text_processing(monolithic_text)

        if processed_text.strip():
            if rt is not None:
                try:
                    processed_text_html = rt.process_text(processed_text)
                    final_full_text = strip_html_tags_and_unescape(processed_text_html)
                    final_full_text = clean_text_final_pass(final_full_text)
                    return final_full_text
                except Exception as e:
                    logger.warning(f"RemoteTypograf failed, fallback to cleaned text: {e}")
                    return clean_text_final_pass(processed_text)
            return clean_text_final_pass(processed_text)

        return "(pdfplumber: В документе не найден текст или текст пуст [обработано])"
    except Exception as e:
        logger.exception(f"PDF extraction failed for {filename}: {e}")
        raise HTTPException(status_code=400, detail=f"Ошибка обработки PDF ({filename}): {str(e)}")


def simple_chunker(text: str, chunk_size: int = 700, chunk_overlap: int = 100) -> List[str]:
    if not text:
        return []
    paragraphs = text.split("\n\n")
    chunks: List[str] = []
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(paragraph) <= chunk_size:
            chunks.append(paragraph)
        else:
            start = 0
            while start < len(paragraph):
                end = start + chunk_size
                chunk = paragraph[start:end]
                chunks.append(chunk)
                if end >= len(paragraph):
                    break
                start += (chunk_size - chunk_overlap)
                if start >= len(paragraph):
                    break

    final_chunks = [c.strip() for c in chunks if len(c.strip()) > 30]
    if not final_chunks and text.strip():
        logger.warning("Chunker returned empty meaningful chunks, fallback to coarse split.")
        if len(text.strip()) <= chunk_size * 1.5:
            return [text.strip()]
        num_parts = (len(text.strip()) + chunk_size - 1) // chunk_size or 1
        part_len = len(text.strip()) // num_parts
        return [text.strip()[i * part_len : (i + 1) * part_len] for i in range(num_parts)]
    return final_chunks if final_chunks else ([text.strip()] if text.strip() else [])


def get_text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def generate_embeddings(text: str) -> List[float]:
    if embedding_model_instance is None:
        raise RuntimeError("Embedding model not loaded yet")
    embedding = embedding_model_instance.encode(text, show_progress_bar=False)
    vector = embedding.tolist()
    if len(vector) != EMBEDDING_DIM:
        logger.warning(f"Embedding dim mismatch: got {len(vector)} expected {EMBEDDING_DIM}")
    return vector


def embedding_to_pgvector_literal(embedding_vector: List[float]) -> str:
    # pgvector accepts literals like [1,2,3]
    return "[" + ",".join(map(str, embedding_vector)) + "]"


@app.post("/api/documents/upload", response_model=UploadDocResponse)
async def upload_document(
    telegram_user_id: int = Form(...),
    pdf_file: UploadFile = File(...),
    doc_name: Optional[str] = Form(None),
    doc_number: Optional[str] = Form(None),
):
    if not pdf_file.filename or not allowed_file(pdf_file.filename):
        raise HTTPException(status_code=400, detail="Пожалуйста, загрузите PDF файл.")

    if not telegram_user_id:
        raise HTTPException(status_code=400, detail="telegram_user_id is required")

    filename = pdf_file.filename
    try:
        pdf_bytes = await pdf_file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Не удалось прочитать файл: {e}")

    extracted_text = extract_text_from_pdf_bytes(pdf_bytes, filename=filename)
    if not extracted_text.strip() or extracted_text.startswith("(pdfplumber: В документе не найден текст"):
        raise HTTPException(status_code=400, detail="Не удалось извлечь текст из PDF.")

    extracted_text_hash = get_text_hash(extracted_text)

    # Гарантируем "один активный документ" на пользователя
    await database.execute("DELETE FROM documents WHERE telegram_user_id = :telegram_user_id", {"telegram_user_id": telegram_user_id})

    effective_doc_name = (doc_name or os.path.splitext(filename)[0]).strip()
    now_metadata: Dict[str, Any] = {
        "source_filename": filename,
        "uploaded_at": datetime.datetime.utcnow().isoformat(),
    }

    query_insert_doc = """
    INSERT INTO documents
      (telegram_user_id, filename, doc_name, doc_number, extracted_text_hash, processing_parameters, metadata, file_path)
    VALUES
      (:telegram_user_id, :filename, :doc_name, :doc_number, :extracted_text_hash, :processing_parameters, :metadata, NULL)
    RETURNING id;
    """
    doc_id_row = await database.fetch_one(
        query_insert_doc,
        {
            "telegram_user_id": telegram_user_id,
            "filename": filename,
            "doc_name": effective_doc_name,
            "doc_number": doc_number,
            "extracted_text_hash": extracted_text_hash,
            "processing_parameters": None,
            "metadata": json.dumps(now_metadata),
        },
    )
    if not doc_id_row:
        raise HTTPException(status_code=500, detail="Не удалось создать запись документа в БД.")
    document_id = int(doc_id_row["id"])

    chunks = simple_chunker(extracted_text)
    if not chunks:
        raise HTTPException(status_code=400, detail="Не удалось нарезать документ на чанки.")

    query_insert_chunk = """
    INSERT INTO chunks
      (document_id, chunk_text, embedding, chunk_order, keywords, metadata)
    VALUES
      (:document_id, :chunk_text, CAST(:embedding AS vector(768)), :chunk_order, :keywords, :metadata);
    """

    keywords_and_embeddings = 0
    for i, chunk_text in enumerate(chunks):
        keywords_list = [kw[0] for kw in kw_extractor.extract_keywords(chunk_text)]
        embedding_vector = generate_embeddings(chunk_text)

        meta = {
            "part_of_document": f"chunk_{i+1}_of_{len(chunks)}",
            "chunk_index": i + 1,
        }

        await database.execute(
            query_insert_chunk,
            {
                "document_id": document_id,
                "chunk_text": chunk_text,
                "embedding": embedding_to_pgvector_literal(embedding_vector),
                "chunk_order": i,
                "keywords": keywords_list if keywords_list else None,
                "metadata": json.dumps(meta),
            },
        )
        keywords_and_embeddings += 1

    return UploadDocResponse(document_id=document_id, chunks_count=len(chunks))


async def call_llm_service(question: str, context: str) -> str:
    async with httpx.AsyncClient(timeout=300) as client:
        resp = await client.post(
            LLM_SERVICE_URL,
            json={"question": question, "context": context},
            headers={"Content-Type": "application/json"},
        )
        if resp.status_code >= 400:
            raise HTTPException(status_code=502, detail=f"LLM service error: {resp.text}")
        data = resp.json()
        answer = (data.get("answer") or "").strip()
        return answer or "Произошла ошибка при формировании ответа."


@app.post("/api/qa/ask", response_model=AskResponse)
async def ask(req: AskRequest) -> AskResponse:
    telegram_user_id = req.telegram_user_id
    question = (req.question or "").strip()
    top_k = int(req.top_k or 5)

    if not telegram_user_id:
        raise HTTPException(status_code=400, detail="telegram_user_id is required")
    if not question:
        raise HTTPException(status_code=400, detail="question is required")
    top_k = max(1, min(top_k, 20))

    question_embedding = generate_embeddings(question)
    question_embedding_literal = embedding_to_pgvector_literal(question_embedding)

    query = """
    SELECT c.chunk_text
    FROM chunks c
    JOIN documents d ON c.document_id = d.id
    WHERE d.telegram_user_id = :telegram_user_id
    ORDER BY c.embedding <-> CAST(:embedding AS vector(768))
    LIMIT :top_k;
    """
    rows = await database.fetch_all(
        query,
        {
            "telegram_user_id": telegram_user_id,
            "embedding": question_embedding_literal,
            "top_k": top_k,
        },
    )

    if not rows:
        raise HTTPException(status_code=404, detail="Информация по данному вопросу не найдена в базе.")

    context = "\n\n---\n\n".join([r["chunk_text"] for r in rows if r and r["chunk_text"]])
    answer = await call_llm_service(question=question, context=context)

    # Унификация ответа на случай, если LLM вернул близкую формулировку.
    if "не найдена в базе" in answer.lower():
        answer = "Информация по данному вопросу не найдена в базе"

    return AskResponse(answer=answer)


@app.post("/api/documents/finish")
async def finish_document(telegram_user_id: int = Form(...)):
    if not telegram_user_id:
        raise HTTPException(status_code=400, detail="telegram_user_id is required")

    deleted = await database.execute(
        "DELETE FROM documents WHERE telegram_user_id = :telegram_user_id",
        {"telegram_user_id": telegram_user_id},
    )
    # databases.execute returns status string; we don't strictly need exact count
    return {"status": "ok", "result": deleted}


@app.get("/api/health")
async def health():
    return {"status": "ok"}

