CREATE EXTENSION IF NOT EXISTS vector;

-- documents: по одному "активному" документу на пользователя (чистим старые записи при загрузке/finish).
CREATE TABLE IF NOT EXISTS documents (
    id BIGSERIAL PRIMARY KEY,
    telegram_user_id BIGINT NOT NULL,

    filename TEXT NOT NULL,
    doc_name TEXT,
    doc_number TEXT,
    extracted_text_hash TEXT,

    processing_parameters JSONB,
    metadata JSONB,

    file_path TEXT,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- chunks: векторы в pgvector
CREATE TABLE IF NOT EXISTS chunks (
    id BIGSERIAL PRIMARY KEY,
    document_id BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,

    chunk_text TEXT NOT NULL,
    embedding vector(768) NOT NULL,

    chunk_order INT NOT NULL,
    keywords TEXT[],
    metadata JSONB,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- updated_at для удобства
CREATE OR REPLACE FUNCTION set_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_documents_updated_at ON documents;
CREATE TRIGGER trg_documents_updated_at
BEFORE UPDATE ON documents
FOR EACH ROW EXECUTE FUNCTION set_updated_at();

DROP TRIGGER IF EXISTS trg_chunks_updated_at ON chunks;
CREATE TRIGGER trg_chunks_updated_at
BEFORE UPDATE ON chunks
FOR EACH ROW EXECUTE FUNCTION set_updated_at();

CREATE INDEX IF NOT EXISTS idx_documents_telegram_user_id ON documents(telegram_user_id);
CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);

