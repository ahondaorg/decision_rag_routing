import os
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import HeadingHierarchyOptions, PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from typing import List
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_postgres.vectorstores import PGVector
from sqlalchemy import create_engine, text, MetaData, Table
from collections import Counter
from dotenv import load_dotenv 


file_path = "./data/"
file_names = ["senstadt_step_broschuere_wohnen.pdf","20230525_step-klima20-online.pdf","senstadt_step-broschuere-wirtschaft.pdf"]

## docling option
pipeline_options = PdfPipelineOptions()
pipeline_options.heading_hierarchy_options = HeadingHierarchyOptions(enabled=True)
# The font-style signal reads the parsed PDF cells, which are dropped unless you keep them:
pipeline_options.generate_parsed_pages = True
pipeline_options.do_ocr = False

## for chunking
HEADER_ORDER = ["h1", "h2", "h3", "h4"]

header_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
        ("####", "h4"),
    ]
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    separators=["\n\n", "\n", " "],
    strip_whitespace=True,
)


def chunk_markdown_for_rag(markdown: str, source: str) -> List[Document]:
    chunks: List[Document] = []

    for doc in header_splitter.split_text(markdown):
        body = doc.page_content.strip()

        if not body:
            continue

        meta = dict(doc.metadata or {})

        header_parts = [
            str(meta[key])
            for key in HEADER_ORDER
            if meta.get(key)
        ]
        header_path = " > ".join(header_parts)

        # Split the section body into smaller chunks.
        for piece in text_splitter.split_text(body):
            piece = piece.strip()
            if not piece:
                continue

            # Include the header path in the embedded content.
            content = f"{header_path}\n\n{piece}" if header_path else piece

            chunks.append(
                Document(
                    page_content=content,
                    metadata={
                        **meta,
                        "source": source,
                        "file": os.path.basename(source),
                    },
                )
            )

    return chunks

# fail safe for wrong .env entry
def require(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise SystemExit(f"{name} is not set. Copy .env.example to .env and fill it in.")
    return value
# Setup information statics

load_dotenv()

DB_USER = require("PG_USER")
DB_PASS = require("PG_PASSWORD")
DB_HOST = require("PG_HOST")
DB_PORT = require("PG_PORT")
DB_NAME = require("PG_DB")

EMBEDDING_MODEL = require("EMBEDDING_MODEL")

DATABASE_URL = f"postgresql+psycopg://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

## embedding model
ollamaembeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

### DB organise
def get_stored_dim(db_url: str):
    # Fixed dimension of the embedding column, or None if dimensionless.
    engine = create_engine(db_url)
    with engine.connect() as conn:
        row = conn.execute(text(
            "SELECT a.atttypmod FROM pg_attribute a "
            "JOIN pg_class c ON c.oid = a.attrelid "
            "WHERE c.relname = 'langchain_pg_embedding' AND a.attname = 'embedding'"
        )).fetchone()
    if row is None or row[0] in (None, -1):
        return None          # plain 'vector' → any dimension allowed
    return row[0] - 4        # pgvector stores dim+4 in atttypmod

def hard_reset_pgvector(db_url: str, model: str = None, expected_dim: int = None):
    """FULL RESET: drops ALL collections and the schema tables.
    Run once whenever you switch embedding models, BEFORE re-ingesting.
    Only touches the two langchain-postgres tables — nothing else in the DB.
    """
    stored = get_stored_dim(db_url)
    print(f"Stored column type : vector{'' if stored is None else f'({stored})'}")
    if expected_dim is not None:
        print(f"New model dimension: {expected_dim}")
        if stored is not None and stored != expected_dim:
            print("⚠️ Fixed dimension mismatch → table drop required (proceeding)")

    engine = create_engine(db_url)
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS langchain_pg_embedding;"))
        conn.execute(text("DROP TABLE IF EXISTS langchain_pg_collection;"))
    print("Hard reset done. Tables will be recreated on next add_documents.")

def check_embedding_dim(db_url: str, embeddings):
    """Pre-flight check: model dim vs stored dim. Fails fast with a clear message."""
    q_dim = len(embeddings.embed_query("test"))
    engine = create_engine(db_url)

    with engine.connect() as conn:
        # to_regclass → NULL if the table doesn't exist yet (fresh after hard reset)
        table_exists = conn.execute(text(
            "SELECT to_regclass('public.langchain_pg_embedding') IS NOT NULL"
        )).scalar()

        if not table_exists:
            print(f"No existing schema — fresh start, model dim {q_dim} will be used.")
            return

        # pgvector is not an array type → count commas in the text form
        row = conn.execute(text(
            "SELECT (length(embedding::text) "
            "       - length(replace(embedding::text, ',', ''))) + 1 "
            "FROM langchain_pg_embedding LIMIT 1"
        )).fetchone()

    db_dim = row[0] if row else None

    if db_dim is not None and db_dim != q_dim:
        print(f"Dimension mismatch: DB stores {db_dim}, model produces {q_dim}. ")
        return False
    else:
        print(f"Dimension check passed (model={q_dim}, stored={db_dim or 'empty table'})")
        return True

def drop_collection_safe(db_url: str, collection_name: str):
    #Drops a specific PGVector collection safely by dynamically finding column names.
    engine = create_engine(db_url)
    meta = MetaData()
    
    # Reflect the database schema to find actual column names
    meta.reflect(bind=engine)
    
    table_name = "langchain_pg_collection"
    
    if table_name not in meta.tables:
        print(f"Table '{table_name}' does not exist.")
        return

    # Find the Primary Key column name (could be 'id', 'uuid', etc.)
    table_obj = Table(table_name, meta, autoload_with=engine)
    pk_column = [c.name for c in table_obj.primary_key][0]
    
    with engine.connect() as conn:
        # 3. Query using the dynamic column name to find the Collection ID
        result = conn.execute(
            text(f"SELECT {pk_column} FROM langchain_pg_collection WHERE name = :name"), 
            {"name": collection_name}
        )
        row = result.fetchone()
        
        if not row:
            print(f" Collection '{collection_name}' does not exist. Nothing to drop.")
            return

        cid = row[0]
        
        # 4. Delete associated embeddings (Assuming standard LangChain FK column 'collection_id')
        conn.execute(text("DELETE FROM langchain_pg_embedding WHERE collection_id = :cid"), {"cid": cid})
        
        # 5. Delete the collection itself
        conn.execute(text(f"DELETE FROM langchain_pg_collection WHERE {pk_column} = :cid"), {"cid": cid})
        
        print(f"Successfully dropped collection '{collection_name}' (ID: {cid}).")

def main():
    for item in file_names:
        this_data = file_path+item
        the_file_name =  os.path.splitext(os.path.basename(this_data))[0]

        COLLECTION_NAME = the_file_name

        converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
        )
        result = converter.convert(this_data)

        markdown_content = result.document.export_to_markdown()

        with open("./out/docling/"+ the_file_name +".md", "w", encoding="utf-8") as f:
            f.write(markdown_content)

        document_chunks = chunk_markdown_for_rag(markdown_content, this_data)

        # Chunking info
        print(f" Extracted & Chunked {len(document_chunks)} sections.")

        lengths = [len(d.page_content) for d in document_chunks]

        if lengths:
            print("Number of chunks:", len(lengths))
            print("Min chars:", min(lengths))
            print("Max chars:", max(lengths))
            print("Avg chars:", sum(lengths) // len(lengths))
        else:
            print("No chunks produced")

        buckets = Counter()

        for d in document_chunks:
            n = len(d.page_content)

            if n < 150:
                buckets["<150"] += 1
            elif n < 300:
                buckets["150-300"] += 1
            elif n < 600:
                buckets["300-600"] += 1
            elif n < 1000:
                buckets["600-1000"] += 1
            else:
                buckets[">1000"] += 1

        for k in ["<150", "150-300", "300-600", "600-1000", ">1000"]:
            print(k, buckets[k])

        if(check_embedding_dim(DATABASE_URL, ollamaembeddings)):
            print("Dimension OK, drop collections")
            drop_collection_safe(DATABASE_URL, COLLECTION_NAME)
        else:
            print("Dimension incompatible, DB hard reset")
            hard_reset_pgvector(DATABASE_URL,expected_dim=len(ollamaembeddings.embed_query("test")),)

        vectorstore = PGVector(
            embeddings=ollamaembeddings,
            collection_name=COLLECTION_NAME,
            connection=DATABASE_URL,
            )
        
            # Insert chunks into the database
        print(f"Inserting {len(document_chunks)} chunks into PGVector...")
        vectorstore.add_documents(document_chunks)
        print("Successfully stored in PostgreSQL!")

if __name__ == "__main__":
    main()
