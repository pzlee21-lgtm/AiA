import argparse
import os
import shutil
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from embedding import get_embedding_function
from langchain_chroma import Chroma

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.join(BASE_DIR, "chroma")
DATA_PATH = os.path.join(BASE_DIR, "data")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        print("✨ Clearing Database")
        clear_database()

    documents = load_documents()
    chunks = split_documents(documents)
    add_to_chroma(chunks)


def load_documents():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()


def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)


def tag_section(chunk: Document) -> Document:
    """
    Tag each chunk with the correct source label based on its filename.
    - Renal Dose PDF → 'RENAL'
    - Anticoagulant PDF → 'ANTICOAG'
    - Everything else → 'ADULT'
    """
    source_path = chunk.metadata.get("source", "")
    filename = os.path.basename(source_path).lower()

    if "renal dose" in filename or "renal" in filename:
        chunk.metadata["source"] = "RENAL"
    elif "anticoagulant" in filename or "anticoag" in filename:
        chunk.metadata["source"] = "ANTICOAG"
    else:
        chunk.metadata["source"] = "ADULT"

    return chunk


def add_to_chroma(chunks: list[Document]):
    # Load the existing database
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=get_embedding_function()
    )

    # Calculate Page IDs
    chunks_with_ids = calculate_chunk_ids(chunks)

    # ✅ Tag every chunk with ADULT, RENAL, or ANTICOAG source label
    chunks_with_ids = [tag_section(chunk) for chunk in chunks_with_ids]

    # Add or Update the documents
    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Only add documents that don't exist in the DB
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
    else:
        print("✅ No new documents to add")


def calculate_chunk_ids(chunks):
    """
    Generate unique IDs for each chunk.
    Format: filename:page_number:chunk_index
    Note: source path is saved separately before being overwritten by tag_section()
    """
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Store original path as file_path, keep id separate
        chunk.metadata["file_path"] = source
        chunk.metadata["id"] = chunk_id

    return chunks


def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)


if __name__ == "__main__":
    main()