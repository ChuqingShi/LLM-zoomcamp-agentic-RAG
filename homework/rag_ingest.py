from gitsource import GithubRepositoryDataReader, chunk_documents
from minsearch import Index
from sqlitesearch import TextSearchIndex


def load_data():
    print("[1/3] Initializing GitHub repository reader...")
    reader = GithubRepositoryDataReader(
        repo_owner="DataTalksClub",
        repo_name="llm-zoomcamp",
        commit_id="8c1834d",
        allowed_extensions={"md"},
        filename_filter=lambda path: "/lessons/" in path,
    )

    print("[2/3] Fetching files from GitHub (this may take a moment)...")
    files = reader.read()
    print(f"      Fetched {len(files)} files.")

    print("[3/3] Parsing and chunking documents...")
    documents = []
    for file in files:
        doc = file.parse()
        documents.append(doc)
    chunks = chunk_documents(documents, size=2000, step=1000)
    print(f"      Created {len(chunks)} chunks from {len(documents)} documents.\n")

    return chunks


def build_min_index(chunks=None):
    if chunks is None: 
        chunks = load_data()
    print("Building minsearch index...")
    index = Index(text_fields=["content"], keyword_fields=["filename"])
    index.fit(chunks)
    print("Index ready.\n")
    return index


def build_sqlite_index(chunks=None, db_path="rag_index_text.db"):
    index = TextSearchIndex(text_fields=["content"], keyword_fields=["filename"], db_path=db_path)

    if index.count() == 0:
        if chunks is None:
            chunks = load_data()
        print(f"Building SQLite index at {db_path}...")
        index.fit(chunks)
        print(f"Indexed {index.count()} chunks.")
    else:
        print(f"Loaded existing SQLite index from {db_path} ({index.count()} chunks).")

    index.close()
    print()
    return index
