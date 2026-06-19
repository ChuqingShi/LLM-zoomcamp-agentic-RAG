from gitsource import GithubRepositoryDataReader, chunk_documents
from minsearch import Index


def build_index():
    print("[1/4] Initializing GitHub repository reader...")
    reader = GithubRepositoryDataReader(
        repo_owner="DataTalksClub",
        repo_name="llm-zoomcamp",
        commit_id="8c1834d",
        allowed_extensions={"md"},
        filename_filter=lambda path: "/lessons/" in path,
    )

    print("[2/4] Fetching files from GitHub (this may take a moment)...")
    files = reader.read()
    print(f"      Fetched {len(files)} files.")

    print("[3/4] Parsing and chunking documents...")
    documents = []
    for file in files:
        doc = file.parse()
        documents.append(doc)
    chunks = chunk_documents(documents, size=2000, step=1000)
    print(f"      Created {len(chunks)} chunks from {len(documents)} documents.")

    print("[4/4] Building search index...")
    index = Index(text_fields=["content"], keyword_fields=["filename"])
    index.fit(chunks)
    print("      Index ready.\n")

    return index
