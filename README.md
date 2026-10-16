# Multi-Modal RAG Chat with Qdrant, SQLite, and Streamlit

A container-ready Streamlit application that lets you upload PDF
documents and ask questions about their content using a multimodal
Retrieval-Augmented Generation (RAG) pipeline. It extracts and indexes
text, tables, and images, supports follow-up questions, displays source
information, and persists document indexes and chat history.

> **Project status:** The application has been tested by its maintainer
> locally and in a Docker container. External services such as OpenAI
> and Qdrant Cloud require valid credentials and configuration.

## Features

-   **Multimodal PDF ingestion** --- extracts text, tables, and images
    from uploaded PDFs using Unstructured.
-   **Content summarization** --- creates summaries of text, tables, and
    images to improve retrieval.
-   **Semantic search with Qdrant** --- stores summary embeddings in
    Qdrant, using a separate collection for each PDF.
-   **Persistent document storage with SQLite** --- stores original
    extracted content, metadata, document references, and ingestion
    status.
-   **Conversational Q&A** --- supports follow-up questions using chat
    history and a history-aware retriever.
-   **Source information** --- shows source types and available page
    numbers for retrieved context.
-   **Per-PDF chat history** --- persists messages separately for each
    processed PDF.
-   **Document management** --- select a processed PDF, clear its chat,
    or delete its stored data through the interface.
-   **Docker support** --- includes a Dockerfile and `.dockerignore` for
    containerized deployment.
-   **Configurable storage** --- supports Qdrant Cloud or local embedded
    Qdrant, with persistent application data under `rag_storage/`.

## Architecture

``` text
                       ┌─────────────────────────┐
                       │     Streamlit UI        │
                       │ Upload PDF / Ask Qs     │
                       └────────────┬────────────┘
                                    │
                                    ▼
                       ┌─────────────────────────┐
                       │   Unstructured PDF      │
                       │ text / tables / images  │
                       └────────────┬────────────┘
                                    │
                                    ▼
                       ┌─────────────────────────┐
                       │ Summarization with      │
                       │ OpenAI chat model       │
                       └────────────┬────────────┘
                                    │
                     ┌──────────────┴──────────────┐
                     ▼                             ▼
          ┌────────────────────┐       ┌────────────────────┐
          │ Qdrant             │       │ SQLite + files     │
          │ Summary embeddings │       │ Original content,  │
          │ and semantic index │       │ metadata, chat     │
          └──────────┬─────────┘       └────────────────────┘
                     │
                     ▼
          ┌────────────────────┐
          │ MultiVectorRetriever│
          │ + chat-aware search │
          └──────────┬─────────┘
                     │
                     ▼
          ┌────────────────────┐
          │ Multimodal prompt  │
          │ + OpenAI model     │
          └──────────┬─────────┘
                     ▼
               Answer + sources
```

### Processing flow

1.  Upload a PDF through the Streamlit interface.
2.  The application calculates a content hash and uses it as the PDF's
    stable identifier.
3.  Unstructured processes the PDF and extracts text, tables, and
    available images.
4.  The application creates text, table, and image summaries using the
    configured OpenAI chat model.
5.  Summary embeddings are stored in Qdrant. A separate collection is
    used for each PDF.
6.  Original extracted content, metadata, PDF files, image files, and
    conversation messages are persisted using SQLite and the storage
    directory.
7.  When you ask a question, the history-aware retriever uses the
    conversation history to retrieve relevant content.
8.  The application builds a multimodal prompt and returns an answer
    with available source details.

## Technology stack

-   Python 3.11 (Docker image base)
-   Streamlit
-   LangChain and LangChain integrations
-   Unstructured for PDF extraction
-   OpenAI chat model for summarization and question answering
-   OpenAI embeddings for semantic search
-   Qdrant for vector storage
-   SQLite for persistent application records and chat history
-   Pillow and OpenCV for image processing
-   Docker

## Requirements

### For Docker deployment

-   Docker Engine or Docker Desktop
-   An OpenAI API key
-   Either a Qdrant Cloud cluster and API key, or local embedded Qdrant
    storage

### For running locally without Docker

-   Python 3.11 recommended
-   The system dependencies needed by Unstructured and PDF/image
    processing (including Poppler utilities, Tesseract OCR, and
    `libmagic` where applicable)
-   An OpenAI API key
-   Qdrant Cloud, or a writable local path for embedded Qdrant

PDF extraction dependencies can be substantial. For the most
reproducible setup, use the included Dockerfile.

## Configuration

Create a file named `.env` in the project root, alongside `app.py` and
`multimodel_rag.py`.

### Option A: Qdrant Cloud

``` dotenv
OPENAI_API_KEY=replace_with_your_openai_api_key
OPENAI_CHAT_MODEL=gpt-4.1-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

QDRANT_URL=https://your-cluster-endpoint.qdrant.io
QDRANT_API_KEY=replace_with_your_qdrant_api_key
QDRANT_COLLECTION_PREFIX=multimodal_rag

RAG_STORAGE_DIR=./rag_storage
SQLITE_DB_PATH=./rag_storage/rag_app.db
```

Replace the example Qdrant URL and credentials with values from your
Qdrant Cloud dashboard. The application expects both `QDRANT_URL` and
`QDRANT_API_KEY` to be configured for cloud mode.

### Option B: Local embedded Qdrant

``` dotenv
OPENAI_API_KEY=replace_with_your_openai_api_key
OPENAI_CHAT_MODEL=gpt-4.1-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

QDRANT_URL=
QDRANT_API_KEY=
QDRANT_LOCAL_PATH=./rag_storage/qdrant_local
QDRANT_COLLECTION_PREFIX=multimodal_rag

RAG_STORAGE_DIR=./rag_storage
SQLITE_DB_PATH=./rag_storage/rag_app.db
```

Local embedded Qdrant stores its index on disk. Keep the configured path
inside the persistent `rag_storage` directory when using Docker, so the
data survives container replacement.

**Security:** Never commit your real `.env` file, OpenAI key, or Qdrant
API key to GitHub. The project `.gitignore` should exclude `.env` and
`rag_storage/`. If a secret is accidentally pushed, revoke or rotate it.

## Project structure

``` text
multimodal-rag/
├── app.py                  # Streamlit user interface
├── multimodel_rag.py       # PDF ingestion, retrieval, LLM chain, persistence
├── requirements.txt        # Python dependencies
├── Dockerfile              # Container build and Streamlit startup
├── .dockerignore           # Files excluded from Docker build context
├── .gitignore              # Files excluded from Git
├── .env                    # Local secrets/configuration (do not commit)
└── rag_storage/            # Persistent PDFs, images, SQLite, local Qdrant data
```

The `rag_storage/` directory is generated/used at runtime. It should
generally not be committed to the repository.

## Run locally

1.  Clone the repository and move into the project directory.
2.  Create and activate a Python 3.11 virtual environment.
3.  Install the requirements.
4.  Create `.env` using one of the configuration examples above.
5.  Start Streamlit.

Example on Windows PowerShell:

``` powershell
py -3.11 -m venv ragvenv
.\ragvenv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```

Open <http://localhost:8501> in your browser.

On Linux/macOS, the virtual environment commands are:

``` bash
python3.11 -m venv ragvenv
source ragvenv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```

If PDF extraction fails locally because of a missing operating-system
dependency, install the required Unstructured/PDF tools for your
operating system. See the [Unstructured
documentation](https://docs.unstructured.io/) for platform-specific
requirements.

## Run with Docker

Run the following commands from the project root, where the `Dockerfile`
and `requirements.txt` are located.

### 1. Create the persistent data directory

``` powershell
New-Item -ItemType Directory -Force -Path .\rag_storage
```

### 2. Build the image

``` bash
docker build -t multimodal-rag:latest .
```

### 3. Start the container

Windows PowerShell:

``` powershell
docker run -d `
  --name multimodal-rag `
  --restart unless-stopped `
  -p 8501:8501 `
  --env-file .env `
  -v "${PWD}/rag_storage:/app/rag_storage" `
  multimodal-rag:latest
```

Linux/macOS shell:

``` bash
mkdir -p rag_storage

docker run -d \
  --name multimodal-rag \
  --restart unless-stopped \
  -p 8501:8501 \
  --env-file .env \
  -v "$(pwd)/rag_storage:/app/rag_storage" \
  multimodal-rag:latest
```

Open <http://localhost:8501>.

The bind mount maps your local `rag_storage/` directory to
`/app/rag_storage` in the container. This preserves the SQLite database,
PDF and image files, and local embedded Qdrant data when the container
is recreated. If using Qdrant Cloud, the persistent directory still
holds the app's SQLite and file data.

### Useful Docker commands

``` bash
# View running containers
docker ps

# Follow application logs
docker logs -f multimodal-rag

# Stop the container
docker stop multimodal-rag

# Start it again
docker start multimodal-rag

# Remove the container (persistent bind-mounted data remains)
docker rm -f multimodal-rag
```

The Dockerfile defines a health check against Streamlit's health
endpoint. The initial health check may wait for the configured start
period while dependencies and the application start.

## Using the application

1.  Open the Streamlit page.
2.  Upload a PDF file.
3.  Wait for extraction, summarization, and indexing to finish.
4.  Ask a question in the chat input.
5.  Ask follow-up questions; the conversation history is used to
    contextualize retrieval.
6.  Use **Sources** on an assistant response to review available source
    types, page numbers, and previews.
7.  Select another processed PDF from the sidebar to open its
    conversation.
8.  Use **Clear Current PDF Chat** to clear the selected PDF's
    conversation, or **Delete Current PDF** to remove its stored index
    and associated data.

Processing time depends on PDF size, the number of images and tables,
model latency, and service availability.

## Data and persistence

-   **Qdrant:** stores vectorized summaries for semantic retrieval.
-   **SQLite:** stores PDF records, extracted document content and
    metadata, summary records, ingestion job status, conversations, and
    messages.
-   **Filesystem:** stores PDF copies and extracted image files under
    the configured storage directory.
-   **Per-PDF separation:** each PDF has its own Qdrant collection and
    conversation record.

Back up the configured storage directory. If using Qdrant Cloud, also
account for your remote Qdrant collection data when planning backups.
Deleting a PDF through the UI deletes its associated Qdrant collection
and application records.

## Troubleshooting

### The app says `OPENAI_API_KEY` is missing

-   Confirm `.env` is in the project root.
-   Confirm the variable is named exactly `OPENAI_API_KEY`.
-   Restart Streamlit or recreate the container after changing
    environment configuration.
-   Do not include surrounding quotes unless needed by your environment
    format.

### Qdrant Cloud connection fails

-   Verify `QDRANT_URL` and `QDRANT_API_KEY`.
-   Ensure the cluster is active and reachable from the host/container.
-   Confirm that the API key has the required collection access.

### Local Qdrant data does not persist

-   Confirm `QDRANT_URL` and `QDRANT_API_KEY` are empty for local
    embedded mode.
-   Confirm `QDRANT_LOCAL_PATH` points inside the mounted
    `/app/rag_storage` directory when running in Docker.
-   Ensure the host directory is writable by the container process.

### PDF ingestion fails

-   Inspect the Streamlit error and container logs:
    `docker logs -f multimodal-rag`.
-   Check that the PDF is valid and readable.
-   Verify the OpenAI API key, quota, and model access.
-   PDF extraction and OCR rely on system and Python dependencies; use
    the provided Dockerfile as the baseline environment.

### The app starts but cannot be reached

-   Confirm the container is running with `docker ps`.
-   Confirm port mapping `-p 8501:8501`.
-   Open `http://localhost:8501`.
-   Review `docker logs multimodal-rag` for startup errors.

## Limitations and notes

-   The application uses OpenAI-hosted models for embeddings,
    summarization, image descriptions, and question answering; an API
    key and network access are required.
-   Model names are configurable using environment variables. The
    defaults are `gpt-4.1-mini` and `text-embedding-3-small`.
-   PDF extraction quality depends on document quality, layout, OCR, and
    the extraction libraries.
-   This README documents the current implementation and deployment
    workflow; it does not claim a formal security audit,
    production-scale load test, or automated evaluation benchmark.

## Contributing

Contributions, bug reports, and suggestions are welcome. Before
submitting changes:

1.  Keep API keys and private documents out of commits.
2.  Avoid committing `.env`, `rag_storage/`, virtual environments, and
    Python cache files.
3.  Test PDF ingestion, retrieval, follow-up questions, and persistence
    after restarting the app/container.
4.  Include steps to reproduce bugs and relevant error logs, with
    secrets removed.

## License

Add a `LICENSE` file before publishing if you intend to grant others
permission to use, modify, and distribute this project. No license is
specified by this README.
