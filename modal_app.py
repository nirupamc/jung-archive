"""Modal deployment entrypoint for the Jung Archive FastAPI backend."""

import os
import modal

app = modal.App("jung-archive-backend")

HF_CACHE = "/root/.cache/huggingface"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .add_local_file("pyproject.toml", "/pyproject.toml", copy=True)
    .add_local_dir("src", "/src", copy=True)
    .workdir("/")
    .pip_install(".[api,rerank]")
    .env(
        {
            "HF_HOME": HF_CACHE,
            "TRANSFORMERS_CACHE": HF_CACHE,
            "SENTENCE_TRANSFORMERS_HOME": HF_CACHE,
            "TOKENIZERS_PARALLELISM": "false",
            "OMP_NUM_THREADS": "1",
            "JUNG_ARCHIVE_DATA_DIR": "/data",
        }
    )
    .run_commands(
        "python -c \""
        "from sentence_transformers import SentenceTransformer, CrossEncoder; "
        "SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); "
        "CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"
        "\""
    )
)

volume = modal.Volume.from_name(
    "jung-archive-data",
    create_if_missing=True,
)

secrets = [
    modal.Secret.from_name("jung-archive-secrets"),
]


@app.function(
    image=image,
    volumes={"/data": volume},
    secrets=secrets,
    cpu=2,
    memory=4096,
    timeout=300,
)
@modal.concurrent(max_inputs=4)
@modal.asgi_app()
def fastapi_app():
    from jung_archive.api.app import app as backend

    data_dir = os.environ.get("JUNG_ARCHIVE_DATA_DIR", "<unset>")
    print(f"[jung-archive] serving with JUNG_ARCHIVE_DATA_DIR={data_dir}")

    return backend