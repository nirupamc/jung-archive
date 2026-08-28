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
            # Keep model download output ASCII-safe in the build container.
            "HF_HUB_DISABLE_PROGRESS_BARS": "1",
            "TQDM_DISABLE": "1",
            "PYTHONIOENCODING": "utf-8",
            "JUNG_ARCHIVE_DATA_DIR": "/data",
        }
    )
    .run_commands(
        # Bake models into the image. Output is redirected: transformers/HF
        # print Unicode (checkmarks) that the build log stream can't encode,
        # and the weights are cached into the image layer either way.
        "TRANSFORMERS_VERBOSITY=error HF_HUB_VERBOSITY=error "
        "python -c \""
        "from sentence_transformers import SentenceTransformer, CrossEncoder; "
        "SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); "
        "CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"
        "\" > /dev/null 2>&1"
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