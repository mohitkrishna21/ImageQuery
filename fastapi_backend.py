import os
import time

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel

from image_search_pipeline import (
    load_clip_model,
    index_folder,
    embed_text_query,
    search_images,
    log_query,
)


app = FastAPI()


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    with open("index.html", "r", encoding="utf-8") as f:
        html_content = f.read()

    return html_content


# Load CLIP model once when the application starts
clip_model, clip_processor = load_clip_model()

# Stores the current LanceDB image table
table = None

# Stores the canonical path of the successfully indexed folder
indexed_root = None


class IndexRequest(BaseModel):
    folder_path: str


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


@app.get("/image")
async def get_image(path: str):
    global indexed_root

    # Do not serve any files before a folder has been indexed
    if indexed_root is None:
        raise HTTPException(
            status_code=403,
            detail="No image folder has been indexed",
        )

    # Resolve ../ components and symbolic links
    resolved_path = os.path.realpath(path)

    try:
        # Ensure the requested file is actually inside
        # the currently indexed directory
        common_path = os.path.commonpath(
            [resolved_path, indexed_root]
        )

        if common_path != indexed_root:
            raise HTTPException(
                status_code=403,
                detail="Access to this file is not allowed",
            )

    except ValueError:
        # Can happen on Windows when paths are on
        # different drives
        raise HTTPException(
            status_code=403,
            detail="Access to this file is not allowed",
        )

    # Make sure the resolved path points to a real file
    if not os.path.isfile(resolved_path):
        raise HTTPException(
            status_code=404,
            detail="Image not found",
        )

    return FileResponse(resolved_path)


@app.post("/index")
async def index_images(request: IndexRequest):
    global table, indexed_root

    # Canonicalize the requested folder path
    resolved_folder = os.path.realpath(
        request.folder_path
    )

    result, error = index_folder(
        resolved_folder,
        clip_model,
        clip_processor,
    )

    # Invalid path, unreadable directory, or empty folder
    if error is not None:
        return {
            "message": error
        }

    # Only replace the current table/root after
    # indexing succeeds
    table = result
    indexed_root = resolved_folder

    return {
        "message": "Indexed Successfully"
    }


@app.post("/search")
async def search_images_endpoint(request: SearchRequest):
    if table is None:
        return {
            "message": "Folder is empty"
        }

    start_time = time.time()

    query_vector = embed_text_query(
        request.query,
        clip_model,
        clip_processor,
    )

    result = search_images(
        query_vector,
        table,
        request.top_k,
    )

    latency = time.time() - start_time

    log_query(
        request.query,
        latency,
        result,
    )

    return result