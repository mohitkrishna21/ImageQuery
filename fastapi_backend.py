from fastapi import FastAPI
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
import time
from image_search_pipeline import (
    load_clip_model,
    index_folder,
    embed_text_query,
    search_images,
    log_query
)

app = FastAPI()


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    with open("index.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return html_content


@app.get("/image")
async def get_image(path: str):
    return FileResponse(path)


clip_model, clip_processor = load_clip_model()
table = None


class IndexRequest(BaseModel):
    folder_path: str


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


@app.post("/index")
async def index_images(request: IndexRequest):
    global table

    result = index_folder(
        request.folder_path,
        clip_model,
        clip_processor
    )

    if result is None:
        return {"message": "Folder is empty"}

    table = result
    return {"message": "Indexed Successfully"}


@app.post("/search")
async def search_images_endpoint(request: SearchRequest):
    if table is None:
        return {"message": "Folder is empty"}

    start_time = time.time()

    query_vector = embed_text_query(
        request.query,
        clip_model,
        clip_processor
    )

    result = search_images(
        query_vector,
        table,
        request.top_k
    )

    latency = time.time() - start_time
    log_query(request.query, latency, result)

    return result