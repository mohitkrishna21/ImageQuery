from transformers import CLIPModel, CLIPProcessor
import os
from PIL import Image
import torch
import lancedb
import csv
import datetime

LOG_PATH="logs/search_log.csv"
LOG_HEADERS= ["timestamp", "query", "latency", "top_results"]


def load_clip_model():
    clip_model=CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor=CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    return clip_model,clip_processor

def index_folder(folder_path,clip_model,clip_processor):
    files_list=os.listdir(folder_path)
    images=[]
    for file in files_list:
        if file.lower().endswith((".png",".jpg",".jpeg")):
            images.append(file)

    loaded_images=[]
    for file_name in images:
        full_path= os.path.join(folder_path, file_name)
        try:
           image=Image.open(full_path).convert("RGB")
        except Exception as e:
            print(f"skipped {file_name}: {e}")
            continue
        loaded_images.append((full_path, image))

    if not loaded_images:
     return None
    
    records=[]
    for full_path,image in loaded_images:
        
        image_inputs = clip_processor(images=image, return_tensors="pt")
        pixels=image_inputs["pixel_values"]

        with torch.no_grad():
            pooler_output = clip_model.vision_model(pixel_values=pixels).pooler_output

            image_embedding = clip_model.visual_projection(pooler_output)
            normalized_image = torch.nn.functional.normalize(image_embedding,dim=-1)
            image_array=normalized_image.squeeze().numpy()

        records.append({"path": full_path,"vector":image_array})

    db=lancedb.connect("./lancedb_data")
    lancedb_table=db.create_table("images", data=records, mode="overwrite")

    return lancedb_table

def embed_text_query(query,clip_model,clip_processor):
    query_inputs = clip_processor(text=[query], return_tensors="pt", padding=True)
    input_ids = query_inputs["input_ids"]

    with torch.no_grad():
        pooler_output = clip_model.text_model(input_ids=input_ids).pooler_output
        query_embedding = clip_model.text_projection(pooler_output)
        normalized_query = torch.nn.functional.normalize(query_embedding,dim=-1)
        query_array=normalized_query.squeeze().numpy()

    return query_array

def search_images(query_embedding,table,top_k=5):
    results = table.search(query_embedding).limit(top_k).to_list()

    return results

def log_query(query,latency,top_results):
    os.makedirs("logs", exist_ok=True)

    if not os.path.exists(LOG_PATH):
        with open(LOG_PATH, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(LOG_HEADERS)

    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            datetime.datetime.now().isoformat(),
            query,
            latency,
            top_results
        ])
        



