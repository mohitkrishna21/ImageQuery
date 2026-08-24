from image_search_pipeline import load_clip_model, index_folder, embed_text_query, search_images
import os

clip_model, clip_processor = load_clip_model()

table = index_folder("data/images_subset", clip_model, clip_processor)

if table is None:
    print("Folder is empty")
    exit()

TEST_QUERIES = ["chihuahua", "beagle", "boxer", "pug", "yorkshire_terrier", "great_pyrenees", "havanese", "german_shorthaired", "saint_bernard", "shiba_inu"]

top_k = 5
precision_scores = []

for query in TEST_QUERIES:
    correct = 0
    emd_query = embed_text_query(query,clip_model,clip_processor)
    result = search_images(emd_query, table, top_k)

    for item in result:
        if query.lower() in item["path"].lower():
            correct +=1

    precision_score = correct/top_k
    precision_scores.append(precision_score)

avg = sum(precision_scores)/len(precision_scores)
print("Avg Precision :",avg)