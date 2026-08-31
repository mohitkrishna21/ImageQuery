from image_search_pipeline import (
    load_clip_model,
    index_folder,
    embed_text_query,
    search_images,
)

clip_model, clip_processor = load_clip_model()

table, error = index_folder(
    "data/images_subset",
    clip_model,
    clip_processor,
)

if error is not None:
    print(error)
    exit()

TEST_QUERIES = [
    "chihuahua",
    "beagle",
    "boxer",
    "pug",
    "yorkshire_terrier",
    "great_pyrenees",
    "havanese",
    "german_shorthaired",
    "saint_bernard",
    "shiba_inu",
]

top_k = 5
precision_scores = []

for query in TEST_QUERIES:
    correct = 0

    emb_query = embed_text_query(
        query,
        clip_model,
        clip_processor,
    )

    result = search_images(
        emb_query,
        table,
        top_k,
    )

    for item in result:
        if query.lower() in item["path"].lower():
            correct += 1

    precision_score = correct / top_k
    precision_scores.append(precision_score)

    print(
        f"{query}: Precision@{top_k} = "
        f"{precision_score:.2f}"
    )

avg = sum(precision_scores) / len(precision_scores)

print("Avg Precision:", avg)