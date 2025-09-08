import os
import json
import requests
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import joblib

def create_embedding(text_list):
    # API call me 'input' me text_list pass kar rahe hain
    r = requests.post("http://localhost:11434/api/embed", json={
        "model": "bge-m3",
        "input": text_list
    })
    embedding = r.json()["embeddings"]
    return embedding

# JSON files ka list
json_files = os.listdir("jsons")

my_dict = []
chunks_id = 0

# Sabhi JSON files ko read kar ke embeddings banaye
for json_file in json_files:
    with open(os.path.join("jsons", json_file), "r") as f:
        content = json.load(f)
        print(f"Creating embeddings for {json_file}")
        
        # Sab chunks ke text ka list banaye aur uska embedding le
        texts = [chunk["text"] for chunk in content["chunks"]]
        embeddings = create_embedding(texts)

        # Har chunk me chunk_id aur embedding add kare
        for chunk, embedding in zip(content["chunks"], embeddings):
            chunk["chunk_id"] = chunks_id
            chunk["embedding"] = embedding
            chunks_id += 1
            my_dict.append(chunk)

# DataFrame banaye
a = pd.DataFrame.from_records(my_dict)
print("DataFrame with chunks and embeddings:")
print(a.head())



joblib.dump(a,"embedding.joblib")








