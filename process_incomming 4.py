import os
import json
import requests
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity # pip install skit-learn
import numpy as np
import joblib
from datetime import datetime


def create_embedding(text_list):
    # API call me 'input' me text_list pass kar rahe hain
    r = requests.post("http://localhost:11434/api/embed", json={
        "model": "bge-m3",
        "input": text_list
    })
    embedding = r.json()["embeddings"]
    return embedding


def interface(prompt):
    r = requests.post("http://localhost:11434/api/generate", json={
        "model": "llama2:latest",
        "prompt":prompt, # type: ignore
        "stream":False
    })
    response=r.json()
    return response

a=joblib.load("embedding.joblib") # type: ignore



# User se input le
income_query = input("Ask question: ")

# Input ka embedding generate kare
question_embedding = create_embedding([income_query])[0]

# Similarity calculate kare
similarities = cosine_similarity(np.vstack(a["embedding"]), [question_embedding]).flatten()

# Similarity ko DataFrame me ek column me add kare
a["similarity"] = similarities

# Top 3 results le ke sort kare similarity ke basis pe
top_result = 5
top_chunks = a.sort_values(by="similarity", ascending=False).head(top_result).reset_index(drop=True)

# Output ko clearly print kare index, text, similarity ke sath
print("\nTop matching chunks:")


output=[]
# Loop to prepare top matching chunks
for idx, row in top_chunks.iterrows():
    output.append({  # Now properly indented inside the loop
        "chunk_id": row["chunk_id"],
        "text": row["text"],
        "similarity": round(row["similarity"], 4),
        "start": row["start"],
        "end": row["end"],
        "tag": "Information",  # Add tag here
        "time": datetime.now().isoformat()  # Add current time # type: ignore
    })
    # Properly indented inside the for loop
    print(f"{row['chunk_id']} - {row['text']} (Similarity: {round(row['similarity'], 4)}) - {row['start']}- {row['end']}")


# Save the generated prompt to a text file
with open("prompt.txt", "w") as f:
    json.dump(output,f,indent=4)

print("Top matching chunks saved to 'prompt.txt'.")


context = ""
for chunk in output:
    context += chunk["text"] + "\n"

final_prompt = f"{context}- {income_query}"  # Combine context and question

raw_response = interface(final_prompt)

with open("response.txt", "w") as f:
    f.write(json.dumps(raw_response, indent=4))


