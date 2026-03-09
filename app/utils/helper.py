import unicodedata
import re
from dataclasses import dataclass
from typing import List, Tuple
from collections import defaultdict
from ingestion.embedder import MetaDataSchema
import numpy as np

def clean_text(text:str)->str:
    'The function cleans text, converts it into form which can be easily chunked'
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()



def group_embeddings_by_user(
    embeddings: Tuple[np.ndarray, List]
) -> List[Tuple[np.ndarray, List]]: # Assuming ChunkRecord is defined globally

    # Unpack 2 items instead of 3
    vectors, metadatas = embeddings

    if not isinstance(vectors, np.ndarray):
        raise TypeError("Vectors must be a numpy ndarray")

    # Removed texts from length validation
    if len(vectors) != len(metadatas):
        raise ValueError(
            "Mismatch between vectors and metadata lengths"
        )

    if len(vectors) == 0:
        return []

    user_to_indices = defaultdict(list)

    for idx, meta in enumerate(metadatas):
        user_to_indices[meta.user_id].append(idx)

    grouped_batches = []

    for user_id, indices in user_to_indices.items():
        user_vectors = vectors[indices]

        user_chunks = [
            MetaDataSchema(
                user_id=metadatas[i].user_id,
                pdf_id=metadatas[i].pdf_id,
                page_no=metadatas[i].page_no,
                chunk_text=metadatas[i].chunk_text, # Extracted from metadata
            )
            for i in indices
        ]

        grouped_batches.append((user_vectors, user_chunks))

    return grouped_batches
def cosine_similarity(a, b):
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)

    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0

    score = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return float(score)

def is_noise(text: str) -> bool:
    words = text.split()
    if len(words) < 20:
        return True
    if text.isupper():
        return True
    digit_ratio = sum(c.isdigit() for c in text) / max(len(text), 1)
    if digit_ratio > 0.3:
        return True
    return False