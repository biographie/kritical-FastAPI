from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import time
from concurrent.futures import ThreadPoolExecutor
from fastembed import SparseTextEmbedding
import voyageai
import os
from typing import List
from transformers import AutoTokenizer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from fastapi.responses import JSONResponse
import numpy as np

VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
DENSE_EMBEDDING_MODEL = "voyageai/voyage-3"
SPARSE_EMBEDDING_MODEL = "prithvida/Splade_PP_en_v1"
app = FastAPI()


class Query(BaseModel):
    text: str


def chunk_text(text, chunk_size):
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        AutoTokenizer.from_pretrained("prithivida/Splade_PP_en_v1"),
        chunk_size=chunk_size,
        chunk_overlap=int(chunk_size / 10),
        add_start_index=True,
        strip_whitespace=True,
    )
    chunked_text = text_splitter.split_text(text)
    return chunked_text


def compute_sparse_vector(text: List[str]):
    model_name = "prithvida/Splade_PP_en_v1"
    # This triggers the model download
    model = SparseTextEmbedding(model_name=model_name, cache_dir="model_cache_dir")
    sparse_embeddings_list = list(model.embed(text))
    return sparse_embeddings_list


def compute_dense_vector(text: List[str]):
    vo = voyageai.Client(api_key=VOYAGE_API_KEY)  # Use the constant directly
    result = vo.embed(text, model="voyage-3", input_type="document")
    return result.embeddings


def compute_embeddings_parallel(text):
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=2) as executor:
        # Submit both tasks
        sparse_future = executor.submit(compute_sparse_vector, text)
        dense_future = executor.submit(compute_dense_vector, text)
        # Get results
        sparse_embeddings = sparse_future.result()
        dense_embeddings = dense_future.result()
    total_time = time.time() - start_time
    return {
        "sparse_embeddings": sparse_embeddings,
        "dense_embeddings": dense_embeddings,
    }


@app.get("/")
async def health_check():
    return {"status": "ok"}


@app.post("/embed")
async def generate_embeddings(query: Query):
    try:
        print("Starting embedding generation...")
        print(f"Processing text: {query.text}")
        embeddings = compute_embeddings_parallel([query.text])

        # Convert SparseEmbedding objects to dictionaries
        response_data = {
            "sparse_embeddings": [
                {"indices": e.indices.tolist(), "values": e.values.tolist()}
                for e in embeddings["sparse_embeddings"]
            ],
            "dense_embeddings": [
                e.tolist() if isinstance(e, np.ndarray) else list(e)
                for e in embeddings["dense_embeddings"]
            ],
        }
        return JSONResponse(content=response_data)
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
