# dags/build_pinecone_search.py
from __future__ import annotations
import os, time, csv, requests, re
from datetime import datetime, timedelta
from typing import List, Dict

from airflow import DAG
from airflow.decorators import task
from airflow.models import Variable

from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec

# --------------------
# Config
# --------------------
DATA_DIR = "/opt/airflow/data"
RAW_TXT = os.path.join(DATA_DIR, "romeo_juliet.txt")
INPUT_CSV = os.path.join(DATA_DIR, "paragraphs.csv")

MODEL_NAME = "all-MiniLM-L6-v2"   # 384 dims
INDEX_DIM = 384
INDEX_METRIC = "dotproduct"       # we'll normalize embeddings (cosine-equivalent)

# Try both variable names for convenience
def _get_pinecone_key() -> str:
    for name in ("pinecone_api_key", "PINECONE_API_KEY"):
        try:
            v = Variable.get(name)
            return v.strip()  # avoid hidden \r\n in headers
        except Exception:
            continue
    raise ValueError("Set Airflow Variable 'pinecone_api_key' (or 'PINECONE_API_KEY').")

def _clean_index_name(raw: str) -> str:
    # Pinecone allows only [a-z0-9-]
    name = (raw or "").strip().lower().replace("_", "-")
    name = re.sub(r"[^a-z0-9-]+", "-", name).strip("-")
    return name or "class-demo-idx"

DEFAULT_INDEX = Variable.get("PINECONE_INDEX_NAME", default_var="class_demo_idx")

# --------------------
# DAG defaults (match Medium-style)
# --------------------
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="build_pinecone_search",
    description="Build a Shakespeare (Romeo & Juliet) semantic search index in Pinecone",
    start_date=datetime(2025, 4, 1),
    schedule_interval=timedelta(days=7),  # similar to Medium DAG cadence
    catchup=False,
    default_args=default_args,
    tags=["pinecone", "nlp", "search-engine"],
) as dag:
    """
    DAG to download Romeo & Juliet, chunk it, create a Pinecone index,
    embed with SentenceTransformers, upsert, and run a sample query.
    """

    @task
    def download_data() -> str:
        """Download the raw text to /opt/airflow/data/romeo_juliet.txt"""
        os.makedirs(DATA_DIR, exist_ok=True)
        url = "https://www.gutenberg.org/cache/epub/1513/pg1513.txt"
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        with open(RAW_TXT, "w", encoding="utf-8") as f:
            f.write(r.text)
        # Quick line count for logs/screenshot
        with open(RAW_TXT, "r", encoding="utf-8", errors="ignore") as f:
            lc = sum(1 for _ in f)
        print(f"Downloaded Romeo & Juliet with {lc} lines → {RAW_TXT}")
        return RAW_TXT

    @task
    def process_data(raw_path: str) -> str:
        """Split into paragraphs and write a CSV {id,text} for embedding"""
        with open(raw_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

        paragraphs = [p.strip().replace("\r", " ") for p in text.split("\n\n")]
        paragraphs = [p for p in paragraphs if len(p.split()) > 8]  # keep non-trivial chunks

        with open(INPUT_CSV, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["id", "text"])
            for i, p in enumerate(paragraphs):
                w.writerow([f"rj-{i:05d}", p])

        print(f"Wrote {len(paragraphs)} paragraphs → {INPUT_CSV}")
        return INPUT_CSV

    @task
    def create_index() -> str:
        """Create (if missing) a Pinecone serverless index and wait until ready"""
        api_key = _get_pinecone_key()
        pc = Pinecone(api_key=api_key)

        # sanitize name to meet Pinecone rules
        index_name = _clean_index_name(DEFAULT_INDEX)

        if not pc.has_index(index_name):
            pc.create_index(
                name=index_name,  # <-- fixed typo (was index-name)
                dimension=INDEX_DIM,
                metric=INDEX_METRIC,
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            # Wait for the index to be ready
            while True:
                desc = pc.describe_index(index_name)
                status = getattr(desc, "status", {})
                if status and status.get("ready"):
                    break
                time.sleep(1)

        print(f"Index ready: {index_name}")
        return index_name

    @task
    def embed_and_upsert(index_name: str, input_csv_path: str) -> str:
        """Embed CSV rows and upsert into Pinecone"""
        api_key = _get_pinecone_key()
        pc = Pinecone(api_key=api_key)
        index = pc.Index(index_name)

        # Load rows
        rows: List[Dict] = []
        with open(input_csv_path, encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                rows.append(row)

        # Load embedding model
        model = SentenceTransformer(MODEL_NAME)

        # Encode & upsert in batches
        batch_size = 128
        total = len(rows)
        for start in range(0, total, batch_size):
            chunk = rows[start : start + batch_size]
            texts = [c["text"] for c in chunk]
            # normalize so dotproduct ≈ cosine
            vecs = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)

            vectors = []
            for item, vec in zip(chunk, vecs):
                vectors.append({
                    "id": item["id"],
                    "values": vec.tolist(),
                    "metadata": {"source": "rj", "n_tokens": len(item["text"].split())},
                })

            index.upsert(vectors=vectors)
            print(f"Upserted {len(vectors)} vectors [{start+1}-{start+len(vectors)} / {total}]")

        return f"Upserted {total} paragraphs into {index_name}"

    @task
    def run_search(index_name: str, query: str = "love at night on the balcony") -> str:
        """Encode a query and print top-K hits"""
        api_key = _get_pinecone_key()
        pc = Pinecone(api_key=api_key)
        index = pc.Index(index_name)

        model = SentenceTransformer(MODEL_NAME)
        qvec = model.encode([query], normalize_embeddings=True)[0].tolist()

        res = index.query(vector=qvec, top_k=5, include_metadata=True)
        print("=== QUERY:", query)
        for m in res.matches:
            print(f"score={m.score:.4f} id={m.id} meta={m.metadata}")
        return "done"

    # Task wiring (TaskFlow style)
    raw = download_data()
    prepared = process_data(raw)
    idx = create_index()
    _ = embed_and_upsert(idx, prepared)
    _ = run_search(idx)
