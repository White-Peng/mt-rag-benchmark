#!/usr/bin/env python3
"""Export MT-RAG clapnq retrieval predictions from Milvus.

This helper script reads the official MT-RAG clapnq query file, executes a
Milvus dense or hybrid retrieval against a pre-populated collection, and dumps a
JSONL file that can be fed directly into ``run_retrieval_eval.py``.

Example:
    python export_clapnq.py \
        --query_file human/retrieval_tasks/clapnq/clapnq_questions.jsonl \
        --output_file outputs/clapnq_predictions.jsonl \
        --collection_name clapnq \
        --collection_tag mt-rag-clapnq-elser-512-100-20240503
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import pickle
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
import xxhash
from FlagEmbedding import FlagAutoModel
from pymilvus import AnnSearchRequest, Collection, WeightedRanker, connections
from tqdm import tqdm

LOGGER = logging.getLogger("export_clapnq")

# ---------------------------------------------------------------------------
# Tokenization + BM25 helpers (adapted from the ingestion pipeline)
# ---------------------------------------------------------------------------


def simple_tokenize(text: str) -> List[str]:
    return text.lower().split()


def term_id(term: str) -> int:
    # xxhash keeps compatibility with the ingestion pipeline.
    return xxhash.xxh64(term).intdigest() & 0x7FFFFFFF


def bm25_doc_vector(
    tokens: Sequence[str],
    idf: Dict[str, float],
    avgdl: float,
    k: float = 1.2,
    b: float = 0.75,
) -> Dict[int, float]:
    tf = Counter(tokens)
    dl = len(tokens)
    if dl == 0:
        return {}

    denom = max(avgdl, 1e-9)
    K = k * (1 - b + b * (dl / denom))

    vec: Dict[int, float] = {}
    for term, freq in tf.items():
        idf_val = idf.get(term, 0.0)
        if idf_val <= 0.0:
            continue
        score = idf_val * ((freq * (k + 1)) / (freq + K))
        if score > 0:
            vec[term_id(term)] = float(score)
    return vec


def chunk_text(
    text: str,
    tokenizer,
    max_tokens: int,
    overlap_tokens: int,
    special_buffer: int,
) -> List[str]:
    effective_max = max(max_tokens - special_buffer, 1)
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    token_ids = encoded["input_ids"]
    if not token_ids:
        return []

    if len(token_ids) <= effective_max:
        return [text]

    chunks: List[str] = []
    step = max(effective_max - overlap_tokens, 1)
    for start in range(0, len(token_ids), step):
        end = min(start + effective_max, len(token_ids))
        chunk_ids = token_ids[start:end]
        chunk = tokenizer.decode(chunk_ids, skip_special_tokens=True).strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(token_ids):
            break
    return chunks


def iter_documents(path: Path) -> Iterator[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                LOGGER.warning("Skipping line %d (decode error: %s)", line_no, exc)
                continue

            text = (obj.get("text") or obj.get("content") or "").strip()
            if not text:
                continue

            doc_id = (
                obj.get("id")
                or obj.get("_id")
                or obj.get("document_id")
                or f"auto_{line_no}"
            )
            title = (obj.get("title") or "").strip()
            yield {"id": str(doc_id), "title": title, "text": text}


def iter_chunks(
    path: Path,
    tokenizer,
    max_tokens: int,
    overlap_tokens: int,
    special_buffer: int,
) -> Iterator[Dict[str, str]]:
    for doc in iter_documents(path):
        parts = chunk_text(
            doc["text"], tokenizer, max_tokens, overlap_tokens, special_buffer
        )
        if not parts:
            continue
        if len(parts) == 1:
            yield {"id": doc["id"], "title": doc["title"], "text": parts[0]}
            continue
        for idx, chunk in enumerate(parts):
            chunk_id = f"{doc['id']}_{idx}"
            title = doc["title"]
            chunk_title = f"{title} [part {idx + 1}]" if title else chunk_id
            yield {"id": chunk_id, "title": chunk_title, "text": chunk}


def build_bm25_stats(
    corpus_path: Path,
    tokenizer,
    max_tokens: int,
    overlap_tokens: int,
    special_buffer: int,
) -> Tuple[Dict[str, float], float, int]:
    df = Counter()
    total_len = 0
    doc_count = 0

    LOGGER.info("Scanning corpus %s for BM25 statistics", corpus_path)
    for chunk in tqdm(
        iter_chunks(corpus_path, tokenizer, max_tokens, overlap_tokens, special_buffer),
        desc="Pass 1/1: corpus stats",
    ):
        tokens = simple_tokenize(chunk["text"])
        if not tokens:
            continue
        doc_count += 1
        total_len += len(tokens)
        df.update(set(tokens))

    avgdl = total_len / max(doc_count, 1)
    idf = {
        term: math.log((doc_count - df_t + 0.5) / (df_t + 0.5) + 1.0)
        for term, df_t in df.items()
    }
    LOGGER.info(
        "Finished corpus scan: %d chunks, avgdl %.2f, vocab size %d",
        doc_count,
        avgdl,
        len(idf),
    )
    return idf, avgdl, doc_count


def load_or_build_bm25(
    cache_path: Optional[Path],
    corpus_path: Optional[Path],
    tokenizer,
    max_tokens: int,
    overlap_tokens: int,
    special_buffer: int,
) -> Optional[Tuple[Dict[str, float], float]]:
    if cache_path and cache_path.exists():
        LOGGER.info("Loading BM25 stats cache from %s", cache_path)
        with cache_path.open("rb") as handle:
            data = pickle.load(handle)
        return data.get("idf"), data.get("avgdl")

    if corpus_path and corpus_path.exists():
        idf, avgdl, _ = build_bm25_stats(
            corpus_path, tokenizer, max_tokens, overlap_tokens, special_buffer
        )
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with cache_path.open("wb") as handle:
                pickle.dump({"idf": idf, "avgdl": avgdl}, handle)
            LOGGER.info("Saved BM25 stats cache to %s", cache_path)
        return idf, avgdl

    LOGGER.warning(
        "No BM25 cache or corpus provided; falling back to dense-only retrieval"
    )
    return None


# ---------------------------------------------------------------------------
# Milvus + FlagEmbedding helpers
# ---------------------------------------------------------------------------


def encode_with_bge(model: FlagAutoModel, texts: List[str], batch_size: int) -> np.ndarray:
    embeddings = model.encode(texts, batch_size=batch_size)
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()
    arr = np.asarray(embeddings, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
    return arr / norms


def connect_milvus(host: str, port: str, alias: str, db_name: Optional[str] = None) -> None:
    if connections.has_connection(alias):
        return
    kwargs = {"alias": alias, "host": host, "port": port}
    if db_name:
        kwargs["db_name"] = db_name
    connections.connect(**kwargs)


def run_search(
    collection: Collection,
    dense_vec: np.ndarray,
    sparse_vec: Optional[Dict[int, float]],
    top_k: int,
    alpha: float,
    output_fields: List[str],
) -> List:
    dense_req = AnnSearchRequest(
        data=[dense_vec.tolist()],
        anns_field="dense_embedding",
        param={"metric_type": "IP", "params": {}},
        limit=top_k,
    )

    if sparse_vec and WeightedRanker is not None:
        sparse_req = AnnSearchRequest(
            data=[sparse_vec],
            anns_field="sparse_embedding",
            param={"metric_type": "IP", "params": {}},
            limit=top_k,
        )
        rerank = WeightedRanker(alpha, 1.0 - alpha)
        results = collection.hybrid_search(
            reqs=[dense_req, sparse_req],
            rerank=rerank,
            limit=top_k,
            output_fields=output_fields,
        )
        return list(results[0])

    results = collection.search(
        data=[dense_vec.tolist()],
        anns_field="dense_embedding",
        param={"metric_type": "IP", "params": {}},
        limit=top_k,
        output_fields=output_fields,
    )
    return list(results[0])


# ---------------------------------------------------------------------------
# Query ingestion + writer
# ---------------------------------------------------------------------------


def iter_queries(path: Path, limit: Optional[int] = None) -> Iterator[Tuple[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle, start=1):
            if limit and idx > limit:
                break
            obj = json.loads(line)
            yield obj["_id"], obj["text"]


def export_predictions(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    connect_milvus(args.milvus_host, args.milvus_port, args.milvus_alias, args.milvus_db)
    collection = Collection(args.collection_name, using=args.milvus_alias)
    collection.load()
    LOGGER.info("Loaded collection %s", args.collection_name)

    LOGGER.info("Loading FlagEmbedding model %s", args.model_name)
    model = FlagAutoModel.from_finetuned(
        args.model_name, use_fp16=torch.cuda.is_available()
    )
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        raise RuntimeError("Unable to read tokenizer from FlagAutoModel instance")

    bm25_tuple = load_or_build_bm25(
        cache_path=Path(args.bm25_cache).expanduser() if args.bm25_cache else None,
        corpus_path=Path(args.corpus_jsonl).expanduser()
        if args.corpus_jsonl
        else None,
        tokenizer=tokenizer,
        max_tokens=args.chunk_max_tokens,
        overlap_tokens=args.chunk_overlap_tokens,
        special_buffer=args.chunk_special_buffer,
    )

    idf = avgdl = None
    if bm25_tuple:
        idf, avgdl = bm25_tuple

    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    num_queries = 0

    query_iter = iter_queries(Path(args.query_file), args.query_limit)
    with Path(args.output_file).open("w", encoding="utf-8") as writer:
        for task_id, query in tqdm(query_iter, desc="Queries"):
            dense_vec = encode_with_bge(model, [query], batch_size=args.bge_batch_size)[0]
            sparse_vec = None
            if idf and avgdl:
                sparse_vec = bm25_doc_vector(
                    simple_tokenize(query),
                    idf=idf,
                    avgdl=avgdl,
                    k=args.bm25_k1,
                    b=args.bm25_b,
                )
            hits = run_search(
                collection=collection,
                dense_vec=dense_vec,
                sparse_vec=sparse_vec,
                top_k=args.top_k,
                alpha=args.search_alpha,
                output_fields=args.output_fields,
            )
            contexts = []
            for hit in hits:
                entity = hit.entity
                context = {
                    "document_id": entity.get("id", getattr(hit, "id", None)),
                    "score": float(hit.score),
                }
                if args.include_title:
                    context["title"] = entity.get("title")
                if args.include_text:
                    context["text"] = entity.get("text")
                contexts.append(context)

            record = {
                "task_id": task_id,
                "Collection": args.collection_tag,
                "contexts": contexts,
            }
            writer.write(json.dumps(record, ensure_ascii=False) + "\n")
            num_queries += 1

    LOGGER.info(
        "Finished writing %d queries to %s",
        num_queries,
        args.output_file,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    default_query = Path("human/retrieval_tasks/clapnq/clapnq_questions.jsonl")
    parser.add_argument("--query_file", type=str, default=str(default_query))
    parser.add_argument("--query_limit", type=int, default=None, help="Optional max queries")
    parser.add_argument(
        "--output_file",
        type=str,
        default="outputs/clapnq_predictions.jsonl",
    )

    parser.add_argument(
        "--collection_name", type=str, default=os.getenv("MILVUS_COLLECTION", "clapnq")
    )
    parser.add_argument(
        "--collection_tag",
        type=str,
        default="mt-rag-clapnq-elser-512-100-20240503",
        help="Value for the Collection field in the output JSONL",
    )
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--search_alpha", type=float, default=0.6)
    parser.add_argument("--include_title", action="store_true")
    parser.add_argument("--include_text", action="store_true")

    parser.add_argument("--milvus_host", type=str, default=os.getenv("MILVUS_HOST", "localhost"))
    parser.add_argument("--milvus_port", type=str, default=os.getenv("MILVUS_PORT", "19530"))
    parser.add_argument("--milvus_db", type=str, default=os.getenv("MILVUS_DB"))
    parser.add_argument("--milvus_alias", type=str, default=os.getenv("MILVUS_ALIAS", "clapnq_conn"))

    parser.add_argument("--model_name", type=str, default=os.getenv("BGE_MODEL_NAME", "BAAI/bge-small-en"))
    parser.add_argument("--bge_batch_size", type=int, default=int(os.getenv("BGE_BATCH_SIZE", "32")))

    parser.add_argument(
        "--bm25_cache",
        type=str,
        default=os.getenv("BM25_CACHE", "~/.cache/mt_rag/clapnq_bm25.pkl"),
    )
    parser.add_argument(
        "--corpus_jsonl",
        type=str,
        default=os.getenv("JSONL_PATH"),
        help="Path to corpus JSONL for BM25 stats (optional if cache exists)",
    )
    parser.add_argument("--bm25_k1", type=float, default=float(os.getenv("BM25_K1", "1.2")))
    parser.add_argument("--bm25_b", type=float, default=float(os.getenv("BM25_B", "0.75")))

    parser.add_argument(
        "--chunk_max_tokens",
        type=int,
        default=int(os.getenv("CHUNK_MAX_TOKENS", "512")),
    )
    parser.add_argument(
        "--chunk_overlap_tokens",
        type=int,
        default=int(os.getenv("CHUNK_OVERLAP_TOKENS", "64")),
    )
    parser.add_argument(
        "--chunk_special_buffer",
        type=int,
        default=int(os.getenv("CHUNK_SPECIAL_TOKENS", "2")),
    )

    parser.add_argument(
        "--output_fields",
        nargs="*",
        default=["title", "text", "id"],
        help="Milvus fields to return alongside the embeddings",
    )

    return parser.parse_args()


if __name__ == "__main__":
    export_predictions(parse_args())
