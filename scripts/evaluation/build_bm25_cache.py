#!/usr/bin/env python3
"""Offline BM25 cache builder for MT-RAG datasets.

This utility scans a JSONL corpus (same format as the MT-RAG collections),
reproduces the chunking logic used during Milvus ingestion, and serializes the
BM25 statistics (idf + avgdl) into a pickle file. The resulting cache can be
passed to ``export_clapnq.py`` via ``--bm25_cache`` so that retrieval export
runs without re-reading the large source corpus.

Example
-------
python scripts/evaluation/build_bm25_cache.py \
    --input_jsonl /data/clapnq.jsonl \
    --output_cache ~/.cache/mt_rag/clapnq_bm25.pkl \
    --model_name BAAI/bge-small-en
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
from typing import Dict, Iterator, List, Sequence, Tuple

from FlagEmbedding import FlagAutoModel
from tqdm import tqdm

LOGGER = logging.getLogger("build_bm25_cache")


# ---------------------------------------------------------------------------
# Helpers (mirrors scripts/evaluation/export_clapnq.py)
# ---------------------------------------------------------------------------

def simple_tokenize(text: str) -> List[str]:
    return text.lower().split()


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
) -> Iterator[str]:
    for doc in iter_documents(path):
        parts = chunk_text(
            doc["text"], tokenizer, max_tokens, overlap_tokens, special_buffer
        )
        if not parts:
            continue
        for chunk in parts:
            yield chunk


def compute_bm25_stats(
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
        tokens = simple_tokenize(chunk)
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input_jsonl",
        type=str,
        required=True,
        help="Path to the MT-RAG corpus JSONL (e.g., clapnq.jsonl)",
    )
    parser.add_argument(
        "--output_cache",
        type=str,
        default=os.path.expanduser("~/.cache/mt_rag/clapnq_bm25.pkl"),
        help="Where to store the pickle cache",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=os.getenv("BGE_MODEL_NAME", "BAAI/bge-small-en"),
        help="FlagEmbedding model used to obtain tokenizer",
    )
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    corpus_path = Path(args.input_jsonl).expanduser()
    if not corpus_path.exists():
        raise FileNotFoundError(f"Input JSONL not found: {corpus_path}")

    LOGGER.info("Loading FlagEmbedding model %s", args.model_name)
    model = FlagAutoModel.from_finetuned(args.model_name)
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        raise RuntimeError("FlagAutoModel does not expose tokenizer")

    idf, avgdl, doc_count = compute_bm25_stats(
        corpus_path=corpus_path,
        tokenizer=tokenizer,
        max_tokens=args.chunk_max_tokens,
        overlap_tokens=args.chunk_overlap_tokens,
        special_buffer=args.chunk_special_buffer,
    )

    cache_path = Path(args.output_cache).expanduser()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "idf": idf,
        "avgdl": avgdl,
        "doc_count": doc_count,
        "meta": {
            "model_name": args.model_name,
            "chunk_max_tokens": args.chunk_max_tokens,
            "chunk_overlap_tokens": args.chunk_overlap_tokens,
            "chunk_special_buffer": args.chunk_special_buffer,
        },
    }
    with cache_path.open("wb") as handle:
        pickle.dump(payload, handle)
    LOGGER.info("Saved BM25 cache to %s", cache_path)


if __name__ == "__main__":
    main()
