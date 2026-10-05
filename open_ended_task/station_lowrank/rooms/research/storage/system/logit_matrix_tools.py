#!/usr/bin/env python3
"""Utilities for structured low-rank logit-matrix experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
import gzip
import json
import random
import os

import numpy as np
import torch


DEFAULT_MODEL_PATHS = tuple(
    path for path in (
        Path(os.environ["LOWRANK_MODEL_PATH"]) if os.environ.get("LOWRANK_MODEL_PATH") else None,
        Path("hf_download/model/OLMo-1B-hf"),
    ) if path is not None
)
DEFAULT_WIKI_DATA_DIRS = tuple(
    path for path in (
        Path(os.environ["LOWRANK_WIKI_DATA_DIR"]) if os.environ.get("LOWRANK_WIKI_DATA_DIR") else None,
        Path("hf_download/data/olmo-mix-1124-wiki/data/wiki"),
    ) if path is not None
)


@dataclass(frozen=True)
class FragmentSets:
    contexts: list[str]
    interventions: list[str]
    heldout_contexts: list[str]
    heldout_interventions: list[str]
    corrupted_interventions: list[str]

    @property
    def probes(self) -> list[str]:
        return self.interventions

    @property
    def heldout_probes(self) -> list[str]:
        return self.heldout_interventions

    @property
    def corrupted_probes(self) -> list[str]:
        return self.corrupted_interventions


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: str | Path, payload: dict[str, Any]) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def emit_run_summary(summary: dict[str, Any]) -> None:
    print("RUN_SUMMARY_JSON:", json.dumps(summary, sort_keys=True))


def _existing_path(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def normalize_text(text: str) -> str:
    return " ".join(text.split())


def _record_text(record: dict[str, Any]) -> str:
    for key in ("text", "content", "contents", "document"):
        value = record.get(key)
        if isinstance(value, str):
            return value
    return ""


def _load_wiki_text(*, min_chars: int = 500_000) -> str | None:
    for data_dir in DEFAULT_WIKI_DATA_DIRS:
        if not data_dir.exists():
            continue
        parts: list[str] = []
        total = 0
        for path in sorted(data_dir.glob("*.json.gz")):
            with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(record, dict):
                        continue
                    text = normalize_text(_record_text(record))
                    if text:
                        parts.append(text)
                        total += len(text)
                    if total >= min_chars:
                        return " ".join(parts)
        if parts:
            return " ".join(parts)
    return None


def _fallback_corpus_text() -> str:
    return (
        "Language contains recurring patterns at many scales, from short phrases "
        "to long-range topical structure. A model that predicts text may respond "
        "to different contexts through numerical directions that can be compared "
        "across examples. Scientific analysis requires controls, held-out data, "
        "and careful separation between measurement and interpretation. "
    ) * 2000


def sample_text_fragments(
    *,
    count: int,
    min_chars: int = 50,
    max_chars: int = 180,
    seed: int = 0,
) -> list[str]:
    text = _load_wiki_text() or _fallback_corpus_text()
    text = normalize_text(text)
    if len(text) < min_chars:
        raise ValueError("Corpus text is too short for fragment sampling.")

    rng = random.Random(seed)
    output: list[str] = []
    attempts = 0
    while len(output) < count and attempts < count * 200:
        attempts += 1
        length = rng.randint(min_chars, min(max_chars, len(text)))
        start = rng.randint(0, len(text) - length)
        fragment = text[start : start + length].strip()
        if len(fragment) >= min_chars:
            output.append(fragment)
    if len(output) < count:
        raise ValueError(f"Only sampled {len(output)} fragments; need {count}.")
    return output


def make_fragment_sets(
    *,
    n_contexts: int,
    n_interventions: int,
    n_heldout_contexts: int = 16,
    n_heldout_interventions: int = 16,
    seed: int = 0,
) -> FragmentSets:
    total = n_contexts + n_interventions + n_heldout_contexts + n_heldout_interventions
    fragments = sample_text_fragments(count=total, seed=seed)
    contexts = fragments[:n_contexts]
    interventions = fragments[n_contexts : n_contexts + n_interventions]
    heldout_contexts = fragments[
        n_contexts + n_interventions : n_contexts + n_interventions + n_heldout_contexts
    ]
    heldout_interventions = fragments[-n_heldout_interventions:]
    corrupted_interventions = permute_text_tokens(interventions, seed=seed + 17)
    return FragmentSets(
        contexts=contexts,
        interventions=interventions,
        heldout_contexts=heldout_contexts,
        heldout_interventions=heldout_interventions,
        corrupted_interventions=corrupted_interventions,
    )


def permute_text_tokens(texts: list[str], *, seed: int) -> list[str]:
    rng = random.Random(seed)
    lengths = [len(text.split()) for text in texts]
    words = [word for text in texts for word in text.split()]
    rng.shuffle(words)
    output: list[str] = []
    offset = 0
    for length in lengths:
        output.append(" ".join(words[offset : offset + length]))
        offset += length
    return output


def load_model_and_tokenizer():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = _existing_path(DEFAULT_MODEL_PATHS)
    model_name = str(model_path) if model_path is not None else os.environ.get("LOWRANK_MODEL_NAME", "distilgpt2")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, local_files_only=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
        local_files_only=True,
    )
    model.to(device)
    model.eval()
    return model, tokenizer, device


def encode_texts(tokenizer, texts: list[str], *, max_tokens: int = 64) -> list[list[int]]:
    output: list[list[int]] = []
    for text in texts:
        token_ids = tokenizer.encode(text, add_special_tokens=False)[:max_tokens]
        if token_ids:
            output.append([int(token) for token in token_ids])
    if len(output) != len(texts):
        raise ValueError("At least one text fragment tokenized to an empty sequence.")
    return output


def final_position_logits(
    model,
    tokenizer,
    token_sequences: list[list[int]],
    *,
    device: torch.device,
    batch_size: int = 16,
) -> np.ndarray:
    parts: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(token_sequences), batch_size):
            batch = token_sequences[start : start + batch_size]
            encoded = tokenizer.pad(
                {"input_ids": batch},
                padding=True,
                return_attention_mask=True,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            final_indices = attention_mask.sum(dim=1) - 1
            batch_indices = torch.arange(input_ids.shape[0], device=device)
            logits = outputs.logits[batch_indices, final_indices, :]
            parts.append(logits.detach().float().cpu().numpy())
    return np.concatenate(parts, axis=0)


def mean_center_rows(logits: np.ndarray) -> np.ndarray:
    return logits - logits.mean(axis=1, keepdims=True)


def selected_token_ids_for_interventions(
    model,
    tokenizer,
    interventions: list[list[int]],
    *,
    k: int,
    device: torch.device,
    batch_size: int = 16,
) -> list[list[int]]:
    if k <= 0:
        raise ValueError("k must be positive")
    logits = final_position_logits(
        model,
        tokenizer,
        interventions,
        device=device,
        batch_size=batch_size,
    )
    selected = np.argsort(logits, axis=1)[:, -k:][:, ::-1]
    return [[int(token) for token in row] for row in selected]


def build_context_intervention_matrix(
    model,
    tokenizer,
    contexts: list[list[int]],
    interventions: list[list[int]],
    *,
    selected_tokens_by_intervention: list[list[int]] | None = None,
    k: int = 20,
    device: torch.device,
    batch_size: int = 16,
) -> tuple[np.ndarray, list[list[int]]]:
    """Build rows=context fragments and column blocks=intervention-conditioned logits."""

    if selected_tokens_by_intervention is None:
        selected_tokens_by_intervention = selected_token_ids_for_interventions(
            model,
            tokenizer,
            interventions,
            k=k,
            device=device,
            batch_size=batch_size,
        )
    if len(selected_tokens_by_intervention) != len(interventions):
        raise ValueError("selected token metadata must match number of interventions")

    matrix = np.empty(
        (len(contexts), len(interventions) * k),
        dtype=np.float32,
    )
    for intervention_index, intervention in enumerate(interventions):
        selected = selected_tokens_by_intervention[intervention_index]
        if len(selected) != k:
            raise ValueError("each selected token list must have length k")
        composed = [context + intervention for context in contexts]
        logits = mean_center_rows(
            final_position_logits(
                model,
                tokenizer,
                composed,
                device=device,
                batch_size=batch_size,
            )
        )
        col_start = intervention_index * k
        matrix[:, col_start : col_start + k] = logits[:, selected]
    return matrix, selected_tokens_by_intervention


def low_rank_metrics(
    matrix: np.ndarray,
    *,
    n_blocks: int,
    k: int,
    ranks: list[int],
) -> dict[str, Any]:
    if matrix.shape[1] != n_blocks * k:
        raise ValueError("matrix shape is inconsistent with n_blocks and k")
    u, s, vh = np.linalg.svd(matrix, full_matrices=False)
    blocks = matrix.reshape(matrix.shape[0], n_blocks, k)
    rank_metrics = []
    for rank in sorted(set(ranks)):
        usable_rank = min(rank, s.shape[0])
        approx = (u[:, :usable_rank] * s[:usable_rank]) @ vh[:usable_rank, :]
        approx_blocks = approx.reshape(matrix.shape[0], n_blocks, k)
        rank_metrics.append(
            {
                "rank": int(usable_rank),
                "relative_frobenius_error": float(
                    np.linalg.norm(matrix - approx) / max(np.linalg.norm(matrix), 1e-12)
                ),
                "average_topk_kl": average_block_kl(blocks, approx_blocks),
            }
        )
    return {
        "shape": list(matrix.shape),
        "singular_values": s.tolist(),
        "rank_metrics": rank_metrics,
    }


def average_block_kl(original: np.ndarray, approximation: np.ndarray) -> float:
    original_t = torch.from_numpy(original.astype(np.float64, copy=False))
    approximation_t = torch.from_numpy(approximation.astype(np.float64, copy=False))
    original_log_probs = torch.nn.functional.log_softmax(original_t, dim=-1)
    approximation_log_probs = torch.nn.functional.log_softmax(approximation_t, dim=-1)
    kl = original_log_probs.exp() * (original_log_probs - approximation_log_probs)
    return float(kl.sum(dim=-1).mean().item())
