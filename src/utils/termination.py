"""Termination utilities for distance-based stopping criteria."""

from __future__ import annotations

from typing import Any, Dict

import mlx.core as mx


def resolve_termination_settings(cfg: Any | None) -> Dict[str, Any]:
    if cfg is None:
        return {
            "mode": "head",
            "distance_latent": "processed",
            "distance_type": "mean_l2",
            "distance_threshold": 0.01,
            "distance_signal": True,
        }

    return {
        "mode": getattr(cfg, "termination_mode", "head"),
        "distance_latent": getattr(cfg, "termination_distance_latent", "processed"),
        "distance_type": getattr(cfg, "termination_distance", "mean_l2"),
        "distance_threshold": getattr(cfg, "termination_distance_threshold", 0.01),
        "distance_signal": getattr(cfg, "termination_distance_signal", True),
    }


def needs_aux_latents(settings: Dict[str, Any]) -> bool:
    return settings["mode"] == "distance" and settings["distance_latent"] != "processed"


def init_previous_latent(prev_latent: mx.array | None, current_latent: mx.array) -> mx.array:
    if prev_latent is None:
        return mx.zeros_like(current_latent)
    return prev_latent


def get_distance_latent(
    settings: Dict[str, Any], processed_embeddings: mx.array, aux: Dict[str, mx.array] | None
) -> mx.array:
    latent_key = settings["distance_latent"]
    if latent_key == "processed":
        return processed_embeddings
    if aux is None:
        raise ValueError(
            f"Distance latent '{latent_key}' requested but auxiliary latents are unavailable."
        )
    if latent_key == "encoded":
        if "encoded" not in aux:
            raise ValueError("Encoded latents requested but not available.")
        return aux["encoded"]
    if latent_key == "encoded_bfs":
        if "bfs_encoded" not in aux:
            raise ValueError("BFS encoded latents requested but not available.")
        return aux["bfs_encoded"]
    if latent_key == "encoded_bf":
        if "bf_encoded" not in aux:
            raise ValueError("BF encoded latents requested but not available.")
        return aux["bf_encoded"]
    if latent_key == "encoded_prim":
        if "prim_encoded" not in aux:
            raise ValueError("Prim encoded latents requested but not available.")
        return aux["prim_encoded"]
    raise ValueError(
        "Unknown termination_distance_latent. "
        "Expected one of: processed, encoded, encoded_bfs, encoded_bf, encoded_prim. "
        f"Got: {latent_key}"
    )


def compute_latent_distance(
    prev_latent: mx.array, current_latent: mx.array, distance_type: str
) -> mx.array:
    diff = current_latent - prev_latent
    if distance_type == "l2":
        return mx.sqrt(mx.sum(diff * diff))
    if distance_type == "mean_l2":
        per_node = mx.sqrt(mx.sum(diff * diff, axis=1))
        return mx.mean(per_node)
    if distance_type == "l1":
        return mx.sum(mx.abs(diff))
    if distance_type == "mse":
        return mx.mean(diff * diff)
    raise ValueError(f"Unknown distance type: {distance_type}")


def compute_distance_termination_logits(
    settings: Dict[str, Any],
    prev_latent: mx.array | None,
    current_latent: mx.array,
) -> Dict[str, mx.array]:
    prev_latent = init_previous_latent(prev_latent, current_latent)
    distance = compute_latent_distance(prev_latent, current_latent, settings["distance_type"])
    threshold = mx.array(settings["distance_threshold"])
    logit = threshold - distance
    return {"bf": logit, "bfs": logit, "prim": logit}
