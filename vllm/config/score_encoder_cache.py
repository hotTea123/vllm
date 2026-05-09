from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ScoreEncoderCacheConfig:
    """Configuration for score-based encoder cache eviction policy."""

    max_clock: int = 15
    clock_decay_every: int = 64
    watermark: float = 0.2
    promote_percentile: float = 0.2


@dataclass
class EncoderCacheManagerConfig:
    """Configuration for pluggable encoder cache hierarchy + policy.

    Backward compatibility:
    - If ``score_encoder_cache_config.enabled`` is true and this config is not
      provided, defaults to ``dram_hbm`` hierarchy + ``score`` policy.
    - Existing score policy knobs are still accepted under
      ``score_encoder_cache_config``.
    """

    enabled: bool = False
    hierarchy_type: str = "dram_hbm"
    eviction_policy: str = "score"
    cpu_cache_slots: int = 100 * 1024 * 10
    default_write_cache: str = "dram"
    enable_promotion: bool = True
    enable_demotion: bool = True
    score_policy: ScoreEncoderCacheConfig = field(
        default_factory=ScoreEncoderCacheConfig
    )


# Keep backward-compatible alias for existing callsites.
get_score_encoder_cache_config: Any


def _read_legacy_score_config(additional_config: dict[str, Any]) -> dict[str, Any]:
    legacy_cfg = additional_config.get("score_encoder_cache_config", {})
    return legacy_cfg if isinstance(legacy_cfg, dict) else {}


def get_encoder_cache_manager_config(vllm_config: Any) -> EncoderCacheManagerConfig:
    additional_config = vllm_config.additional_config or {}
    manager_cfg = additional_config.get("encoder_cache_manager_config", {})
    manager_cfg = manager_cfg if isinstance(manager_cfg, dict) else {}

    legacy_cfg = _read_legacy_score_config(additional_config)
    legacy_enabled = bool(legacy_cfg.get("enabled", False))

    enabled = bool(manager_cfg.get("enabled", legacy_enabled))
    hierarchy_type = str(manager_cfg.get("hierarchy_type", "dram_hbm"))
    eviction_policy = str(manager_cfg.get("eviction_policy", "score"))

    cpu_cache_slots = int(
        manager_cfg.get("cpu_cache_slots", legacy_cfg.get("cpu_cache_slots", 100 * 1024 * 10))
    )
    default_write_cache = str(manager_cfg.get("default_write_cache", "dram"))
    enable_promotion = bool(manager_cfg.get("enable_promotion", True))
    enable_demotion = bool(manager_cfg.get("enable_demotion", True))

    score_policy_cfg = manager_cfg.get("score_policy", {})
    score_policy_cfg = score_policy_cfg if isinstance(score_policy_cfg, dict) else {}
    score_cfg = ScoreEncoderCacheConfig(
        max_clock=int(score_policy_cfg.get("max_clock", legacy_cfg.get("max_clock", 15))),
        clock_decay_every=int(
            score_policy_cfg.get(
                "clock_decay_every", legacy_cfg.get("clock_decay_every", 64)
            )
        ),
        watermark=float(score_policy_cfg.get("watermark", legacy_cfg.get("watermark", 0.2))),
        promote_percentile=float(
            score_policy_cfg.get(
                "promote_percentile", legacy_cfg.get("promote_percentile", 0.2)
            )
        ),
    )

    cfg = EncoderCacheManagerConfig(
        enabled=enabled,
        hierarchy_type=hierarchy_type,
        eviction_policy=eviction_policy,
        cpu_cache_slots=cpu_cache_slots,
        default_write_cache=default_write_cache,
        enable_promotion=enable_promotion,
        enable_demotion=enable_demotion,
        score_policy=score_cfg,
    )

    if cfg.cpu_cache_slots <= 0:
        raise ValueError("encoder_cache_manager_config.cpu_cache_slots must be > 0")
    if cfg.default_write_cache not in {"dram", "hbm"}:
        raise ValueError(
            "encoder_cache_manager_config.default_write_cache must be one of {'dram', 'hbm'}"
        )
    if not 0.0 <= cfg.score_policy.watermark <= 1.0:
        raise ValueError("score_policy.watermark must be in [0, 1]")
    if not 0.0 <= cfg.score_policy.promote_percentile <= 1.0:
        raise ValueError("score_policy.promote_percentile must be in [0, 1]")
    if cfg.score_policy.clock_decay_every <= 0:
        raise ValueError("score_policy.clock_decay_every must be > 0")

    return cfg


class _LegacyScoreEncoderCacheConfig:
    def __init__(self, manager_cfg: EncoderCacheManagerConfig):
        self.enabled = manager_cfg.enabled
        self.cpu_cache_slots = manager_cfg.cpu_cache_slots
        self.max_clock = manager_cfg.score_policy.max_clock
        self.clock_decay_every = manager_cfg.score_policy.clock_decay_every
        self.watermark = manager_cfg.score_policy.watermark
        self.promote_percentile = manager_cfg.score_policy.promote_percentile



def get_score_encoder_cache_config(vllm_config: Any) -> _LegacyScoreEncoderCacheConfig:
    # Legacy compatibility shim.
    return _LegacyScoreEncoderCacheConfig(get_encoder_cache_manager_config(vllm_config))
