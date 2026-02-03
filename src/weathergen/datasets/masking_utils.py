"""Utilities for masking validation and logging."""

import logging

import omegaconf

logger = logging.getLogger(__name__)

# Valid strategies and relationships
VALID_STRATEGIES = {"random", "healpix", "cropping_healpix", "forecast", "causal"}
VALID_RELATIONSHIPS = {"independent", "complement", "identity", "subset", "disjoint"}

# Relationships where source config is ignored (derived from target)
SOURCE_IGNORED_RELATIONSHIPS = {"complement", "identity"}

# Invalid combinations (raise error)
INVALID_COMBINATIONS = {
    ("forecast", "complement"),
    ("forecast", "identity"),
    ("forecast", "subset"),
    ("forecast", "disjoint"),
}

# Default relationship per strategy
DEFAULT_RELATIONSHIP = {
    "random": "complement",
    "healpix": "independent",
    "cropping_healpix": "independent",
    "forecast": "independent",
    "causal": "independent",
}


def parse_source_target_mapping(losses, num_sources, num_targets):
    """Extract (relationship, target_idx) for each source from loss config."""
    mapping = {i: (None, min(i, num_targets - 1) if num_targets > 0 else 0)
               for i in range(num_sources)}

    if not losses:
        return mapping

    for _, loss_cfg in losses.items():
        if not hasattr(loss_cfg, "get"):
            continue
        for _, fct_cfg in loss_cfg.get("loss_fcts", {}).items():
            if not hasattr(fct_cfg, "get"):
                continue
            corr = fct_cfg.get("target_source_correspondence")
            if not corr:
                continue

            for tgt_idx, src_spec in corr.items():
                tgt = int(tgt_idx)
                if isinstance(src_spec, (dict, omegaconf.DictConfig)):
                    for src_idx, rel in src_spec.items():
                        mapping[int(src_idx)] = (rel if isinstance(rel, str) else None, tgt)
                elif isinstance(src_spec, (list, omegaconf.ListConfig)):
                    for src_idx in src_spec:
                        mapping[int(src_idx)] = (None, tgt)
                elif isinstance(src_spec, int):
                    mapping[src_spec] = (None, tgt)

    return mapping


def validate_masking_config(source_cfgs, target_cfgs, losses, strict=False):
    """Validate masking strategy/relationship combinations.

    Returns list of warning messages. Raises ValueError for invalid combinations.
    """
    warnings = []
    if not source_cfgs:
        return warnings

    target_cfgs = target_cfgs or source_cfgs
    mapping = parse_source_target_mapping(losses, len(source_cfgs), len(target_cfgs))

    for i, src_cfg in enumerate(source_cfgs):
        src_strat = src_cfg.get("masking_strategy", "random")
        src_conf = src_cfg.get("masking_strategy_config", {})

        rel, tgt_idx = mapping.get(i, (None, 0))
        tgt_cfg = target_cfgs[tgt_idx] if tgt_idx < len(target_cfgs) else {}
        tgt_strat = tgt_cfg.get("masking_strategy", "random")

        using_default = rel is None
        if using_default:
            rel = DEFAULT_RELATIONSHIP.get(src_strat, "independent")

        # Invalid combinations
        if (src_strat, rel) in INVALID_COMBINATIONS:
            raise ValueError(
                f"source[{i}]: '{src_strat}' incompatible with relationship='{rel}'"
            )

        # Confusing combinations (source config ignored)
        if rel in SOURCE_IGNORED_RELATIONSHIPS:
            ignored = [f"{k}={v}" for k, v in src_conf.items()
                       if k in ("rate", "hl_mask", "method")]
            eff = f"~{tgt_strat}_target" if rel == "complement" else f"{tgt_strat}_target"

            msg = (f"source[{i}]: '{src_strat}' + '{rel}' -> {tgt_strat} | "
                   f"config ({', '.join(ignored) or 'all'}) IGNORED, mask={eff}")

            if src_strat == "cropping_healpix":
                msg += " | NOT spatially contiguous!"
            elif src_strat != tgt_strat:
                msg += f" | pattern='{tgt_strat}' not '{src_strat}'"

            if using_default:
                msg += f" | '{rel}' is default for '{src_strat}'"

            warnings.append(msg)
            logger.warning(msg)

            if strict:
                raise ValueError(f"Strict: {msg}")

    return warnings


def log_masking_summary(source_cfgs, target_cfgs, losses):
    """Log effective masking configuration summary."""
    if not source_cfgs:
        return

    target_cfgs = target_cfgs or source_cfgs
    mapping = parse_source_target_mapping(losses, len(source_cfgs), len(target_cfgs))

    logger.info("=" * 50)
    logger.info("MASKING SUMMARY")
    logger.info("-" * 50)

    for i, tgt in enumerate(target_cfgs):
        s, c = tgt.get("masking_strategy", "random"), dict(tgt.get("masking_strategy_config", {}))
        logger.info(f"target[{i}]: {s} {c}")

    logger.info("-" * 50)

    for i, src in enumerate(source_cfgs):
        s = src.get("masking_strategy", "random")
        c = dict(src.get("masking_strategy_config", {}))
        rel, tgt_idx = mapping.get(i, (None, 0))
        tgt_strat = target_cfgs[tgt_idx].get("masking_strategy", "random") if tgt_idx < len(target_cfgs) else "random"

        if rel is None:
            rel = DEFAULT_RELATIONSHIP.get(s, "independent")
            rel_note = "default"
        else:
            rel_note = "explicit"

        used = rel not in SOURCE_IGNORED_RELATIONSHIPS
        if used:
            eff = f"{s}_mask" if rel == "independent" else f"{s}_mask & {'~' if rel == 'disjoint' else ''}{tgt_strat}_target"
        else:
            eff = f"{'~' if rel == 'complement' else ''}{tgt_strat}_target"

        logger.info(f"source[{i}]: {s} {c}")
        logger.info(f"  -> {rel} ({rel_note}) -> target[{tgt_idx}]")
        logger.info(f"  = {eff} | config {'USED' if used else 'IGNORED'}")

    logger.info("=" * 50)
