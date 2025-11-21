import numpy as np
from typing import Tuple, List, Union
from weathergen.datasets.masking import Masker
from weathergen.datasets.batch import ViewMetadata
from weathergen.datasets.ssl_strategies import (
    build_jepa_views,
    build_dino2_views,
    build_ibot_views,
    build_multi_scale_views,
)


def build_views_for_stream(
    masker: Masker,
    num_cells: int,
    teacher_cfg: dict,
    student_cfg: dict,
    relationship: str = "subset",
    ssl_strategy: str | None = None,
) -> Tuple[Union[np.ndarray, List[np.ndarray]], List[np.ndarray], List[ViewMetadata]]:
    """
    Per-stream view construction: teacher + N student keep masks.

    Supports multiple SSL strategies for self-supervised learning:
    - None/"standard": Basic student-teacher with configurable relationship
    - "jepa": JEPA-style (1 global teacher + N local students)
    - "dino2": Dino 2-style (2 global teachers + many local students)
    - "ibot": iBoT-style (global teacher + masked students)
    - "multi_scale": Multi-scale views at different HEALPix levels

    Parameters
    ----------
    masker : Masker
        Instance providing RNG and healpix-level info.
    num_cells : int
        Number of healpix cells at data level.
    teacher_cfg : dict
        Config: {strategy, rate|keep_m, hl_mask, masking_strategy_config, rate_sampling}.
    student_cfg : dict
        Config: {masking_strategy, rate, num_views, hl_mask, masking_strategy_config, rate_sampling}.
    relationship : str
        One of {'subset','disjoint','independent'}. Determines derivation of student masks.
        Only used for standard strategy.
    ssl_strategy : str | None
        SSL strategy to use: None, "standard", "jepa", "dino2", "ibot", "multi_scale".

    Returns
    -------
    teacher_keep_mask : np.ndarray | list[np.ndarray]
        Boolean keep mask(s) for teacher view(s). List for dino2 (2 teachers), otherwise single mask.
    student_keep_masks : list[np.ndarray]
        Boolean keep masks for each student view.
    metadata : list[ViewMetadata]
        Metadata objects (teacher(s) first, then students).

    """
    # Dispatch to appropriate SSL strategy
    if ssl_strategy == "jepa":
        return build_jepa_views(masker, num_cells, teacher_cfg, student_cfg)
    elif ssl_strategy == "dino2":
        return build_dino2_views(masker, num_cells, teacher_cfg, student_cfg)
    elif ssl_strategy == "ibot":
        return build_ibot_views(masker, num_cells, teacher_cfg, student_cfg)
    elif ssl_strategy == "multi_scale":
        return build_multi_scale_views(masker, num_cells, teacher_cfg, student_cfg)
    elif ssl_strategy is not None and ssl_strategy != "standard":
        raise ValueError(
            f"Unknown ssl_strategy: {ssl_strategy}. "
            f"Supported: 'jepa', 'dino2', 'ibot', 'multi_scale', 'standard', or None."
        )

    # Standard strategy (original implementation)
    return _build_standard_views(masker, num_cells, teacher_cfg, student_cfg, relationship)


def _build_standard_views(
    masker: Masker,
    num_cells: int,
    teacher_cfg: dict,
    student_cfg: dict,
    relationship: str = "subset",
) -> Tuple[np.ndarray, List[np.ndarray], List[ViewMetadata]]:
    """
    Standard view construction: teacher + N student keep masks.

    This is the original implementation kept for backward compatibility.

    Parameters
    ----------
    masker : Masker
        Instance providing RNG and healpix-level info.
    num_cells : int
        Number of healpix cells at data level.
    teacher_cfg : dict
        Config: {strategy, rate|keep_m, hl_mask, masking_strategy_config, rate_sampling}.
    student_cfg : dict
        Config: {masking_strategy, rate, num_views, hl_mask, masking_strategy_config, rate_sampling}.
    relationship : str
        One of {'subset','disjoint','independent'}. Determines derivation of student masks.

    Returns
    -------
    teacher_keep_mask : np.ndarray
        Boolean keep mask for teacher view.
    student_keep_masks : list[np.ndarray]
        Boolean keep masks for each student view.
    metadata : list[ViewMetadata]
        Metadata objects (teacher first, then students).

    """
    strat_teacher = teacher_cfg.get("strategy", "random")
    rate_teacher = teacher_cfg.get("rate")
    t_cfg_extra = teacher_cfg.get("masking_strategy_config")

    teacher_keep_mask = masker.generate_cell_keep_mask(
        num_cells=num_cells,
        strategy=strat_teacher,
        rate=rate_teacher,
        masking_strategy_config=t_cfg_extra,
    )

    # Student base masks
    num_views = student_cfg.get("num_views", 1)
    strat_student = student_cfg.get("masking_strategy", student_cfg.get("strategy", "random"))
    rate_student = student_cfg.get("rate")
    s_cfg_extra = student_cfg.get("masking_strategy_config")

    student_keep_masks: List[np.ndarray] = []
    for v in range(num_views):
        base = masker.generate_cell_keep_mask(
            num_cells=num_cells,
            strategy=strat_student,
            rate=rate_student,
            masking_strategy_config=s_cfg_extra,
        )
        if relationship == "subset":
            keep = base & teacher_keep_mask
        elif relationship == "disjoint":
            keep = base & (~teacher_keep_mask)
        else:  # independent
            keep = base
        student_keep_masks.append(keep)

    metadata: List[ViewMetadata] = []
    metadata.append(
        ViewMetadata(
            view_id="teacher_global",
            keep_mask=teacher_keep_mask,
            strategy=strat_teacher,
            healpix_level=masker.healpix_level_data,
            rate=rate_teacher,
            parent_view_id=None,
        )
    )
    for i, m in enumerate(student_keep_masks):
        metadata.append(
            ViewMetadata(
                view_id=f"student_local_{i}",
                keep_mask=m,
                strategy=strat_student,
                healpix_level=masker.healpix_level_data,
                rate=rate_student,
                parent_view_id="teacher_global",
            )
        )

    return teacher_keep_mask, student_keep_masks, metadata
