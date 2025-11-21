# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
SSL (Self-Supervised Learning) cropping strategies for student-teacher training.

This module implements different SSL approaches:
- JEPA (Joint-Embedding Predictive Architecture): 1 global teacher + N local students
- Dino 2: 2 global teachers + many local students
- iBoT: Global teacher + masked student views
- Multi-scale: Views at different HEALPix levels for hierarchical learning

Each strategy defines specific relationships and coverage patterns between views.
"""

import numpy as np
from typing import List, Tuple
from weathergen.datasets.masking import Masker
from weathergen.datasets.batch import ViewMetadata


def build_jepa_views(
    masker: Masker,
    num_cells: int,
    teacher_cfg: dict,
    student_cfg: dict,
) -> Tuple[np.ndarray, List[np.ndarray], List[ViewMetadata]]:
    """
    JEPA-style cropping: 1 global teacher + N local student views.

    The teacher has high coverage (typically 70-90% of cells), and each student
    has lower coverage (20-50%). Students can overlap with teacher (subset) or
    be independent based on configuration.

    Parameters
    ----------
    masker : Masker
        Instance providing RNG and healpix-level info.
    num_cells : int
        Number of healpix cells at data level.
    teacher_cfg : dict
        Teacher config: {strategy, rate, masking_strategy_config}.
    student_cfg : dict
        Student config: {masking_strategy, rate, num_views, relationship, masking_strategy_config}.

    Returns
    -------
    teacher_keep_mask : np.ndarray
        Boolean keep mask for teacher view (high coverage).
    student_keep_masks : list[np.ndarray]
        Boolean keep masks for each student view (lower coverage).
    metadata : list[ViewMetadata]
        Metadata objects (teacher first, then students).
    """
    # Generate teacher view (global, high coverage)
    strat_teacher = teacher_cfg.get("strategy", "random")
    rate_teacher = teacher_cfg.get("rate", 0.8)  # Default 80% coverage
    t_cfg_extra = teacher_cfg.get("masking_strategy_config")

    teacher_keep_mask = masker.generate_cell_keep_mask(
        num_cells=num_cells,
        strategy=strat_teacher,
        rate=rate_teacher,
        masking_strategy_config=t_cfg_extra,
    )

    # Generate student views (local, lower coverage)
    num_views = student_cfg.get("num_views", 4)
    strat_student = student_cfg.get("masking_strategy", "random")
    rate_student = student_cfg.get("rate", 0.3)  # Default 30% coverage
    s_cfg_extra = student_cfg.get("masking_strategy_config")
    relationship = student_cfg.get("relationship", "subset")

    student_keep_masks: List[np.ndarray] = []
    for v in range(num_views):
        base = masker.generate_cell_keep_mask(
            num_cells=num_cells,
            strategy=strat_student,
            rate=rate_student,
            masking_strategy_config=s_cfg_extra,
        )

        if relationship == "subset":
            # Student views are subsets of teacher
            keep = base & teacher_keep_mask
        elif relationship == "disjoint":
            # Students are disjoint from teacher (useful for contrastive learning)
            keep = base & (~teacher_keep_mask)
        else:  # independent
            # Students are independent of teacher
            keep = base

        student_keep_masks.append(keep)

    # Build metadata
    metadata: List[ViewMetadata] = []
    metadata.append(
        ViewMetadata(
            view_id="jepa_teacher_global",
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
                view_id=f"jepa_student_local_{i}",
                keep_mask=m,
                strategy=strat_student,
                healpix_level=masker.healpix_level_data,
                rate=rate_student,
                parent_view_id="jepa_teacher_global" if relationship == "subset" else None,
            )
        )

    return teacher_keep_mask, student_keep_masks, metadata


def build_dino2_views(
    masker: Masker,
    num_cells: int,
    teacher_cfg: dict,
    student_cfg: dict,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[ViewMetadata]]:
    """
    Dino 2-style cropping: 2 global teacher views + N local student views.

    Both teachers have high coverage (70-80%) and are independent. Students
    have lower coverage (10-30%) and are all independent from each other and
    from teachers.

    Parameters
    ----------
    masker : Masker
        Instance providing RNG and healpix-level info.
    num_cells : int
        Number of healpix cells at data level.
    teacher_cfg : dict
        Teacher config: {strategy, rate, num_global_views, masking_strategy_config}.
    student_cfg : dict
        Student config: {masking_strategy, rate, num_views, masking_strategy_config}.

    Returns
    -------
    teacher_keep_masks : list[np.ndarray]
        Boolean keep masks for teacher views (typically 2).
    student_keep_masks : list[np.ndarray]
        Boolean keep masks for each student view.
    metadata : list[ViewMetadata]
        Metadata objects (teachers first, then students).
    """
    # Generate teacher views (2 global views with high coverage)
    num_global_views = teacher_cfg.get("num_global_views", 2)
    strat_teacher = teacher_cfg.get("strategy", "random")
    rate_teacher = teacher_cfg.get("rate", 0.75)  # Default 75% coverage
    t_cfg_extra = teacher_cfg.get("masking_strategy_config")

    teacher_keep_masks: List[np.ndarray] = []
    for g in range(num_global_views):
        teacher_mask = masker.generate_cell_keep_mask(
            num_cells=num_cells,
            strategy=strat_teacher,
            rate=rate_teacher,
            masking_strategy_config=t_cfg_extra,
        )
        teacher_keep_masks.append(teacher_mask)

    # Generate student views (many local views with lower coverage)
    num_local_views = student_cfg.get("num_views", 8)
    strat_student = student_cfg.get("masking_strategy", "random")
    rate_student = student_cfg.get("rate", 0.2)  # Default 20% coverage
    s_cfg_extra = student_cfg.get("masking_strategy_config")

    student_keep_masks: List[np.ndarray] = []
    for v in range(num_local_views):
        student_mask = masker.generate_cell_keep_mask(
            num_cells=num_cells,
            strategy=strat_student,
            rate=rate_student,
            masking_strategy_config=s_cfg_extra,
        )
        student_keep_masks.append(student_mask)

    # Build metadata
    metadata: List[ViewMetadata] = []
    for g, m in enumerate(teacher_keep_masks):
        metadata.append(
            ViewMetadata(
                view_id=f"dino2_teacher_global_{g}",
                keep_mask=m,
                strategy=strat_teacher,
                healpix_level=masker.healpix_level_data,
                rate=rate_teacher,
                parent_view_id=None,
            )
        )
    for i, m in enumerate(student_keep_masks):
        metadata.append(
            ViewMetadata(
                view_id=f"dino2_student_local_{i}",
                keep_mask=m,
                strategy=strat_student,
                healpix_level=masker.healpix_level_data,
                rate=rate_student,
                parent_view_id=None,  # All views are independent in Dino 2
            )
        )

    return teacher_keep_masks, student_keep_masks, metadata


def build_ibot_views(
    masker: Masker,
    num_cells: int,
    teacher_cfg: dict,
    student_cfg: dict,
) -> Tuple[np.ndarray, List[np.ndarray], List[ViewMetadata]]:
    """
    iBoT-style cropping: Global teacher + masked student views.

    iBoT combines self-distillation (like Dino) with masked image modeling (like BERT).
    The teacher has high coverage, and students have similar coverage but with
    additional random masking applied.

    Parameters
    ----------
    masker : Masker
        Instance providing RNG and healpix-level info.
    num_cells : int
        Number of healpix cells at data level.
    teacher_cfg : dict
        Teacher config: {strategy, rate, masking_strategy_config}.
    student_cfg : dict
        Student config: {masking_strategy, rate, num_views, mask_ratio, masking_strategy_config}.

    Returns
    -------
    teacher_keep_mask : np.ndarray
        Boolean keep mask for teacher view (high coverage).
    student_keep_masks : list[np.ndarray]
        Boolean keep masks for each student view (with additional masking).
    metadata : list[ViewMetadata]
        Metadata objects (teacher first, then students).
    """
    # Generate teacher view (global, high coverage)
    strat_teacher = teacher_cfg.get("strategy", "random")
    rate_teacher = teacher_cfg.get("rate", 0.9)  # Default 90% coverage
    t_cfg_extra = teacher_cfg.get("masking_strategy_config")

    teacher_keep_mask = masker.generate_cell_keep_mask(
        num_cells=num_cells,
        strategy=strat_teacher,
        rate=rate_teacher,
        masking_strategy_config=t_cfg_extra,
    )

    # Generate student views with additional masking
    num_views = student_cfg.get("num_views", 2)
    strat_student = student_cfg.get("masking_strategy", "random")
    rate_student = student_cfg.get("rate", 0.8)  # Start with similar coverage
    mask_ratio = student_cfg.get("mask_ratio", 0.4)  # Additional masking ratio
    s_cfg_extra = student_cfg.get("masking_strategy_config")

    student_keep_masks: List[np.ndarray] = []
    for v in range(num_views):
        # First generate base coverage (similar to teacher)
        base = masker.generate_cell_keep_mask(
            num_cells=num_cells,
            strategy=strat_student,
            rate=rate_student,
            masking_strategy_config=s_cfg_extra,
        )

        # Apply additional random masking (BERT-style)
        # This simulates masked token prediction
        # We can use the masker to generate a mask for the additional masking 
        # instead of using keep cells logic
        mask_additional = masker.generate_cell_keep_mask(
            num_cells=num_cells,
            strategy="random",
            rate=1.0 - mask_ratio,  # Invert: we want to keep (1 - mask_ratio)
            masking_strategy_config=None,
        )

        # Combine: student sees base coverage minus masked tokens
        keep = base & mask_additional
        student_keep_masks.append(keep)

    # Build metadata
    metadata: List[ViewMetadata] = []
    metadata.append(
        ViewMetadata(
            view_id="ibot_teacher_global",
            keep_mask=teacher_keep_mask,
            strategy=strat_teacher,
            healpix_level=masker.healpix_level_data,
            rate=rate_teacher,
            parent_view_id=None,
        )
    )
    for i, m in enumerate(student_keep_masks):
        effective_rate = np.sum(m) / num_cells
        metadata.append(
            ViewMetadata(
                view_id=f"ibot_student_masked_{i}",
                keep_mask=m,
                strategy=strat_student,
                healpix_level=masker.healpix_level_data,
                rate=effective_rate,
                parent_view_id="ibot_teacher_global",
            )
        )

    return teacher_keep_mask, student_keep_masks, metadata


def build_multi_scale_views(
    masker: Masker,
    num_cells: int,
    teacher_cfg: dict,
    student_cfg: dict,
) -> Tuple[np.ndarray, List[np.ndarray], List[ViewMetadata]]:
    """
    Multi-scale cropping: 1 teacher + multiple students at different scales.

    Creates student views at different HEALPix levels to capture multi-scale
    patterns. Useful for hierarchical representation learning.

    Parameters
    ----------
    masker : Masker
        Instance providing RNG and healpix-level info.
    num_cells : int
        Number of healpix cells at data level.
    teacher_cfg : dict
        Teacher config: {strategy, rate, masking_strategy_config}.
    student_cfg : dict
        Student config: {masking_strategy, rates, hl_masks, num_views_per_scale}.

    Returns
    -------
    teacher_keep_mask : np.ndarray
        Boolean keep mask for teacher view.
    student_keep_masks : list[np.ndarray]
        Boolean keep masks for student views at different scales.
    metadata : list[ViewMetadata]
        Metadata objects (teacher first, then students).
    """
    # Generate teacher view
    strat_teacher = teacher_cfg.get("strategy", "healpix")
    rate_teacher = teacher_cfg.get("rate", 0.8)
    t_cfg_extra = teacher_cfg.get("masking_strategy_config")

    teacher_keep_mask = masker.generate_cell_keep_mask(
        num_cells=num_cells,
        strategy=strat_teacher,
        rate=rate_teacher,
        masking_strategy_config=t_cfg_extra,
    )

    # Generate multi-scale student views
    strat_student = student_cfg.get("masking_strategy", "healpix")
    rates = student_cfg.get("rates", [0.5, 0.3, 0.15])  # Different scales
    hl_masks = student_cfg.get("hl_masks", [2, 1, 0])  # Different HEALPix levels
    num_views_per_scale = student_cfg.get("num_views_per_scale", 1)
    relationship = student_cfg.get("relationship", "subset")

    student_keep_masks: List[np.ndarray] = []
    scale_info: List[Tuple[float, int]] = []

    for rate, hl_mask in zip(rates, hl_masks):
        for _ in range(num_views_per_scale):
            s_cfg = {"hl_mask": hl_mask}
            base = masker.generate_cell_keep_mask(
                num_cells=num_cells,
                strategy=strat_student,
                rate=rate,
                masking_strategy_config=s_cfg,
            )

            if relationship == "subset":
                keep = base & teacher_keep_mask
            else:
                keep = base

            student_keep_masks.append(keep)
            scale_info.append((rate, hl_mask))

    # Build metadata
    metadata: List[ViewMetadata] = []
    metadata.append(
        ViewMetadata(
            view_id="multiscale_teacher_global",
            keep_mask=teacher_keep_mask,
            strategy=strat_teacher,
            healpix_level=masker.healpix_level_data,
            rate=rate_teacher,
            parent_view_id=None,
        )
    )
    for i, (m, (rate, hl_mask)) in enumerate(zip(student_keep_masks, scale_info)):
        metadata.append(
            ViewMetadata(
                view_id=f"multiscale_student_scale{hl_mask}_view{i}",
                keep_mask=m,
                strategy=strat_student,
                healpix_level=masker.healpix_level_data,
                rate=rate,
                parent_view_id="multiscale_teacher_global" if relationship == "subset" else None,
            )
        )

    return teacher_keep_mask, student_keep_masks, metadata
