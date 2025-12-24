import logging
import warnings

import astropy_healpix as hp
from astropy import units as u
import numpy as np
import torch
from numpy.typing import NDArray

from weathergen.common.config import Config
from weathergen.datasets.batch import SampleMetaData

_logger = logging.getLogger(__name__)


class MaskData:
    masks: list[np.typing.NDArray] = []
    metadata: list[SampleMetaData] = []

    def __init__(self):
        self.masks = []
        self.metadata = []

    def __len__(self):
        return len(self.masks)

    def add_mask(self, mask, params, cfg):
        self.masks += [mask]
        self.metadata += [
            SampleMetaData(
                params={**cfg, **params},
                mask=mask,
                global_params={
                    "loss": cfg.get("loss", {}),
                    "masking_strategy": cfg.get("strategy", {}),
                    "relationship": cfg.get("relationship", {}),
                },
            )
        ]


# Convert to torch.bool
def to_bool_tensor(arr):
    return torch.from_numpy(np.asarray(arr)).to(torch.bool)


class Masker:
    """Class to generate masks for token sequences and apply them.
    This class supports different masking strategies and combinations.

    Attributes:
        masking_rate (float): The base rate at which tokens are masked.
        masking_strategy (str): The strategy used for masking (e.g., "random",
        "block", "healpix", "cropping_healpix", "channel").
        current_strategy (str): The current strategy in use, relevant
                                when using "combination" strategy.
        "random" - random masking of tokens at the level of the data
        "block" - masking out large blocks of tokens in 1D, without spatial meaning
        "healpix" - masking at the level of HEALPix cells, where all child cells
                    of a parent cell at a specific HEALpix level are masked
                    if the parent is masked.
                    The healpix level must be configured with hl_mask.
                    e.g. masking_strategy_config = {"hl_mask": 1}
                    with hl_mask the level for masking that we want to apply
                    e.g. level 1 very large cells masked
        "cropping_healpix" - spatial cropping that keeps spatially contiguous regions
                    and masks everything else. Uses neighbor relationships or geodesic
                    distance to ensure spatial contiguity. For DINO/JEPA/IBOT.
                    e.g. masking_strategy_config = {"hl_mask": 0, "method": "geodesic_disk"}
                    method: "disk" (neighbor growth), "random_walk", or "geodesic_disk" (circular)
        "channel" - masking data channels, where channels of the data are masked
                    can be done per-cell (each cell has different channels masked)
                    or globally (all have the same channels masked).
                    e.g. masking_strategy_config = {"mode": "per_cell"} or
                    {"mode": "global"}
        "causal" - masking the latest timesteps in each token, according to the masking rate.
        masking_rate_sampling (bool): Whether to sample the masking rate from a distribution.
        masking_strategy_config (dict): Configuration for the masking strategy, can include
                                        additional parameters like "hl_mask", etc.
                                        specific to the masking strategy. See above.
    """

    def __init__(self, cf: Config):
        self.rng = None

        self.mask_value = 0.0
        self.dim_time_enc = 6

        # number of healpix cells
        self.healpix_level_data = cf.healpix_level
        self.healpix_num_cells = 12 * (4**cf.healpix_level)

    def reset_rng(self, rng) -> None:
        """
        Reset rng after mini_epoch to ensure proper randomization
        """
        self.rng = rng

    def _get_sampling_rate(self):
        """
        Get the sampling, if requested by sampling it itself
        """

        # if masking_rate_sampling is enabled, sample the rate from a normal distribution.
        if self.masking_rate_sampling:
            rate = np.clip(
                np.abs(self.rng.normal(loc=self.masking_rate, scale=1.0 / (2.5 * np.pi))),
                0.01,
                0.99,
            )
        else:
            rate = self.masking_rate

        return rate

    def _generate_healpix_mask(self, token_lens: list[int], rate: float) -> np.typing.NDArray:
        """
        Generates a token-level mask based on hierarchical HEALPix cell selection.

        This method identifies parent cells at a lower resolution (hl_mask) and
        masks all the child cells (and their corresponding tokens) at the data
        resolution (hl_data).

        Args:
            token_lens (list[int]): A list containing the number of tokens in each cell.
            rate (float): The desired masking rate, applied to the parent cells.

        Returns:
            np.ndarray: A flat boolean array (the token-level mask).
        """

        # hl_mask should be provided in masking_strategy_config
        hl_data = self.healpix_level_data
        hl_mask = self.masking_strategy_config.get("hl_mask")

        assert len(token_lens) == self.healpix_num_cells, (
            f"Expected {self.healpix_num_cells} cells at level {hl_data}, got {len(token_lens)}."
        )

        # Calculate the number of parent cells at the mask level (hl_mask)
        num_parent_cells = 12 * (4**hl_mask)
        level_diff = hl_data - hl_mask
        num_children_per_parent = 4**level_diff

        rate = self._get_sampling_rate()

        # Choose parent cells to mask based on the specified rate.
        num_parents_to_mask = int(np.round(rate * num_parent_cells))

        if num_parents_to_mask == 0:
            return np.zeros(sum(token_lens), dtype=bool)

        # Select parent cells to mask
        parent_ids_to_mask = self.rng.choice(num_parent_cells, num_parents_to_mask, replace=False)

        # For each parent ID, calculate the child indices and set them in the mask
        parent_ids = np.asarray(parent_ids_to_mask)
        child_offsets = np.arange(num_children_per_parent)
        child_indices = (parent_ids[:, None] * num_children_per_parent + child_offsets).reshape(-1)

        # set mask list for children
        cell_mask = np.zeros(self.healpix_num_cells, dtype=bool)
        cell_mask[child_indices] = True

        # Make the cell-level mask flat and apply it to the token lengths.
        # np.repeat repeats each element of `cell_mask` a number of times specified by `token_lens`.
        flat_mask = np.repeat(cell_mask, token_lens)

        return flat_mask

    def _generate_channel_mask(
        self,
        tokenized_data: list[torch.Tensor],
        rate: float,
        coords: torch.Tensor,
        geoinfos: torch.Tensor,
        source: torch.Tensor,
    ) -> list[np.typing.NDArray]:
        """
        Generates a channel mask for each cell, handling completely empty tensors.
        This method is robust against cells represented as 1D tensors of shape [0].

        Args:
            tokenized_data (list[torch.Tensor]): A list of tensors. Most will have a shape of
                                                (dim, num_tokens, num_channels), but some may
                                            be empty with a shape of (0,), no data in cell
            rate (float): The desired masking rate for channels.
            coords (torch.Tensor): The coordinates tensor.
            geoinfos (torch.Tensor): The geoinfos tensor.

        Returns:
            list[np.ndarray]: A list of boolean masks. Each mask corresponds to a tensor
                            in tokenized_data.
        """

        if not tokenized_data:
            return []

        # masking rate sampling, to be refactored as shared between methods
        rate = self._get_sampling_rate()

        # isolate the number of actual data channels. 6 refers to time.
        num_channels = self.dim_time_enc + coords.shape[-1] + geoinfos.shape[-1] + source.shape[-1]
        assert num_channels > 0, "For channel masking, number of channels has to be nonzero."
        num_fixed_channels = self.dim_time_enc + coords.shape[-1] + geoinfos.shape[-1]
        num_data_channels = source.shape[-1]
        mask_count = int(num_data_channels * rate)

        # cat all tokens for efficient processing, split at the end again
        # masks are generated simulatneously for all cells

        tokenized_data_lens = [len(t) for t in tokenized_data]
        tokenized_data_merged = torch.cat(tokenized_data)

        num_tokens = tokenized_data_merged.shape[0]
        token_size = tokenized_data_merged.shape[1]

        if self.masking_strategy_config.get("mode") == "global":
            # generate global mask
            channel_mask = np.zeros(num_channels, dtype=bool)
            m = num_fixed_channels + self.rng.choice(num_data_channels, mask_count, replace=False)
            channel_mask[m] = True

            full_mask = np.zeros_like(tokenized_data_merged).astype(np.bool)
            full_mask[:, :] = channel_mask

        else:  # different mask per cell
            # generate all False mask but with swapped token_size and num_tokens dims so that
            # the masking is constant per token
            channel_mask = np.zeros((token_size, num_tokens, num_channels), dtype=bool)
            # apply masking
            nc = (num_tokens, num_data_channels)
            channel_mask[:, :, num_fixed_channels:] = self.rng.uniform(0, 1, nc) < rate
            # recover correct shape, i.e. swap token_size and num_tokens
            full_mask = channel_mask.transpose([1, 0, 2])

        # split across cells again
        full_mask = np.split(full_mask, np.cumsum(tokenized_data_lens[:-1]))

        return full_mask

    def _generate_causal_mask(
        self,
        tokenized_data: list[torch.Tensor],
        rate: float,
        coords: torch.Tensor,
        geoinfos: torch.Tensor,
        source: torch.Tensor,
    ) -> list[np.typing.NDArray]:
        """
        Generates a causal mask, masking the latest times
        in each tokenized_data according to the masking rate.
        """
        if not tokenized_data:
            return []

        rate = self._get_sampling_rate()

        # Extract all lengths at once
        token_lens = np.array([len(token_data) for token_data in tokenized_data])

        if len(token_lens) == 0:
            return []

        # Calculate start indices for masking
        # astype(int) performs floor operation by truncation
        num_future_to_mask = (rate * token_lens).astype(int)
        start_mask_indices = np.maximum(1, token_lens - num_future_to_mask)

        # Handle edge cases
        mask_valid = token_lens > 1  # Only cells with >1 timestep can be masked
        start_mask_indices = np.where(mask_valid, start_mask_indices, token_lens)

        # Create masks with list comprehension
        # Needed to handle variable lengths
        full_mask = [
            (
                np.concatenate(
                    [
                        np.zeros(start_idx, dtype=bool),
                        np.ones(max(0, token_len - start_idx), dtype=bool),
                    ]
                )
                if token_len > 1
                else (np.zeros(1, dtype=bool) if token_len == 1 else np.array([], dtype=bool))
            )
            for token_len, start_idx in zip(token_lens, start_mask_indices, strict=False)
        ]

        return full_mask

    def build_samples_for_stream(
        self, training_mode: str, num_cells: int, training_cfg: dict
    ) -> tuple[np.typing.NDArray, list[np.typing.NDArray], list[SampleMetaData]]:
        """
        Construct teacher/student keep masks for a stream.
        SampleMetaData is currently just a dict with the masking params used.
        """

        target_cfgs = training_cfg.get("target_input", [])
        source_cfgs = training_cfg.get("model_input", [])

        # target and source are assumed identical when target is not specified
        if len(target_cfgs) == 0:
            target_cfgs = source_cfgs

        target_masks = MaskData()
        source_masks = MaskData()
        source_target_mapping = []
        i_target = 0
        # iterate over all target samples
        # different strategies
        for _, target_cfg in enumerate(target_cfgs):
            # different samples/view per strategy
            for _ in range(target_cfg.get("num_samples", 1)):
                target_mask, target_params = self._get_mask(
                    num_cells=num_cells,
                    strategy=target_cfg.get("masking_strategy"),
                    target_mask=None,
                    target_metadata=None,
                    masking_strategy_config=target_cfg.get("masking_strategy_config", {}),
                )
                target_masks.add_mask(target_mask, target_params, target_cfg)

                for _i_source, source_cfg in enumerate(source_cfgs):
                    # samples per strategy
                    for _ in range(source_cfg.get("num_samples", 1)):
                        source_mask, source_params = self._get_mask(
                            num_cells=num_cells,
                            strategy=source_cfg.get("masking_strategy"),
                            target_mask=target_mask,
                            target_metadata=target_params,  # Pass teacher metadata for cone_distance
                            masking_strategy_config=source_cfg.get("masking_strategy_config", {}),
                            relationship=source_cfg.get("relationship", "independent"),
                        )
                        source_masks.add_mask(source_mask, source_params, source_cfg)
                        # TODO: proper correspondence between source and target
                        source_target_mapping += [i_target]
                i_target += 1

        source_target_mapping = np.array(source_target_mapping, dtype=np.int32)

        return (target_masks, source_masks, source_target_mapping)

    def _get_mask(
        self,
        num_cells: int,
        strategy: str | None = None,
        masking_strategy_config: dict | None = None,
        target_mask: np.typing.NDArray | None = None,
        target_metadata: dict | None = None,
        relationship: str | None = None,
    ) -> (np.typing.NDArray, dict):
        """Get effective mask, combining with target mask if specified.

        Parameters
        ----------
        num_cells : int
            Number of cells at data level (should equal 12 * 4**healpix_level).
        strategy : str | None
            Cell selection strategy: currently supports 'random' and 'healpix'. Uses
            instance default if None.
        masking_strategy_config : dict | None
            Optional override of strategy config (e.g., {'hl_mask': 3}).
        target_mask : np.ndarray | None
            Optional teacher mask for controlled overlap (used with 'overlap' relationship)
        target_metadata : dict | None
            Optional teacher metadata (e.g., contains 'center_cell' for cone_distance relationship)
        relationship : str | None
            How to relate student to teacher mask:
            - "independent": no relationship (default)
            - "complement": student = NOT teacher (0% overlap)
            - "subset": student ⊆ teacher (100% overlap)
            - "disjoint": student ∩ teacher = ∅ (0% overlap)
            - "identity": student = teacher (100% overlap)
            - "overlap": fractional overlap controlled by overlap_ratio in config
            - "cone_distance": geometric overlap via cone centers at specified angular distance

        Returns
        -------
        np.ndarray
            Boolean array of shape [num_cells] where True indicates the cell is kept.
        dict
            Parameters describing the masking that was applied
        """

        if strategy == "forecast":
            if relationship is not None:
                assert relationship == "independent", (
                    "strategy forecast requires relationship independent "
                )

        # handle cases where mask is directly derived from target_mask
        if relationship == "complement":
            assert target_mask is not None, (
                "relationship: {relationship} incompatible with target_mask None"
            )
            mask = ~target_mask
            return mask, {}

        # get mask
        mask, params = self._generate_cell_mask(num_cells, strategy, masking_strategy_config)

        # handle cases where mask needs to be combined with target_mask
        # without the assert we can fail silently
        if relationship == "subset":
            assert target_mask is not None, (
                "relationship: {relationship} incompatible with target_mask None"
            )
            mask = mask & target_mask
        elif relationship == "disjoint":
            assert target_mask is not None, (
                "relationship: {relationship} incompatible with target_mask None"
            )
            mask = mask & (~target_mask)
        elif relationship == "identity":
            assert target_mask is not None, (
                "relationship: {relationship} incompatible with target_mask None"
            )
            mask = target_mask
        elif relationship == "overlap":
            # Fractional overlap: deterministically control overlap percentage
            assert target_mask is not None, (
                "relationship 'overlap' requires target_mask"
            )
            overlap_ratio = masking_strategy_config.get("overlap_ratio", None)
            assert overlap_ratio is not None, (
                "relationship 'overlap' requires 'overlap_ratio' in masking_strategy_config"
            )
            mask = self._create_fractional_overlap_mask(
                mask, target_mask, overlap_ratio, num_cells
            )
        elif relationship == "cone_distance":
            # Geometric overlap via cone centers separated by angular distance
            assert target_mask is not None, (
                "relationship 'cone_distance' requires target_mask"
            )
            assert target_metadata is not None, (
                "relationship 'cone_distance' requires target_metadata with center_cell"
            )
            center_distance_degrees = masking_strategy_config.get("center_distance_degrees", None)
            assert center_distance_degrees is not None, (
                "relationship 'cone_distance' requires 'center_distance_degrees' in masking_strategy_config"
            )

            # Get teacher center from metadata
            teacher_center_cell = target_metadata.get("center_cell", None)
            assert teacher_center_cell is not None, (
                "Teacher metadata must contain 'center_cell' for cone_distance relationship"
            )

            # Create student cone at specified distance from teacher
            mask, student_center_cell = self._create_cone_distance_mask(
                num_cells,
                masking_strategy_config,
                teacher_center_cell,
                center_distance_degrees
            )

            # Store student center in params for potential chaining
            params["center_cell"] = student_center_cell

        return (mask, params)

    def _generate_cell_mask(
        self, num_cells: int, strategy: str, masking_strategy_config: dict
    ) -> (np.typing.NDArray, dict):
        """Generate a boolean keep mask at data healpix level (True = keep cell).

        Parameters
        ----------
        num_cells : int
            Number of cells at data level (should equal 12 * 4**healpix_level).
        strategy : str | None
            Cell selection strategy: currently supports 'random' and 'healpix'. Uses
            instance default if None.
        masking_strategy_config : dict | None
            Optional override of strategy config (e.g., {'hl_mask': 3}).

        Returns
        -------
        np.ndarray
            Boolean array of shape [num_cells] where True indicates the cell is kept.
        """

        # params describing the masking
        masking_params = {}

        # get config for mask

        cfg = masking_strategy_config
        keep_rate = cfg.get("rate", None)
        assert keep_rate is not None, 'No sampling rate "rate" specified.'

        # sample rate if requested (only if explicit rate not provided)
        # if rate is None and self.masking_rate_sampling:
        #     keep_rate = self._get_sampling_rate()

        assert 0.0 <= keep_rate <= 1.0, f"keep_rate out of bounds: {keep_rate}"
        assert num_cells == self.healpix_num_cells, (
            "num_cells inconsistent with configured healpix level."
        )

        # generate cell mask

        if strategy == "random":
            mask = self.rng.uniform(0, 1, num_cells) < keep_rate

        elif "forecast" in strategy or strategy == "causal":
            mask = np.ones(num_cells, dtype=np.bool)

            if "diffusion_rn" in masking_strategy_config:
                masking_params["noise_level_rn"] = self.rng.normal(0.0, 1.0)

        elif strategy == "healpix":
            hl_data = self.healpix_level_data
            hl_mask = cfg.get("hl_mask")
            assert hl_mask is not None and hl_mask < hl_data, (
                "For healpix keep mask generation, cfg['hl_mask'] must be set and < data level."
            )
            num_parent_cells = 12 * (4**hl_mask)
            level_diff = hl_data - hl_mask
            num_children_per_parent = 4**level_diff
            # number of parents to keep
            num_parents_to_keep = int(np.round(keep_rate * num_parent_cells))
            if num_parents_to_keep == 0:
                mask = np.zeros(num_cells, dtype=bool)
            else:
                parent_ids = self.rng.choice(num_parent_cells, num_parents_to_keep, replace=False)
                child_offsets = np.arange(num_children_per_parent)
                child_indices = (
                    parent_ids[:, None] * num_children_per_parent + child_offsets
                ).reshape(-1)
                mask = np.zeros(num_cells, dtype=bool)
                mask[child_indices] = True

        elif strategy == "cropping_healpix":
            # Spatial cropping: select contiguous region and keep it (mask rest)
            # This is the inverse of healpix masking
            hl_data = self.healpix_level_data
            hl_mask = cfg.get("hl_mask")
            assert hl_mask is not None and hl_mask < hl_data, (
                "For cropping_healpix, cfg['hl_mask'] must be set and < data level."
            )
            num_parent_cells = 12 * (4**hl_mask)
            level_diff = hl_data - hl_mask
            num_children_per_parent = 4**level_diff

            # Number of parents to keep (spatially contiguous)
            num_parents_to_keep = int(np.round(keep_rate * num_parent_cells))

            if num_parents_to_keep == 0:
                mask = np.zeros(num_cells, dtype=bool)
            else:
                # Spatial selection method
                method = cfg.get("method", "geodesic_disk")  # Default to best method for SSL

                # Use standard spatial selection
                parent_ids = self._select_spatially_contiguous_cells(
                    healpix_level=hl_mask,
                    num_cells_to_select=num_parents_to_keep,
                    center_cell=None,
                    method=method,
                )

                # Store center cell for potential use by cone_distance relationship
                # (stored in self._last_center_cell by _select_spatially_contiguous_cells)
                masking_params["center_cell"] = self._last_center_cell

                # Project to data level
                child_offsets = np.arange(num_children_per_parent)
                child_indices = (
                    parent_ids[:, None] * num_children_per_parent + child_offsets
                ).reshape(-1)

                # Create mask: True = MASK (masked tokens), False = KEEP (kept tokens)
                mask = np.zeros(num_cells, dtype=bool)
                mask[child_indices] = True

        else:
            raise NotImplementedError(
                f"Cell selection strategy '{strategy}' not supported for keep mask generation."
            )

        mask = to_bool_tensor(mask)

        return (mask, masking_params)

    def _select_spatially_contiguous_cells(
        self,
        healpix_level: int,
        num_cells_to_select: int,
        center_cell: int | None = None,
        method: str = "disk",
    ) -> NDArray:
        """
        Select spatially contiguous cells on the sphere using neighbor relationships.

        This is the core spatial selection helper used for both masking and cropping.

        Args:
            healpix_level: HEALPix level for selection
            num_cells_to_select: Number of cells to select
            center_cell: Starting cell (None = random)
            method: Selection method:
                - "disk": Layer-by-layer neighbor growth (compact regions)
                - "random_walk": Random neighbor selection (irregular shapes)
                - "geodesic_disk": Angular distance selection (circular regions, best for SSL)

        Returns:
            Array of selected cell indices forming a spatially contiguous region

        Note:
            The center cell used for selection is stored in self._last_center_cell for use
            by relationships like "cone_distance" that need geometric information.

        Examples:
            # Independent crop
            crop1 = _select_spatially_contiguous_cells(0, 9, method="geodesic_disk")
            # Access center via self._last_center_cell if needed
        """

        num_total_cells = 12 * (4**healpix_level)
        nside = 2**healpix_level

        assert num_cells_to_select <= num_total_cells

        # Random starting point if not specified
        if center_cell is None:
            center_cell = self.rng.integers(0, num_total_cells)

        # Store center cell for potential use by cone_distance relationship
        self._last_center_cell = int(center_cell)

        if method == "disk":
            selected = self._select_disk(center_cell, num_cells_to_select, nside)
        elif method == "random_walk":
            selected = self._select_random_walk(center_cell, num_cells_to_select, nside)
        elif method == "geodesic_disk":
            selected = self._select_geodesic_disk(
                center_cell, num_cells_to_select, nside, num_total_cells
            )
        else:
            raise ValueError(f"Unknown selection method: {method}")

        return np.array(sorted(selected))

    # separate functions for the different methods of producing spatially contiguous regions
    def _select_disk(self, center_cell: int, num_cells_to_select: int, nside: int) -> set[int]:
        """
        Select cells in a disk shape by expanding layer by layer.
        """
        selected = {center_cell}
        frontier = {center_cell}

        while len(selected) < num_cells_to_select and frontier:
            # Expand frontier by one layer
            next_frontier = set()
            for cell in frontier:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="invalid value encountered")
                    neighbors = hp.neighbours(cell, nside, order="nested")
                valid_neighbors = [n for n in neighbors if n != -1 and n not in selected]
                next_frontier.update(valid_neighbors)

            if not next_frontier:
                break

            # Randomly select from frontier to reach target count
            candidates = list(next_frontier)
            self.rng.shuffle(candidates)
            num_to_add = min(len(candidates), num_cells_to_select - len(selected))
            selected.update(candidates[:num_to_add])
            frontier = set(candidates[:num_to_add])

        return selected

    def _select_random_walk(
        self, center_cell: int, num_cells_to_select: int, nside: int
    ) -> set[int]:
        """
        Random walk through neighbors, creates elongated irregular regions
        """
        selected = {center_cell}
        frontier = {center_cell}

        while len(selected) < num_cells_to_select:
            # Get all neighbors of current frontier
            neighbors = set()
            for cell in frontier:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="invalid value encountered")
                    cell_neighbors = hp.neighbours(cell, nside, order="nested")
                valid = [n for n in cell_neighbors if n != -1 and n not in selected]
                neighbors.update(valid)

            if not neighbors:
                break

            # Randomly pick one neighbor and continue from there
            next_cell = self.rng.choice(list(neighbors))
            selected.add(next_cell)
            frontier = {next_cell}

        return selected

    def _select_geodesic_disk(
        self, center_cell: int, num_cells_to_select: int, nside: int, num_total_cells: int
    ) -> set:
        """
        Angular distance selection, creates most uniform somewhat circular regions
        """

        def lonlat_to_xyz(lon, lat):
            """
            Convert lon/lat to 3D cartesian coordinates.
            """
            return np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])

        # Get center coordinates
        center_lonlat = hp.healpix_to_lonlat(center_cell, nside, order="nested")
        center_lon = float(
            center_lonlat[0].value if hasattr(center_lonlat[0], "value") else center_lonlat[0]
        )
        center_lat = float(
            center_lonlat[1].value if hasattr(center_lonlat[1], "value") else center_lonlat[1]
        )
        center_xyz = lonlat_to_xyz(center_lon, center_lat)

        # Get all cell coordinates
        all_indices = np.arange(num_total_cells)
        all_lonlat = hp.healpix_to_lonlat(all_indices, nside, order="nested")
        all_lon = all_lonlat[0].value if hasattr(all_lonlat[0], "value") else all_lonlat[0]
        all_lat = all_lonlat[1].value if hasattr(all_lonlat[1], "value") else all_lonlat[1]

        all_xyz = np.stack(
            [
                np.cos(all_lat) * np.cos(all_lon),
                np.cos(all_lat) * np.sin(all_lon),
                np.sin(all_lat),
            ],
            axis=1,
        )
        # Compute angular distances and select closest cells
        dot_products = np.clip(np.dot(all_xyz, center_xyz), -1.0, 1.0)
        angular_distances = np.arccos(dot_products)
        selected = np.argsort(angular_distances)[:num_cells_to_select]

        return selected

    def _create_cone_distance_mask(
        self,
        num_cells: int,
        masking_strategy_config: dict,
        teacher_center_cell: int,
        center_distance_degrees: float,
    ) -> tuple[torch.Tensor, int]:
        """
        Create a student cone (geodesic disk) at specified angular distance from teacher center.

        This creates geometrically controlled overlap where both teacher and student are perfect
        geodesic disks (spatially contiguous circular regions) and their overlap is determined by:
        - The radii of the two cones (from their 'rate' configs)
        - The angular distance between their centers

        This is ideal for SSL training as it provides:
        - Perfect spatial contiguity (both are clean circular regions)
        - Deterministic geometric overlap (no random cell selection)
        - Intuitive control ("cones are X degrees apart")

        Args:
            num_cells: Total cells at data level
            masking_strategy_config: Config for student cone, must contain:
                - 'rate': Fraction of sphere for student cone (e.g., 0.4 = 40%)
                - 'hl_mask': HEALPix level for cone generation
                - 'center_azimuth_degrees' (optional): Direction from teacher (0-360°).
                  If not specified, random direction is chosen.
            teacher_center_cell: HEALPix cell index of teacher cone center (at hl_mask level)
            center_distance_degrees: Angular distance between centers (in degrees, 0-180)

        Returns:
            Tuple of (student_mask, student_center_cell):
                - student_mask: Boolean mask with student cone (geodesic disk)
                - student_center_cell: HEALPix cell index of student cone center (at hl_mask level)

        Example:
            Teacher: cone with radius ~108° (rate=0.6) centered at cell 1000
            Student: cone with radius ~72° (rate=0.4)
            center_distance_degrees: 45°
            Result: Student cone centered 45° away from teacher
                   Overlap exists because 108° + 72° = 180° > 45° (cones intersect)
        """
        # ============================================================================
        # 1. Configuration and Setup
        # ============================================================================
        hl_mask = masking_strategy_config.get("hl_mask", 0)
        rate = masking_strategy_config.get("rate", 0.5)
        nside = 2**hl_mask
        num_parent_cells = 12 * (4**hl_mask)
        num_parents_to_keep = max(1, int(np.round(rate * num_parent_cells)))

        # ============================================================================
        # 2. Get Teacher Center Coordinates (lon, lat in radians)
        # ============================================================================
        teacher_lonlat = hp.healpix_to_lonlat(teacher_center_cell, nside, order="nested")
        # Handle astropy Quantity objects (extract float values)
        teacher_lon = float(
            teacher_lonlat[0].value if hasattr(teacher_lonlat[0], "value") else teacher_lonlat[0]
        )
        teacher_lat = float(
            teacher_lonlat[1].value if hasattr(teacher_lonlat[1], "value") else teacher_lonlat[1]
        )

        # ============================================================================
        # 3. Determine Azimuth (Direction from Teacher to Student)
        # ============================================================================
        center_azimuth_degrees = masking_strategy_config.get("center_azimuth_degrees", None)
        if center_azimuth_degrees is None:
            # Random direction if not specified
            center_azimuth_degrees = self.rng.uniform(0, 360)

        # Convert angles to radians for trigonometry
        distance_rad = np.deg2rad(center_distance_degrees)
        azimuth_rad = np.deg2rad(center_azimuth_degrees)

        # ============================================================================
        # 4. Calculate Student Center Using Spherical Trigonometry
        # ============================================================================
        # Great circle navigation formula: given starting point (lon, lat), distance, and azimuth,
        # compute destination point on sphere
        student_lat_rad = np.arcsin(
            np.sin(teacher_lat) * np.cos(distance_rad)
            + np.cos(teacher_lat) * np.sin(distance_rad) * np.cos(azimuth_rad)
        )

        student_lon_rad = teacher_lon + np.arctan2(
            np.sin(azimuth_rad) * np.sin(distance_rad) * np.cos(teacher_lat),
            np.cos(distance_rad) - np.sin(teacher_lat) * np.sin(student_lat_rad),
        )

        # Normalize longitude to [-π, π] range
        student_lon_rad = np.arctan2(np.sin(student_lon_rad), np.cos(student_lon_rad))

        # Convert student center coordinates to HEALPix cell index
        # (astropy_healpix expects Quantity objects with units)
        student_center_cell = hp.lonlat_to_healpix(
            student_lon_rad * u.rad, student_lat_rad * u.rad, nside, order="nested"
        )

        _logger.debug(
            f"Cone distance - Teacher center: cell {teacher_center_cell} "
            f"({np.rad2deg(teacher_lon):.1f}°, {np.rad2deg(teacher_lat):.1f}°), "
            f"Student center: cell {student_center_cell} "
            f"({np.rad2deg(student_lon_rad):.1f}°, {np.rad2deg(student_lat_rad):.1f}°), "
            f"Distance: {center_distance_degrees:.1f}°, Azimuth: {center_azimuth_degrees:.1f}°"
        )

        # ============================================================================
        # 5. Create Student Cone (Geodesic Disk)
        # ============================================================================
        # Select cells within angular radius from student center
        selected_parents = self._select_geodesic_disk(
            student_center_cell, num_parents_to_keep, nside, num_parent_cells
        )

        # ============================================================================
        # 6. Expand Parent Cells to Data Level
        # ============================================================================
        # Each parent cell at hl_mask level has 4^(level_diff) children at data level
        level_diff = self.healpix_level_data - hl_mask
        num_children_per_parent = 4**level_diff
        parent_ids = np.asarray(list(selected_parents))
        child_offsets = np.arange(num_children_per_parent)
        child_indices = (parent_ids[:, None] * num_children_per_parent + child_offsets).reshape(-1)

        # Create final mask at data level
        mask = np.zeros(num_cells, dtype=bool)
        mask[child_indices] = True

        return to_bool_tensor(mask), int(student_center_cell)

    def _create_fractional_overlap_mask(
        self,
        proposed_mask: torch.Tensor,
        target_mask: torch.Tensor,
        overlap_ratio: float,
        num_cells: int,
    ) -> torch.Tensor:
        """
        Create a student mask with deterministic fractional overlap with teacher.

        This method takes a proposed mask (typically a spatially contiguous crop from
        cropping_healpix strategy) and adjusts it to achieve exactly the specified overlap
        fraction with the teacher mask. The key innovation is maintaining spatial contiguity
        by preferentially selecting cells from the proposed crop.

        Strategy:
            1. Partition proposed crop into cells that overlap teacher vs. those that don't
            2. Select overlap cells preferentially from proposed∩teacher (maintains contiguity)
            3. Select non-overlap cells preferentially from proposed\teacher (maintains contiguity)
            4. Fall back to global selection only if proposed crop doesn't have enough cells

        Args:
            proposed_mask: Initial mask determining student crop size and shape (True = keep).
                          Typically a spatially contiguous crop from cropping_healpix.
            target_mask: Teacher mask to overlap with (True = keep)
            overlap_ratio: Fraction of student cells that should overlap with teacher [0.0-1.0]
                         - 0.0 = disjoint (no overlap)
                         - 0.5 = half of student cells come from teacher
                         - 1.0 = subset (complete overlap)
            num_cells: Total cells at data level

        Returns:
            Student mask with exact overlap_ratio with teacher, maintaining spatial structure
            of the proposed crop as much as possible

        Example:
            Teacher: 6144 cells (50% of sphere)
            Proposed student crop: 3712 cells (30% of sphere, spatially contiguous)
            overlap_ratio: 0.5
            Result: 1856 cells from proposed∩teacher + 1856 cells from proposed\teacher
                   (maintains spatial structure of proposed crop)
        """
        assert 0.0 <= overlap_ratio <= 1.0, f"overlap_ratio must be in [0.0, 1.0], got {overlap_ratio}"

        # ============================================================================
        # 1. Setup and Validation
        # ============================================================================
        # Convert to numpy for easier manipulation
        proposed_np = proposed_mask.cpu().numpy() if hasattr(proposed_mask, 'cpu') else proposed_mask
        target_np = target_mask.cpu().numpy() if hasattr(target_mask, 'cpu') else target_mask

        # Get cells from the proposed contiguous crop
        proposed_cells = np.where(proposed_np)[0]
        num_student_cells = len(proposed_cells)

        if num_student_cells == 0:
            _logger.warning("Proposed mask is empty, returning empty student mask")
            return to_bool_tensor(np.zeros(num_cells, dtype=bool))

        # Calculate target cell counts for overlap and non-overlap
        num_overlap_cells = int(np.round(overlap_ratio * num_student_cells))
        num_non_overlap_cells = num_student_cells - num_overlap_cells

        # ============================================================================
        # 2. Partition Proposed Crop by Teacher Overlap
        # ============================================================================
        # Split proposed cells into two sets based on teacher overlap
        # This is key to maintaining spatial contiguity while controlling overlap
        proposed_in_teacher = proposed_cells[target_np[proposed_cells]]  # proposed ∩ teacher
        proposed_not_in_teacher = proposed_cells[~target_np[proposed_cells]]  # proposed \ teacher

        # ============================================================================
        # 3. Select Overlap Cells (preferentially from proposed ∩ teacher)
        # ============================================================================
        # Primary selection: from proposed cells that overlap with teacher
        # This maintains spatial contiguity since proposed is a contiguous crop
        if num_overlap_cells > 0 and len(proposed_in_teacher) > 0:
            actual_overlap = min(num_overlap_cells, len(proposed_in_teacher))
            overlap_cells = self.rng.choice(
                proposed_in_teacher,
                size=actual_overlap,
                replace=False
            )
        else:
            overlap_cells = np.array([], dtype=int)
            actual_overlap = 0

        # Fallback: if proposed∩teacher doesn't have enough cells, select from teacher globally
        if actual_overlap < num_overlap_cells:
            teacher_cells = np.where(target_np)[0]
            additional_teacher = np.setdiff1d(teacher_cells, proposed_in_teacher)
            if len(additional_teacher) > 0:
                num_additional = min(num_overlap_cells - actual_overlap, len(additional_teacher))
                additional_cells = self.rng.choice(additional_teacher, size=num_additional, replace=False)
                overlap_cells = np.concatenate([overlap_cells, additional_cells])
                actual_overlap += num_additional

        # ============================================================================
        # 4. Select Non-Overlap Cells (preferentially from proposed \ teacher)
        # ============================================================================
        # Primary selection: from proposed cells that don't overlap with teacher
        # Again, this maintains spatial contiguity
        if num_non_overlap_cells > 0 and len(proposed_not_in_teacher) > 0:
            actual_non_overlap = min(num_non_overlap_cells, len(proposed_not_in_teacher))
            non_overlap_cells = self.rng.choice(
                proposed_not_in_teacher,
                size=actual_non_overlap,
                replace=False
            )
        else:
            non_overlap_cells = np.array([], dtype=int)
            actual_non_overlap = 0

        # Fallback: if proposed\teacher doesn't have enough cells, select from outside teacher globally
        if actual_non_overlap < num_non_overlap_cells:
            non_teacher_cells = np.where(~target_np)[0]
            additional_non = np.setdiff1d(non_teacher_cells, proposed_not_in_teacher)
            if len(additional_non) > 0:
                num_additional = min(num_non_overlap_cells - actual_non_overlap, len(additional_non))
                additional_cells = self.rng.choice(additional_non, size=num_additional, replace=False)
                non_overlap_cells = np.concatenate([non_overlap_cells, additional_cells])
                actual_non_overlap += num_additional

        # ============================================================================
        # 5. Create Final Student Mask and Validate
        # ============================================================================
        student_mask = np.zeros(num_cells, dtype=bool)
        student_mask[overlap_cells] = True
        student_mask[non_overlap_cells] = True

        # Diagnostics: log actual overlap achieved
        actual_student_size = len(overlap_cells) + len(non_overlap_cells)
        actual_overlap_ratio = len(overlap_cells) / actual_student_size if actual_student_size > 0 else 0.0

        _logger.debug(
            f"Fractional overlap - Target: {overlap_ratio:.1%}, "
            f"Actual: {actual_overlap_ratio:.1%} "
            f"({len(overlap_cells)}/{actual_student_size} cells)"
        )

        # Warning if overlap deviates significantly (may indicate incompatible teacher/student sizes)
        if abs(actual_overlap_ratio - overlap_ratio) > 0.05:
            _logger.warning(
                f"Overlap ratio deviation: target={overlap_ratio:.1%}, "
                f"actual={actual_overlap_ratio:.1%}. "
                f"This may happen if teacher/student sizes are incompatible."
            )

        return to_bool_tensor(student_mask)
