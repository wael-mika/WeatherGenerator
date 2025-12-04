import logging
from dataclasses import dataclass

import numpy as np
import torch

from weathergen.common.config import Config
from weathergen.datasets.batch import SampleMetaData

_logger = logging.getLogger(__name__)


# Convert to torch.bool
def to_bool_tensor(arr):
    return torch.from_numpy(np.asarray(arr)).to(torch.bool)


@dataclass
class MaskingStrategy:
    strategy: str
    config: dict
    num_samples: int


class Masker:
    """Class to generate masks for token sequences and apply them.
    This class supports different masking strategies and combinations.

    Attributes:
        masking_rate (float): The base rate at which tokens are masked.
        masking_strategy (str): The strategy used for masking (e.g., "random",
        "block", "healpix", "channel", "causal", "temporal_crop").
        current_strategy (str): The current strategy in use, relevant
                                when using "combination" strategy.

    Supported Masking Strategies:
        "random" - random masking of tokens at the level of the data

        "block" - masking out large blocks of tokens in 1D, without spatial meaning

        "forecast" - all tokens go to target (none kept in source)

        "healpix" - masking at the level of HEALPix cells, where all child cells
                    of a parent cell at a specific HEALpix level are masked
                    if the parent is masked.
                    The healpix level must be configured with hl_mask.
                    e.g. masking_strategy_config = {"hl_mask": 1}
                    with hl_mask the level for masking that we want to apply
                    e.g. level 1 very large cells masked

        "spatial" - spatial masking at cell level using generate_cell_keep_mask.
                    Same spatial mask applied to all timesteps.
                    Config options:
                    - cell_strategy: "random" or "healpix" (default: "random")
                    - hl_mask: required if cell_strategy="healpix"
                    e.g. masking_strategy_config = {"cell_strategy": "healpix", "hl_mask": 3}

        "spatiotemporal" - spatio-temporal masking with different spatial mask per timestep.
                           Each timestep gets independently sampled spatial mask.
                           Config options:
                           - cell_strategy: "random" or "healpix" (default: "random")
                           - hl_mask: required if cell_strategy="healpix"
                           e.g. masking_strategy_config = {"cell_strategy": "healpix", "hl_mask": 3}

        "channel" - masking data channels, where channels of the data are masked
                    can be done per-cell (each cell has different channels masked)
                    or globally (all have the same channels masked).
                    e.g. masking_strategy_config = {"mode": "per_cell"} or
                    {"mode": "global"}

        "causal" - masking the latest timesteps in each token, according to the masking rate.
                   Source gets early timesteps, target gets late timesteps.
                   Useful for autoregressive/causal modeling.
                   No additional config required.

        "temporal_crop" - keeps a specific temporal portion of the data based on crop_direction.
                          Source gets the selected portion, target gets the complement.
                          Required config:
                          - crop_direction: "start", "end", or "middle"
                          - masking_rate is used to determine fraction to keep in source
                          e.g. masking_strategy_config = {"crop_direction": "end"}
                          with masking_rate=0.3 keeps the last 30% of timesteps in source

        masking_rate_sampling (bool): Whether to sample the masking rate from a distribution.
        masking_strategy_config (dict): Configuration for the masking strategy, can include
                                        additional parameters like "hl_mask", "crop_direction", etc.
                                        specific to the masking strategy. See above.
    """

    def __init__(self, cf: Config):
        self.rng = None
        self.masking_rate = cf.masking_rate
        self.masking_strategy = cf.masking_strategy
        self.current_strategy = cf.masking_strategy  # Current strategy in use
        self.masking_rate_sampling = cf.masking_rate_sampling
        # masking_strategy_config is a dictionary that can hold any additional parameters
        self.healpix_level_data = cf.healpix_level
        self.masking_strategy_config = cf.get("masking_strategy_config", {})
        self.perm_sel = None
        self.mask_tokens = None
        self.mask_channels = None

        self.mask_value = 0.0
        self.dim_time_enc = 6

        # number of healpix cells
        self.healpix_num_cells = 12 * (4**self.healpix_level_data)

        # Per-batch strategy tracking
        self.same_strategy_per_batch = self.masking_strategy_config.get(
            "same_strategy_per_batch", False
        )
        self.batch_strategy_set = False

        # Check for required masking_strategy_config at construction time
        if self.current_strategy == "healpix":
            hl_data = self.healpix_level_data
            hl_mask = self.masking_strategy_config.get("hl_mask")
            assert hl_data is not None and hl_mask is not None, (
                "If HEALPix masking, hl_mask must be given in masking_strategy_config."
            )
            assert hl_mask < hl_data, "hl_mask must be less than hl_data for HEALPix masking."

        if self.current_strategy == "channel":
            # Ensure that masking_strategy_config contains either 'global' or 'per_cell'
            assert self.masking_strategy_config.get("mode") in [
                "global",
                "per_cell",
            ], "masking_strategy_config must contain 'mode' key with value 'global' or 'per_cell'."

            # check all streams that source and target channels are identical
            for stream in cf.streams:
                # check explicit includes
                source_include = stream.get("source_include", [])
                target_include = stream.get("target_include", [])
                assert set(source_include) == set(target_include), (
                    "Source and target channels not identical. Required for masking_mode=channel"
                )
                # check excludes
                source_exclude = stream.get("source_exclude", [])
                target_exclude = stream.get("target_exclude", [])
                assert set(source_exclude) == set(target_exclude), (
                    "Source and target channels not identical. Required for masking_mode=channel"
                )

        if self.current_strategy == "temporal_crop":
            # Ensure that crop_direction is specified
            crop_direction = self.masking_strategy_config.get("crop_direction")
            assert crop_direction in ["start", "end", "middle"], (
                "temporal_crop strategy requires 'crop_direction' in masking_strategy_config "
                "with value 'start', 'end', or 'middle'."
            )

        if self.current_strategy in ["spatial", "spatiotemporal"]:
            # Validate spatial/spatiotemporal strategy config
            cell_strategy = self.masking_strategy_config.get("cell_strategy", "random")
            assert cell_strategy in ["random", "healpix"], (
                f"{self.current_strategy} strategy requires 'cell_strategy' to be 'random' or 'healpix', "
                f"got '{cell_strategy}'"
            )
            if cell_strategy == "healpix":
                hl_mask = self.masking_strategy_config.get("hl_mask")
                assert hl_mask is not None and hl_mask < self.healpix_level_data, (
                    f"{self.current_strategy} with cell_strategy='healpix' requires 'hl_mask' "
                    f"in masking_strategy_config and hl_mask < data level {self.healpix_level_data}"
                )

    def reset_rng(self, rng) -> None:
        """
        Reset rng after mini_epoch to ensure proper randomization
        """
        self.rng = rng

    def set_batch_strategy(self):
        """
        Set strategy for this batch.
        Only relevant with combination and same_strategy_per_batch.
        """
        if self.masking_strategy == "combination" and self.same_strategy_per_batch:
            self.current_strategy = self.rng.choice(
                self.masking_strategy_config["strategies"],
                p=self.masking_strategy_config["probabilities"],
            )
            self.batch_strategy_set = True

    def reset_batch_strategy(self):
        """
        Reset for next batch.
        """
        if self.masking_strategy == "combination" and self.same_strategy_per_batch:
            self.current_strategy = None
            self.batch_strategy_set = False

    def _select_strategy(self):
        """
        Select the strategy to use.
        """
        if self.masking_strategy == "combination":
            if self.same_strategy_per_batch:
                assert self.batch_strategy_set, "Must call set_batch_strategy() first"
                return self.current_strategy
            else:
                # Sample new strategy for each stream
                return self.rng.choice(
                    self.masking_strategy_config["strategies"],
                    p=self.masking_strategy_config["probabilities"],
                )
        else:
            # Non-combination strategy, return as is
            return self.masking_strategy


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


    def _generate_causal_mask_idxs(
        self,
        idxs_cells_lens: list[list[int]],
        rate: float,
    ) -> np.typing.NDArray:
        """
        Generates a causal mask at the index level, masking the latest times
        in each cell according to the masking rate.

        Args:
            idxs_cells_lens: List of lists of token lengths per cell
            rate: Fraction of timesteps to MASK (goes to target)

        Returns:
            np.ndarray: Boolean mask where True = KEEP in source, False = MASK for target
        """
        if not idxs_cells_lens:
            return np.array([], dtype=bool)

        # Extract all lengths at once
        token_lens = np.array([len(lens_cell) for lens_cell in idxs_cells_lens])

        if len(token_lens) == 0:
            return np.array([], dtype=bool)

        # Calculate start indices for masking
        # astype(int) performs floor operation by truncation
        num_future_to_mask = (rate * token_lens).astype(int)
        start_mask_indices = np.maximum(1, token_lens - num_future_to_mask)

        # Handle edge cases
        mask_valid = token_lens > 1  # Only cells with >1 timestep can be masked
        start_mask_indices = np.where(mask_valid, start_mask_indices, token_lens)

        # Create masks with list comprehension
        # True = KEEP in source, False = MASK for target
        full_mask = []
        for token_len, start_idx in zip(token_lens, start_mask_indices, strict=True):
            if token_len > 1:
                # Keep early timesteps, mask late ones
                mask = np.concatenate([
                    np.ones(start_idx, dtype=bool),
                    np.zeros(max(0, token_len - start_idx), dtype=bool),
                ])
            elif token_len == 1:
                mask = np.ones(1, dtype=bool)  # Keep single timestep
            else:
                mask = np.array([], dtype=bool)
            full_mask.append(mask)

        return np.concatenate(full_mask) if full_mask else np.array([], dtype=bool)

    def _generate_temporal_crop_mask_idxs(
        self,
        idxs_cells_lens: list[list[int]],
        temporal_config: dict,
    ) -> np.typing.NDArray:
        """
        Generates a temporal cropping mask at the index level that KEEPS selected timesteps.

        Args:
            idxs_cells_lens: List of lists of token lengths per cell
            temporal_config: Dict with 'crop_direction' ("start", "end", "middle")
                             and 'rate' (fraction of timesteps to KEEP)

        Returns:
            np.ndarray: Boolean mask where True = KEEP in source, False = MASK for target
        """
        if not idxs_cells_lens:
            return np.array([], dtype=bool)

        crop_direction = temporal_config.get("crop_direction", "end")
        rate = temporal_config.get("rate", 0.5)

        assert crop_direction in {"start", "end", "middle"}, (
            f"crop_direction must be 'start', 'end', or 'middle', got {crop_direction}"
        )

        # Extract all lengths at once
        token_lens = np.array([len(lens_cell) for lens_cell in idxs_cells_lens])

        if len(token_lens) == 0:
            return np.array([], dtype=bool)

        # Calculate how many timesteps to keep per cell
        num_to_keep = np.maximum(1, (rate * token_lens).astype(int))

        # Create masks based on crop direction
        full_mask = []
        for token_len, n_keep in zip(token_lens, num_to_keep, strict=True):
            if token_len == 0:
                full_mask.append(np.array([], dtype=bool))
                continue

            # Ensure we don't try to keep more than we have
            n_keep = min(n_keep, token_len)

            # Create mask based on direction (True = KEEP in source)
            mask = np.zeros(token_len, dtype=bool)

            if crop_direction == "start":
                # Keep first n_keep timesteps in source
                mask[:n_keep] = True
            elif crop_direction == "end":
                # Keep last n_keep timesteps in source
                mask[-n_keep:] = True
            else:  # middle
                # Keep middle n_keep timesteps in source
                start_idx = (token_len - n_keep) // 2
                mask[start_idx : start_idx + n_keep] = True

            full_mask.append(mask)

        return np.concatenate(full_mask) if full_mask else np.array([], dtype=bool)

    def _generate_temporal_mask(
        self,
        idxs_cells_lens: list[list[int]],
        strategy: str,
        rate: float,
        masking_strategy_config: dict,
        target_mask: np.typing.NDArray | None = None,
        target_mask_metadata: dict | None = None,
        relationship: str = "subset",
    ) -> np.typing.NDArray:
        """
        Generate temporal mask at token level (after tokenization).

        This method generates masks for temporal strategies (causal, temporal_crop)
        that require knowledge of the temporal structure of tokens.

        Args:
            idxs_cells_lens: List of lists of token lengths per cell (from tokenization)
            strategy: "causal" or "temporal_crop"
            rate: Masking rate (fraction to MASK for target)
            masking_strategy_config: Strategy-specific config (e.g., crop_direction)
            target_mask: Optional token-level target mask for relationship constraints
            target_mask_metadata: Optional metadata if target is also temporal
            relationship: "complement", "subset", or "disjoint"

        Returns:
            np.ndarray: Boolean mask [num_tokens] where True = KEEP in source
        """
        # Handle complement relationship with temporal target
        if target_mask_metadata is not None and relationship == "complement":
            # If target is temporal, we need to generate its mask first then complement
            target_mask = self._generate_temporal_mask(
                idxs_cells_lens,
                target_mask_metadata["strategy"],
                target_mask_metadata["rate"],
                target_mask_metadata["config"],
                None,
                None,
                "subset",
            )

        # Generate base temporal mask based on strategy
        if strategy == "causal":
            # Mask the LATEST timesteps (rate = fraction to MASK)
            mask = self._generate_causal_mask_idxs(idxs_cells_lens, rate)

        elif strategy == "temporal_crop":
            # Keep timesteps based on crop_direction (rate = fraction to MASK)
            temporal_config = {
                "crop_direction": masking_strategy_config.get("crop_direction", "end"),
                "rate": 1.0 - rate,  # Convert from mask rate to keep rate
            }
            mask = self._generate_temporal_crop_mask_idxs(idxs_cells_lens, temporal_config)

        else:
            raise ValueError(f"Unsupported temporal strategy: {strategy}")

        # Apply relationship with target_mask if provided
        if target_mask is not None:
            if relationship == "complement":
                mask = ~target_mask
            elif relationship == "subset":
                mask = mask & target_mask
            elif relationship == "disjoint":
                mask = mask & (~target_mask)

        return mask

    def build_samples_for_stream(
        self,
        training_mode: str,
        num_cells: int,
        target_cfg: dict,
        source_cfg: dict,
    ) -> tuple[tuple, tuple, np.typing.NDArray]:
        """
        Construct teacher/student keep masks for a stream.
        SampleMetaData is currently just a dict with the masking params used.

        Returns:
            Tuple of:
                - (target_masks, target_metadata_list, target_mask_metadata_list)
                - (source_masks, source_metadata_list, source_mask_metadata_list)
                - source_target_mapping
            where:
                - masks are cell-level masks (or None for temporal strategies)
                - mask_metadata_list contains dicts for deferred temporal mask generation
        """

        # get source and target configs; target defaults to source config

        source_num_samples = source_cfg.get("num_samples", 1)
        source_strategy = source_cfg.get("masking_strategy", source_cfg.get("strategy", "random"))
        source_masking_params = source_cfg.get("masking_strategy_config")
        relationship = source_cfg.get("relationship", "complement")

        if target_cfg is not None:
            target_num_samples = target_cfg.get("num_samples", 1)
            target_strategy = target_cfg.get("strategy", "random")
            target_masking_params = target_cfg.get("masking_strategy_config")
        else:
            target_strategy = source_strategy
            target_num_samples = source_num_samples
            target_masking_params = source_masking_params
            # # do other relationships make sense
            # assert relationship == "complement"

        assert source_num_samples % target_num_samples == 0, (
            "number of source samples has to be multiple of target samples"
        )

        # translate settings into sampling masks

        # iterate over all target samples
        target_masks: list[np.typing.NDArray | None] = []
        target_metadata: list[SampleMetaData] = []
        target_mask_metadata_list: list[dict | None] = []
        for _ in range(target_num_samples):
            mask, mask_metadata = self._get_mask(
                num_cells=num_cells,
                strategy=target_strategy,
                target_mask=None,
                target_mask_metadata=None,
                masking_strategy_config=target_masking_params,
            )
            target_masks.append(mask)
            target_mask_metadata_list.append(mask_metadata)
            target_metadata.append(SampleMetaData(params=target_cfg))

        # iterate over all source samples
        source_masks: list[np.typing.NDArray | None] = []
        source_metadata: list[SampleMetaData] = []
        source_mask_metadata_list: list[dict | None] = []
        source_target_mapping = np.zeros(source_num_samples, dtype=np.int32)
        for it in range(source_num_samples):
            mask, mask_metadata = self._get_mask(
                num_cells=num_cells,
                strategy=source_strategy,
                masking_strategy_config=source_masking_params,
                target_mask=target_masks[it % target_num_samples],
                target_mask_metadata=target_mask_metadata_list[it % target_num_samples],
                relationship=relationship,
            )
            source_masks.append(mask)
            source_mask_metadata_list.append(mask_metadata)
            source_metadata.append(SampleMetaData(params=source_cfg))
            source_target_mapping[it] = it % target_num_samples

        return (
            (target_masks, target_metadata, target_mask_metadata_list),
            (source_masks, source_metadata, source_mask_metadata_list),
            source_target_mapping,
        )

    def _get_mask(
        self,
        num_cells: int,
        strategy: str | None = None,
        rate: float | None = None,
        masking_strategy_config: dict | None = None,
        target_mask: np.typing.NDArray | None = None,
        target_mask_metadata: dict | None = None,
        relationship: str = "subset",
    ) -> tuple[np.typing.NDArray | None, dict | None]:
        """Get effective mask, combining with target mask if specified.

        Parameters
        ----------
        num_cells : int
            Number of cells at data level (should equal 12 * 4**healpix_level).
        strategy : str | None
            Cell selection strategy: supports 'random', 'healpix' (spatial) and
            'causal', 'temporal_crop' (temporal). Uses instance default if None.
        rate : float | None
            Fraction of parent cells (healpix) or data cells (random) to keep. Falls back
            to instance masking_rate if None.
        masking_strategy_config : dict | None
            Optional override of strategy config (e.g., {'hl_mask': 3}).
        target_mask : np.ndarray | None
            Optional target mask for relationship constraints (for cell-level strategies).
        target_mask_metadata : dict | None
            Optional target mask metadata (for temporal strategies).
        relationship : str
            How to combine with target_mask: "complement", "subset", or "disjoint".

        Returns
        -------
        tuple[np.ndarray | None, dict | None]
            - For spatial strategies: (cell_mask, None) where cell_mask is [num_cells]
            - For temporal strategies: (None, metadata_dict) to defer mask generation
        """

        strat = strategy or self.masking_strategy
        cfg = masking_strategy_config or self.masking_strategy_config
        keep_rate = rate if rate is not None else self.masking_rate

        # sample rate if requested (only if explicit rate not provided)
        if rate is None and self.masking_rate_sampling:
            keep_rate = self._get_sampling_rate()

        # Check if this is a temporal strategy (needs deferred generation)
        temporal_strategies = {"causal", "temporal_crop"}
        if strat in temporal_strategies:
            # Return metadata for deferred generation after tokenization
            metadata = {
                "is_temporal": True,
                "strategy": strat,
                "rate": keep_rate,
                "config": cfg,
                "relationship": relationship,
                "target_mask_metadata": target_mask_metadata,
            }
            return (None, metadata)

        # Spatial strategies: generate mask immediately
        # handle cases where mask is directly derived from target_mask
        if target_mask is not None:
            if relationship == "complement":
                mask = ~target_mask
                return (mask, None)

        # get mask
        mask = self._generate_cell_mask(num_cells, strategy, rate, masking_strategy_config)

        # handle cases where mask needs to be combined with target_mask
        if target_mask is not None:
            if relationship == "subset":
                mask = mask & target_mask
            elif relationship == "disjoint":
                mask = mask & (~target_mask)

        return (mask, None)

    def _generate_cell_mask(
        self,
        num_cells: int,
        strategy: str | None = None,
        rate: float | None = None,
        masking_strategy_config: dict | None = None,
    ) -> np.typing.NDArray:
        """Generate a boolean keep mask at data healpix level (True = keep cell).

        Parameters
        ----------
        num_cells : int
            Number of cells at data level (should equal 12 * 4**healpix_level).
        strategy : str | None
            Cell selection strategy: currently supports 'random' and 'healpix'. Uses
            instance default if None.
        rate : float | None
            Fraction of parent cells (healpix) or data cells (random) to keep. Falls back
            to instance masking_rate if None.
        masking_strategy_config : dict | None
            Optional override of strategy config (e.g., {'hl_mask': 3}).
        constraint_keep_mask : np.ndarray | None
            Optional boolean mask of allowed cells (True = allowed). Selection will be
            limited to these cells. For subset/disjoint relationships.

        Returns
        -------
        np.ndarray
            Boolean array of shape [num_cells] where True indicates the cell is kept.
        """

        # get config for mask

        strat = strategy or self.masking_strategy
        cfg = masking_strategy_config or self.masking_strategy_config
        keep_rate = rate if rate is not None else self.masking_rate

        # sample rate if requested (only if explicit rate not provided)
        if rate is None and self.masking_rate_sampling:
            keep_rate = self._get_sampling_rate()

        assert 0.0 <= keep_rate <= 1.0, f"keep_rate out of bounds: {keep_rate}"
        assert num_cells == self.healpix_num_cells, (
            "num_cells inconsistent with configured healpix level."
        )

        if strat not in {"random", "healpix"}:
            raise NotImplementedError(
                f"Cell selection strategy '{strat}' not supported for keep mask generation."
            )

        # generate cell mask

        if strat == "random":
            mask = self.rng.uniform(0, 1, num_cells) < keep_rate

        elif strat == "forecast" or strat == "causal":
            mask = np.ones(num_cells, dtype=np.bool)

        elif strat == "healpix":
            hl_data = self.healpix_level_data
            hl_mask = cfg.get("hl_mask")
            assert hl_mask is not None and hl_mask < hl_data, (
                "For healpix keep mask generation, cfg['hl_mask'] must be set and < data level."
            )
            num_parent_cells = 12 * (4**hl_mask)
            level_diff = hl_data - hl_mask
            num_children_per_parent = 4**level_diff
            # number of parents to KEEP
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

        else:
            assert False, "Unknown strategy."

        mask = to_bool_tensor(mask)

        return mask
