import copy
import logging
import warnings

import astropy_healpix as hp
import numpy as np
import omegaconf
import torch
from numpy.typing import NDArray

from weathergen.common.config import Config
from weathergen.datasets.batch import SampleMetaData
from weathergen.train.utils import Stage
from weathergen.utils.utils import is_stream_diagnostic, is_stream_forcing

logger = logging.getLogger(__name__)


class MaskData:
    masks: list[np.typing.NDArray] = []
    metadata: list[SampleMetaData] = []
    # Per-mask optional channel drop mask: shape (num_channels,), True = keep.
    channel_drop_masks: list[np.typing.NDArray | None] = []
    # Per-mask optional per-group spatial masks: {group_name: (num_cells,) bool}.
    group_spatial_masks: list[dict[str, np.typing.NDArray] | None] = []

    def __init__(self):
        self.masks = []
        self.metadata = []
        self.channel_drop_masks = []
        self.group_spatial_masks = []

    def __len__(self):
        return len(self.masks)

    def add_mask(
        self,
        mask,
        params,
        cfg,
        losses,
        idx,
        correspondence,
        relationship,
        channel_drop_mask=None,
        group_spatial_masks=None,
    ):
        self.masks += [mask]
        self.channel_drop_masks += [channel_drop_mask]
        self.group_spatial_masks += [group_spatial_masks]
        self.metadata += [
            SampleMetaData(
                params={**cfg, **params},
                mask=mask,
                global_params={
                    "idx": idx,
                    "correspondence": correspondence,
                    "loss": losses,
                    "relationship": relationship,
                },
            )
        ]

    def get_mask(self, idx: int) -> np.typing.NDArray:
        return self.masks[idx]

    def get_channel_drop_mask(self, idx: int) -> np.typing.NDArray | None:
        return self.channel_drop_masks[idx]

    def get_group_spatial_masks(self, idx: int) -> dict[str, np.typing.NDArray] | None:
        return self.group_spatial_masks[idx]


def get_num_samples(config) -> np.typing.NDArray:
    """
    Get number of samples in source/target config
    """
    return np.array([s_cfg.get("num_samples", 1) for _, s_cfg in config.items()])


def _is_forecast_like(strategy: str) -> bool:
    """Return True for forecast-type masking strategies (``"forecast*"`` or ``"causal"``)."""
    return "forecast" in strategy or strategy == "causal"


def _filter_active_configs(cfgs) -> dict:
    """Return plain dict of config entries that are enabled and have num_samples > 0.

    An entry is active when:
    - ``enabled`` is absent or not False
    - ``num_samples`` is absent (defaults to 1) or > 0

    Accepts both OmegaConf DictConfig and plain dict.
    Guards against ListConfig([]) returned when the YAML key is absent.
    Returns a plain dict so downstream code can safely iterate and mutate.
    """
    if not cfgs:
        return {}
    return {
        k: v
        for k, v in cfgs.items()
        if v.get("enabled", True) is not False and v.get("num_samples", 1) != 0
    }


def validate_correspondence_mode(correspondence_mode, target_cfgs, source_cfgs):
    """
    Validate that the configs are consistent with the correspondence mode
    """

    num_target_samples = np.array([t.get("num_samples", 1) for t in target_cfgs]).sum()
    num_source_samples = np.array([s.get("num_samples", 1) for s in source_cfgs]).sum()

    if correspondence_mode == "one-to-one":
        assert len(target_cfgs) == len(source_cfgs), (
            "With target_correspondence_mode mode one-to-one, number of source and target "
            + "strategies has to match."
        )
        assert num_target_samples.item() == num_source_samples.item(), (
            "With target_correspondence_mode mode one-to-one, number of source and target "
            + "samples has to match."
        )

    if correspondence_mode == "equal-split-all":
        assert num_source_samples.item() % num_target_samples.item() == 0, (
            "With target_correspondence_mode mode equal-split-all, number of source samples "
            + "has to be divisible by number of target samples."
        )


# Convert to torch.bool
def to_bool_tensor(arr):
    return torch.from_numpy(np.asarray(arr)).to(torch.bool)


class Masker:
    """Class to generate masks for token sequences and apply them.
    This class supports different masking strategies and combinations.

    Attributes:
        masking_rate (float): The base rate at which tokens are masked.
        masking_strategy (str): The strategy used for masking (e.g., "random",
        "healpix", "cropping_healpix").
        current_strategy (str): The current strategy in use, relevant
                                when using "combination" strategy.
        "random" - random masking of tokens at the level of the data
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
        "satellite_swath" - simulate polar-orbiting satellite ground tracks. Each orbit produces
                    a swath that is a diagonal band in lat-lon space: the Earth rotates beneath
                    the satellite, so consecutive pole-to-pole passes shift westward in longitude.
                    Cells within swath_width_deg of any pass centerline are kept; all others are
                    masked. Unlike purely zonal or purely meridional stripes, swaths span the full
                    latitude range while varying in longitude, forcing the model to learn both
                    zonal and meridional structure simultaneously.
                    e.g. masking_strategy_config = {"num_swaths": 5, "swath_width_deg": 20,
                                                     "orbit_drift_deg": -25}
                    num_swaths: number of orbital passes (each at a random starting longitude)
                    swath_width_deg: total width of each swath in great-circle degrees at equator
                    orbit_drift_deg: longitude drift from south to north pole (~-25 for LEO polar)
        "satellite_swath_sparse" - like satellite_swath but with more, thinner swaths and
                    random subsampling within each swath footprint (at rate keep_rate).
                    Cells outside all swaths are always masked; cells inside swaths are kept
                    with probability rate. Combines orbital spatial structure with random dropout
                    inside each pass — forcing the model to handle both geographic coverage gaps
                    and fine-grained sparse observations simultaneously.
                    e.g. masking_strategy_config = {"num_swaths": 12, "swath_width_deg": 10,
                                                     "orbit_drift_deg": -25, "rate": 0.6}
                    num_swaths: number of orbital passes (more passes, thinner → denser coverage)
                    swath_width_deg: total swath width in great-circle degrees at equator
                    orbit_drift_deg: longitude drift from south to north pole
                    rate: fraction of in-swath cells to keep (rate_sampling supported)
        masking_rate_sampling (bool): Whether to sample the masking rate from a distribution.
        masking_strategy_config (dict): Configuration for the masking strategy, can include
                                        additional parameters like "hl_mask", etc.
                                        specific to the masking strategy. See above.
    """

    def __init__(self, healpix_level: int, stage: Stage, streams=None, mode_cfg=None):
        self.rng = None

        self.mask_value = 0.0
        self.dim_time_enc = 6

        # number of healpix cells
        self.healpix_level_data = healpix_level
        self.healpix_num_cells = 12 * (4**healpix_level)

        self.stage = stage

        # Build and store per-stream effective masking configs
        if streams is not None and mode_cfg is not None:
            self._effective_masking_cfgs = self.build_effective_masking_cfgs(streams, mode_cfg)
        else:
            self._effective_masking_cfgs = {}

    def reset_rng(self, rng) -> None:
        """
        Reset rng after mini_epoch to ensure proper randomization
        """
        self.rng = rng

    def _get_cell_coords(self) -> tuple[NDArray, NDArray]:
        """Return (lats, lons) in degrees for every HEALPix cell (cached).

        lats: [-90, 90], lons: [-180, 180].
        Uses NESTED ordering to match the data pipeline.
        Computed once and reused across all masking calls.
        """
        if not hasattr(self, "_cell_coords_cache"):
            nside = 2**self.healpix_level_data
            all_idx = np.arange(self.healpix_num_cells)
            lonlat = hp.healpix_to_lonlat(all_idx, nside, order="nested")
            lon, lat = lonlat[0], lonlat[1]
            # astropy returns Angle objects; .value gives radians
            lat_rad = lat.value if hasattr(lat, "value") else lat
            lon_rad = lon.value if hasattr(lon, "value") else lon
            lats = np.degrees(lat_rad)  # [-90, 90]
            lons = (np.degrees(lon_rad) + 180.0) % 360.0 - 180.0  # [-180, 180]
            self._cell_coords_cache = (lats, lons)
        return self._cell_coords_cache

    def _get_cell_lats(self) -> NDArray:
        """Return latitude in degrees [-90, 90] for every HEALPix cell (cached)."""
        return self._get_cell_coords()[0]

    def _get_xyz_at_hl_level(self, hl: int) -> NDArray:
        """Return (N, 3) Cartesian unit vectors for every HEALPix cell at level *hl* (cached).

        Results are cached per level so forked data workers compute this at most once.
        """
        if not hasattr(self, "_xyz_cache"):
            self._xyz_cache: dict[int, NDArray] = {}
        if hl not in self._xyz_cache:
            nside = 2**hl
            num_cells = 12 * nside * nside
            all_idx = np.arange(num_cells)
            lonlat = hp.healpix_to_lonlat(all_idx, nside, order="nested")
            lon = lonlat[0].value if hasattr(lonlat[0], "value") else lonlat[0]
            lat = lonlat[1].value if hasattr(lonlat[1], "value") else lonlat[1]
            xyz = np.stack(
                [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)],
                axis=1,
            )
            self._xyz_cache[hl] = xyz
        return self._xyz_cache[hl]

    def _resolve_mixed_strategies(self, cfgs: dict) -> dict:
        """Replace any 'mixed' masking strategy with a single randomly-chosen sub-strategy.

        Draws the sub-strategy exactly once per config entry so that source and target
        loops share the same sub-strategy for a given sample.  Without this, both loops
        would call ``_generate_cell_mask("mixed", ...)`` independently and could draw
        different sub-strategies (e.g. target draws "random", source draws "cropping").

        Args:
            cfgs: Deep copy of a masking config section (model_input dict). Modified in-place.

        Returns:
            The modified cfgs dict with "mixed" replaced by a concrete sub-strategy.
        """
        for _, cfg in cfgs.items():
            if cfg.get("masking_strategy") != "mixed":
                continue
            msc = cfg.get("masking_strategy_config", {})
            strategies = list(msc.get("strategies", ["random", "healpix"]))
            raw_weights = msc.get("strategy_weights", None)
            if raw_weights is not None:
                w = np.array([raw_weights.get(s, 1.0) for s in strategies], dtype=float)
                w /= w.sum()
            else:
                w = None
            chosen_idx = self.rng.choice(len(strategies), p=w)
            chosen = strategies[chosen_idx]

            strategy_cfgs = msc.get("strategy_configs", {})
            sub_cfg = {**msc, **strategy_cfgs.get(chosen, {})}

            if "hl_mask_levels" in msc and "hl_mask" not in strategy_cfgs.get(chosen, {}):
                hl_mask_levels = list(msc["hl_mask_levels"])
                hl_mask = hl_mask_levels[self.rng.integers(0, len(hl_mask_levels))]
                sub_cfg["hl_mask"] = hl_mask

            cfg["masking_strategy"] = chosen
            cfg["masking_strategy_config"] = sub_cfg
        return cfgs

    def merge_masking_config(self, mode_cfg, override):
        """Merge a stream's masking override into the base mode config.

        Only masking strategy fields are overridden. Structural keys like
        ``num_samples`` and ``num_steps_input`` remain unchanged.

        The override is flat per section (``model_input`` / ``target_input``),
        not per named strategy.  If a section has multiple strategies (e.g.
        ``"input_physical"`` and ``"input_jepa"``), masking strategy fields are
        broadcast to all of them.  ``randomly_drop_as_source_rate`` is a
        per-stream rate; the drop decision is made once per call to
        ``build_samples_for_stream`` and applies to all source strategies
        uniformly (training only).

        Expected YAML in a stream config, e.g.:

            STREAM_NAME:
              type: ...
              filenames: ...
              ...
              masking_override:
                target_input:
                  masking_strategy_config:
                    hl_mask: 3
              ...

        This overrides only ``hl_mask`` within ``masking_strategy_config`` for
        every target strategy, inheriting rate, rate_sampling, etc. from the
        global config.  ``masking_strategy`` itself can also be replaced.
        """

        stream_cfg = copy.deepcopy(mode_cfg)

        # Copy top-level masking keys from override
        if "randomly_drop_as_source_rate" in override:
            stream_cfg["randomly_drop_as_source_rate"] = override["randomly_drop_as_source_rate"]

        for section_key in ("model_input", "target_input"):
            section = stream_cfg.get(section_key, {})
            override_values = override.get(section_key)

            # Materialize target_input from model_input ONLY when the stream explicitly
            # overrides the target section (e.g. era5_out forcing target rate to 1.0).
            # An absent target_input is meaningful: it selects Mode A/B in
            # build_samples_for_stream (target auto-generated as the source complement).
            # Unconditionally copying here would make every stream look like Mode C
            # (explicit target) and silently disable the MAE complement correspondence.
            if section == {} and section_key == "target_input":
                if override_values is None:
                    continue
                # by the processing order of "model_input" and "target_input", the target_input
                # here will have stream specific model_input overrides
                stream_cfg["target_input"] = copy.deepcopy(stream_cfg.get("model_input", {}))
                section = stream_cfg["target_input"]

            if override_values is None:
                continue

            for strategy_cfg in section.values():
                if "masking_strategy" in override_values:
                    strategy_cfg["masking_strategy"] = override_values["masking_strategy"]
                if "masking_strategy_config" in override_values:
                    strategy_cfg["masking_strategy_config"] = omegaconf.OmegaConf.merge(
                        strategy_cfg.get("masking_strategy_config", omegaconf.OmegaConf.create({})),
                        override_values["masking_strategy_config"],
                    )

        return stream_cfg

    def build_effective_masking_cfgs(self, streams: Config, mode_cfg):
        """Build effective masking configs for all streams."""
        cfgs = {}
        for stream_name, stream_info in streams.items():
            override = stream_info.get("masking_override", {})
            cfgs[stream_name] = self.merge_masking_config(mode_cfg, override)

        return cfgs

    def _get_sampling_rate(self, cfg):
        """
        Get the sampling rate, optionally sampled from a distribution.

        Supported distributions (``rate_distribution``):
        - ``"normal"`` (default): clip(|N(rate, 1/(2.5π))|, 0.01, 0.99)
        - ``"beta"``: Beta(rate_alpha, rate_beta) where rate_beta defaults to
          ``rate_alpha * (1 - rate) / rate`` so the distribution mean equals ``rate``.
          Set ``rate_flip: true`` to sample ``1 - Beta(alpha, beta)`` — useful when
          the convention switches from keep_rate to masking_rate, since flipping
          preserves the distribution shape while mirroring it around 0.5.
        """

        rate = cfg.get("rate", None)
        assert rate is not None, 'No sampling rate "rate" specified.'

        if cfg.get("rate_sampling", False):
            dist = cfg.get("rate_distribution", "normal")
            if dist == "normal":
                rate = np.clip(
                    np.abs(self.rng.normal(loc=rate, scale=1.0 / (2.5 * np.pi))),
                    0.01,
                    0.99,
                )
            elif dist == "beta":
                alpha = float(cfg.get("rate_alpha", 2.0))
                # Auto-compute beta so mean = rate; override with rate_beta if provided.
                default_beta = alpha * (1.0 - rate) / max(rate, 1e-9)
                beta = float(cfg.get("rate_beta", default_beta))
                sampled = float(self.rng.beta(alpha, beta))
                if cfg.get("rate_flip", False):
                    sampled = 1.0 - sampled
                rate = float(np.clip(sampled, 0.01, 0.99))
            else:
                raise ValueError(
                    f"Unknown rate_distribution {dist!r}. Supported: 'normal', 'beta'."
                )

        assert 0.0 <= rate <= 1.0, f"keep_rate out of bounds: {rate}"

        return rate

    def get_target_rel_mask(self, target_masks, masking_config):
        """
        Get target relationship strategy and target mask
        """
        relationship = masking_config.get("target_relationship", {"independent": None})
        assert len(relationship) == 1, "Only one target_relationship supported."

        target_idx = list(relationship.values())[0]

        target_relationship_mask = (
            list(relationship.keys())[0],  # target relationship strategy
            target_masks.get_mask(target_idx),  # target mask
        )

        return target_relationship_mask, target_idx

    def parse_src_target_correspondence(self, losses, target_cfgs, source_cfgs) -> dict:
        """
        Parses losses and obtain consolidated source -> target correspondence dict
        """

        # collect target-source correspondence for all loss terms
        corrs = []
        for _, loss_term in losses.items():
            for loss_name, loss_fct in loss_term.loss_fcts.items():
                corr = loss_fct.get("target_source_correspondence", None)

                # correspondence not specified; falling back to default 1-to-1 correspondence
                # at the level of the configs
                if corr is None:
                    assert len(target_cfgs) == len(source_cfgs), (
                        "No source/target correspondence specified but number of source and target "
                        + "configs also not matching."
                    )
                    corr = dict([(i, i) for i in range(len(target_cfgs))])

                corr_dict = {}
                for target_idx, source_spec in corr.items():
                    # process into common long format
                    target_idx = int(target_idx)
                    if type(source_spec) is omegaconf.dictconfig.DictConfig:
                        # TODO: check format of dict
                        # append loss_name
                        corr_dict[target_idx] = dict(
                            [(int(k), (v, loss_name)) for k, v in source_spec.items()]
                        )
                    elif type(source_spec) is omegaconf.listconfig.ListConfig:
                        corr_dict[target_idx] = dict(
                            [(int(v), (None, loss_name)) for v in source_spec]
                        )
                    elif type(source_spec) is int:
                        corr_dict[target_idx] = {source_spec: (None, loss_name)}
                    else:
                        assert False, (
                            "Invalid target_source_correspondence specification. Needs to be "
                            + "integer corresponding to a specific source, list of source or a "
                            + "dictionary specifying the correspondence."
                        )

                corrs += [corr_dict]

        # check that all target/sources indices are ints; conf can have type mismatches due to
        # conf merging
        are_ints = np.array(
            [
                [type(k) is int and type(next(iter(v.keys()))) is int for k, v in corr.items()]
                for corr in corrs
            ]
        ).all()
        assert are_ints, "error parsing correspondence, all indices must be int"

        # merge correspondences
        corr_dict = {}
        for k_target in range(len(target_cfgs)):
            # require identical relationship type when target has same source correspondence in
            # different loss terms
            vs = [c.get(k_target) for c in corrs if c.get(k_target) is not None]
            vs_ks_unique = list(set([kk for v in vs for kk in list(v.keys())]))
            for k_source in vs_ks_unique:
                rel_loss = [v.get(k_source) for v in vs if v.get(k_source) is not None]
                # check that specified relationship is consistent
                assert len(list(set([rl[0] for rl in rel_loss]))) == 1, (
                    "Inconsistent target_source correspondence: one source has multiple target "
                    + "with different source/target relationships"
                )
                if k_source >= len(source_cfgs):
                    logger.warning(
                        f"target_source_correspondence contains non-existent source {k_source}."
                    )
                    continue
                if k_target >= len(target_cfgs):
                    logger.warning(
                        f"target_source_correspondence contains non-existent source {k_target}."
                    )
                    continue
                # add valid entry, source-target pair can have multiple losses
                losses = [rl[1] for rl in rel_loss]
                # add, making sure that each source has only one target (subset relationships
                # but also physical loss )
                assert corr_dict.get(k_source) is None, "source cfg needs unique target"
                corr_dict[k_source] = (k_target, (rel_loss[0][0], losses))

        # TODO: check validity of target_source_correspondence with target and source cfgs

        return corr_dict

    # ── Per-variable-group spatial mask helpers ──────────────────────────────

    def _generate_group_masks_stream(
        self, stream_info: dict, num_cells: int
    ) -> dict[str, np.typing.NDArray] | None:
        """Mode A: generate per-group spatial masks from variable_groups.masking in stream config.

        Returns a dict {group_name: (num_cells,) bool tensor} when at least one group has a
        ``masking`` or ``masking_rate`` config key, otherwise returns None.

        Supports two sub-variants:
        - Full config: ``variable_groups.<group>.masking = {strategy: ..., config: {...}}``
        - Rate-only:   ``variable_groups.<group>.masking_rate: 0.1``  (uses "random" strategy)
        """
        vgroups = stream_info.get("variable_groups", {})
        if not vgroups:
            return None
        has_masking = any("masking" in gcfg or "masking_rate" in gcfg for gcfg in vgroups.values())
        if not has_masking:
            return None

        group_masks: dict[str, np.typing.NDArray] = {}
        for gname, gcfg in vgroups.items():
            mcfg = gcfg.get("masking", None)
            masking_rate = gcfg.get("masking_rate", None)
            if mcfg is not None:
                strategy = mcfg.get("strategy", "random")
                config = dict(mcfg.get("config", {}))
            elif masking_rate is not None:
                strategy = "random"
                config = {"rate": float(masking_rate)}
            else:
                continue
            mask, _ = self._generate_cell_mask(num_cells, strategy, config)
            group_masks[gname] = mask

        return group_masks if group_masks else None

    def _generate_group_masks_model_input(
        self, source_cfgs: dict, num_cells: int
    ) -> dict[str, np.typing.NDArray] | None:
        """Mode B: generate per-group spatial masks from variable_groups tags in model_input.

        Each model_input entry may carry a ``variable_groups`` list.  The entry's
        masking_strategy is used to generate a spatial mask that is then assigned to every
        group named in that list.  If multiple entries claim the same group, the last one wins.

        Returns None when no entry has a ``variable_groups`` tag.
        """
        has_tags = any(cfg.get("variable_groups") for cfg in source_cfgs.values())
        if not has_tags:
            return None

        group_masks: dict[str, np.typing.NDArray] = {}
        for _, cfg in source_cfgs.items():
            groups = cfg.get("variable_groups", None)
            if not groups:
                continue
            mask, _ = self._generate_cell_mask(
                num_cells,
                cfg.get("masking_strategy"),
                cfg.get("masking_strategy_config", {}),
            )
            for g in list(groups):
                group_masks[g] = mask

        return group_masks if group_masks else None

    def build_samples_for_stream(
        self,
        training_mode: str,
        num_cells: int,
        stream_info: dict,
        num_channels: int | None = None,
    ) -> tuple[np.typing.NDArray, list[np.typing.NDArray], list[SampleMetaData]]:
        """Construct encoder/decoder keep-masks for one stream.

        Three operating modes, determined from the active config entries:

        Mode A — MAE Reconstruction
            Trigger: no active ``target_input`` entries AND no forecast source.
            Encoder sees a masked fraction of tokens at timestep *t*;
            decoder reconstructs the complement (tokens encoder did not see).

        Mode B — Forecasting fine-tuning
            Trigger: no active ``target_input`` entries AND at least one
            ``"forecast"`` or ``"causal"`` source present.
            Encoder sees all tokens at timestep *t* (all-True mask);
            decoder predicts all tokens at *t + offset* (all-True mask).
            Non-forecast sources (e.g., a leftover ``mae_swath`` entry from a
            config-merge failure) are silently excluded — only forecast/causal
            sources survive into ``source_cfgs`` / ``target_cfgs``.

        Mode C — Explicit target (e.g., IASI cross-stream fine-tuning)
            Trigger: at least one active ``target_input`` entry.
            Both ``model_input`` and ``target_input`` are used as-is after
            filtering.  Stream flags still apply:
            • ``forcing: True``    → target mask forced to all-False
            • ``diagnostic: True`` → source mask forced to all-False

        An entry is **active** when ``enabled`` is absent or not False AND
        ``num_samples`` is absent or > 0.  ``_filter_active_configs`` enforces
        this before any mode logic runs, so ``enabled: False`` in YAML is
        finally honoured in code.
        """

        stream_masking_cfg = self._effective_masking_cfgs[stream_info["name"]]

        # ── Phase 1: Load raw configs ─────────────────────────────────────────
        raw_source_cfgs = stream_masking_cfg.get("model_input", {})
        raw_target_cfgs = stream_masking_cfg.get("target_input", {})

        # ── Phase 2: Filter to active-only entries ────────────────────────────
        # Removes entries with enabled:False or num_samples:0 before any
        # mode logic runs, so those YAML conventions are enforced in code.
        active_source_cfgs = _filter_active_configs(raw_source_cfgs)
        active_target_cfgs = _filter_active_configs(raw_target_cfgs)

        # ── Phase 3: Determine mode ───────────────────────────────────────────
        target_auto_generated = len(active_target_cfgs) == 0
        has_forecast_source = any(
            _is_forecast_like(cfg.get("masking_strategy", ""))
            for cfg in active_source_cfgs.values()
        )
        is_mode_a = target_auto_generated and not has_forecast_source  # MAE reconstruction
        is_mode_b = target_auto_generated and has_forecast_source  # Forecasting

        # ── Phase 4: Build source_cfgs and target_cfgs ────────────────────────
        if is_mode_b:
            # Keep only forecast/causal sources.  This acts as a safety net for
            # config-merge failures where a pretraining source (e.g. mae_swath)
            # retains num_samples=1 despite the intended override to 0 — those
            # sources are excluded by the comprehension rather than mutated.
            source_cfgs = {
                k: v
                for k, v in active_source_cfgs.items()
                if _is_forecast_like(v.get("masking_strategy", ""))
            }
            target_cfgs = copy.deepcopy(source_cfgs)
        else:
            # Modes A and C start from the full active sets; Mode A will also
            # run mixed-strategy resolution in Phase 5.
            source_cfgs = active_source_cfgs
            target_cfgs = active_target_cfgs

        # ── Phase 5: Resolve "mixed" strategy (Mode A only) ──────────────────
        # Resolve once per sample so source and target share the same
        # sub-strategy draw (same spatial structure, no double-draw for "mixed").
        if is_mode_a:
            source_cfgs = self._resolve_mixed_strategies(copy.deepcopy(source_cfgs))
            target_cfgs = copy.deepcopy(source_cfgs)

        # ── Phase 6: Build source→target correspondence mapping ───────────────
        losses = stream_masking_cfg.losses
        corr_dict = self.parse_src_target_correspondence(losses, target_cfgs, source_cfgs)

        # randomly_drop_as_source_rate from consolidated masking config (training only)
        randomly_drop_rate = (
            stream_masking_cfg.get("randomly_drop_as_source_rate", 0.0)
            if self.stage == "train"
            else 0.0
        )

        # ── Channel dropout (training only, source-side only) ─────────────────
        # Drop individual channels with a very low probability, independently of
        # spatial masking. Targets are never channel-dropped so the loss is always
        # computed against full ground-truth channel values.
        channel_drop_rate = (
            stream_info.get("channel_drop_rate", 0.0) if self.stage == "train" else 0.0
        )
        source_channel_drop_mask = None
        if channel_drop_rate > 0.0 and num_channels is not None and num_channels > 0:
            # True = keep channel, False = drop (zero out) channel
            source_channel_drop_mask = self.rng.random(num_channels) >= channel_drop_rate

        # ── Phase 5.5: Per-group spatial masks ────────────────────────────────
        # Mode A (stream config) takes priority over Mode B (model_input tags).
        # Both produce {group_name: (num_cells,) bool tensor}; the stream-level mask
        # then becomes the union so the encoder sees any cell covered by any group.
        stream_group_masks = self._generate_group_masks_stream(stream_info, num_cells)
        if stream_group_masks is not None:
            # Guard: warn when Mode A (stream config masking) and Mode B (model_input tags) are
            # both configured — Mode A wins silently, so alert the author.
            has_mode_b_tags = any(cfg.get("variable_groups") for cfg in source_cfgs.values())
            if has_mode_b_tags:
                logger.warning(
                    "Stream '%s': Mode A per-group masking (variable_groups.masking in stream "
                    "config) takes priority — variable_groups tags in model_input (Mode B) are "
                    "ignored. Remove one of the two configurations to silence this warning.",
                    stream_info.get("name", "?"),
                )
        elif is_mode_a:
            stream_group_masks = self._generate_group_masks_model_input(source_cfgs, num_cells)
            # Guard: validate that Mode B group names match stream config variable_groups keys.
            # A mismatch produces a silent no-op in the 2-D channel mask — catch it early.
            if stream_group_masks is not None:
                stream_vgroups = stream_info.get("variable_groups", {})
                if stream_vgroups:
                    unknown = [g for g in stream_group_masks if g not in stream_vgroups]
                    if unknown:
                        raise ValueError(
                            f"model_input variable_groups tags {unknown!r} do not match any "
                            f"variable_groups entry in the stream config for stream "
                            f"'{stream_info.get('name', '?')}'. "
                            f"Defined groups: {sorted(stream_vgroups)}. "
                            "Group names must be spelled identically in both configs."
                        )

        # ── Phase 7: Generate target masks ────────────────────────────────────
        target_masks = MaskData()
        i_target = 0
        for i_cfg, (_, target_cfg) in enumerate(target_cfgs.items()):
            for _ in range(target_cfg.get("num_samples", 1)):
                # forcing stream: decoder never predicts this stream → all-False
                if is_stream_forcing(stream_info, self.stage):
                    target_mask, mask_params = torch.zeros(num_cells, dtype=torch.bool), {}
                else:
                    masking_config = target_cfg.get("masking_strategy_config", {})
                    # targets are never randomly dropped
                    target_mask, mask_params = self._get_mask(
                        num_cells=num_cells,
                        strategy=target_cfg.get("masking_strategy"),
                        masking_strategy_config=masking_config,
                        target_relationship_mask=("independent", None),
                    )

                # get all losses and flatten
                losses = [v[1][1] for _, v in corr_dict.items() if len(v) > 0 and v[0] == i_cfg]
                losses = [ll for lt in losses for ll in lt]
                # corresponding sources
                corr = [k for k, v in corr_dict.items() if len(v) > 0 and v[0] == i_cfg]
                # skip items that do not appear in loss
                if len(corr) == 0:
                    continue
                target_masks.add_mask(
                    target_mask, mask_params, target_cfg, losses, i_target, corr, None
                )
                i_target += 1

        # ── Phase 8: Generate source masks + complement override (Mode A) ─────
        source_masks = MaskData()
        source_target_mapping = []
        target_num_samples = get_num_samples(target_cfgs)
        is_stream_dropped = randomly_drop_rate > 0.0 and self.rng.uniform() < randomly_drop_rate
        i_source = 0
        for i_src_cfg, (_, source_cfg) in enumerate(source_cfgs.items()):
            # skip items that do not appear in loss
            if i_src_cfg not in corr_dict:
                continue
            for i_sample in range(source_cfg.get("num_samples", 1)):
                masking_config = source_cfg.get("masking_strategy_config", {})
                # extract corresponding target
                target_cfg_idx, rel_losses = corr_dict[i_src_cfg]
                relationship, losses = rel_losses
                # ensure proper default relationships
                if relationship is None:
                    if is_mode_a:
                        # Source is generated independently; target is overridden to
                        # complement below, so source and target never overlap or gap.
                        relationship = "independent"
                    elif source_cfg.get("masking_strategy") == "random":
                        # default for masked-token modeling with explicit target
                        relationship = "complement"
                    else:
                        relationship = "independent"
                target_idx = target_num_samples[:target_cfg_idx].sum()
                # iterate sequentially through targets (1-to-1 when target auto-generated)
                target_idx += i_sample % target_num_samples[target_cfg_idx].item()

                # diagnostic stream or randomly dropped: encoder ignores this stream → all-False
                is_skipped = is_stream_diagnostic(stream_info, self.stage) or is_stream_dropped
                if is_skipped:
                    source_mask, mask_params = torch.zeros(num_cells, dtype=torch.bool), {}
                    source_sample_group_masks = None
                elif stream_group_masks is not None:
                    # Per-group masking: encoder sees the union of all group masks.
                    # The per-group masks are stored separately so the tokenizer can
                    # apply different spatial visibility to each variable group.
                    source_sample_group_masks = stream_group_masks
                    source_mask = torch.stack(list(stream_group_masks.values())).any(dim=0)
                    mask_params = {"group_masking": True}
                else:
                    source_sample_group_masks = None
                    source_mask, mask_params = self._get_mask(
                        num_cells=num_cells,
                        strategy=source_cfg.get("masking_strategy"),
                        masking_strategy_config=masking_config,
                        target_relationship_mask=(relationship, target_masks.get_mask(target_idx)),
                    )

                # Mode A complement override: decoder reconstructs exactly what the
                # encoder did not see.  Skipped for diagnostic/dropped streams (source
                # mask is all-False, complement would be all-True which is wrong),
                # for forcing streams (target must stay all-False from Phase 7; the
                # model builds no decoder for them, so any target here is dead weight
                # the dataloader would tokenize and ship every sample) and
                # for Modes B/C (different timesteps or explicit target configured).
                if is_mode_a and not is_skipped and not is_stream_forcing(stream_info, self.stage):
                    if source_sample_group_masks is not None:
                        # Per-group complement: each group's target is the cells NOT seen
                        # by the encoder for that group.  Target union is the union of
                        # all per-group complements (a superset of the source complement).
                        target_group_masks = {g: ~m for g, m in source_sample_group_masks.items()}
                        union_complement = torch.stack(list(target_group_masks.values())).any(dim=0)
                        target_masks.masks[target_idx] = union_complement
                        target_masks.metadata[target_idx].mask = union_complement
                        target_masks.group_spatial_masks[target_idx] = target_group_masks
                    else:
                        complement = ~source_mask
                        target_masks.masks[target_idx] = complement
                        target_masks.metadata[target_idx].mask = complement

                corr = target_idx
                source_masks.add_mask(
                    source_mask,
                    mask_params,
                    source_cfg,
                    losses,
                    i_source,
                    corr,
                    relationship,
                    channel_drop_mask=source_channel_drop_mask,
                    group_spatial_masks=source_sample_group_masks,
                )

                source_target_mapping += [target_idx]
                i_source += 1

        source_target_mapping = np.array(source_target_mapping, dtype=np.int32)

        return (target_masks, source_masks, source_target_mapping)

    def _get_mask(
        self,
        num_cells: int,
        strategy: str,
        masking_strategy_config: dict,
        target_relationship_mask: (str, np.typing.NDArray),
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

        Returns
        -------
        np.ndarray
            Boolean array of shape [num_cells] where True indicates the cell is kept.
        dict
            Parameters describing the masking that was applied
        """

        relationship, target_mask = target_relationship_mask

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
        elif relationship == "identity":
            assert target_mask is not None, (
                "relationship: {relationship} incompatible with target_mask None"
            )
            mask = target_mask
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

        return (mask, params)

    def _generate_cell_mask(
        self,
        num_cells: int,
        strategy: str,
        masking_strategy_config: dict,
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

        assert num_cells == self.healpix_num_cells, (
            "num_cells inconsistent with configured healpix level."
        )

        # generate cell mask

        if strategy == "random":
            keep_rate = self._get_sampling_rate(masking_strategy_config)
            mask = self.rng.uniform(0, 1, num_cells) < keep_rate

        elif "forecast" in strategy or strategy == "causal":
            mask = np.ones(num_cells, dtype=bool)

            if "diffusion_rn" in masking_strategy_config:
                masking_params["noise_level_rn"] = self.rng.normal(0.0, 1.0)

        elif strategy == "healpix":
            # prepare healpix-based masking
            keep_rate = self._get_sampling_rate(masking_strategy_config)
            hl_mask, num_parent_cells, num_children_per_parent, num_parents_to_keep = (
                self._prepare_healpix_based_masking(masking_strategy_config, keep_rate)
            )

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

        # Spatial healpix based cropping, select contiguous region
        elif strategy == "cropping_healpix":
            # prepare healpix-based masking
            keep_rate = self._get_sampling_rate(masking_strategy_config)
            hl_mask, num_parent_cells, num_children_per_parent, num_parents_to_keep = (
                self._prepare_healpix_based_masking(masking_strategy_config, keep_rate)
            )

            if num_parents_to_keep == 0:
                mask = np.zeros(num_cells, dtype=bool)
            else:
                # Spatial selection method
                method = masking_strategy_config.get("method", "geodesic_disk")

                # Use standard spatial selection
                mask = self._select_spatially_contiguous_cells(
                    healpix_level=hl_mask,
                    num_cells=num_cells,
                    num_cells_to_select=num_parents_to_keep,
                    num_children_per_parent=num_children_per_parent,
                    center_cell=None,
                    method=method,
                )

        elif strategy == "satellite_swath":
            num_swaths = int(masking_strategy_config.get("num_swaths", 5))
            swath_half_width = float(masking_strategy_config.get("swath_width_deg", 20.0)) / 2.0
            # Longitude drift from south pole to north pole during one ascending half-orbit.
            # A typical LEO polar orbit drifts ~-25° westward across one pole-to-pole pass
            # because the Earth rotates ~12.5° during the ~50-minute half-orbit.
            orbit_drift_deg = float(masking_strategy_config.get("orbit_drift_deg", -25.0))

            cell_lats, cell_lons = self._get_cell_coords()

            mask = np.zeros(num_cells, dtype=bool)

            # Each swath starts at a random longitude at the south pole and drifts
            # orbit_drift_deg westward by the time it reaches the north pole.
            start_lons = self.rng.uniform(-180.0, 180.0, num_swaths)
            for start_lon in start_lons:
                # Centerline longitude at each cell's latitude:
                # linear interpolation from start_lon (lat=-90) to start_lon+orbit_drift (lat=90)
                swath_lon = start_lon + orbit_drift_deg * (cell_lats + 90.0) / 180.0

                # Longitude difference wrapped to [-180, 180]
                dlon = cell_lons - swath_lon
                dlon = (dlon + 180.0) % 360.0 - 180.0

                # Distance in longitude-space (NOT scaled by cos(lat)).
                # Scaling by cos(lat) would make dist→0 near the poles, causing every cell
                # near the poles to fall inside every swath — destroying the stripe pattern.
                # Constant lon-space width produces clear diagonal bands in lat-lon scatter plots.
                mask |= np.abs(dlon) < swath_half_width

            masking_params["num_swaths"] = num_swaths
            masking_params["swath_start_lons"] = start_lons.tolist()

        elif strategy == "satellite_swath_sparse":
            # Like satellite_swath but with more, thinner swaths and random subsampling
            # within the swath footprint. Combines orbital spatial structure with random
            # dropout inside each swath — the model sees a sparse random pattern that is
            # geographically structured rather than uniformly distributed.
            num_swaths = int(masking_strategy_config.get("num_swaths", 12))
            swath_half_width = float(masking_strategy_config.get("swath_width_deg", 10.0)) / 2.0
            orbit_drift_deg = float(masking_strategy_config.get("orbit_drift_deg", -25.0))
            keep_rate = self._get_sampling_rate(masking_strategy_config)

            cell_lats, cell_lons = self._get_cell_coords()

            # Pass 1: determine which cells fall within any swath footprint.
            # Use raw longitude difference (no cos(lat) scaling) so swaths appear as
            # clear diagonal bands at all latitudes in the scatter-plot visualization.
            in_swath = np.zeros(num_cells, dtype=bool)
            start_lons = self.rng.uniform(-180.0, 180.0, num_swaths)
            for start_lon in start_lons:
                swath_lon = start_lon + orbit_drift_deg * (cell_lats + 90.0) / 180.0
                dlon = cell_lons - swath_lon
                dlon = (dlon + 180.0) % 360.0 - 180.0
                in_swath |= np.abs(dlon) < swath_half_width

            # Pass 2: random subsampling within the swath footprint at keep_rate
            cell_draw = self.rng.uniform(0.0, 1.0, num_cells)
            mask = in_swath & (cell_draw < keep_rate)

            masking_params["num_swaths"] = num_swaths
            masking_params["swath_start_lons"] = start_lons.tolist()

        elif strategy == "mixed":
            strategies = list(masking_strategy_config.get("strategies", ["random", "healpix"]))
            raw_weights = masking_strategy_config.get("strategy_weights")
            if raw_weights is not None:
                w = np.array([raw_weights.get(s, 1.0) for s in strategies], dtype=float)
                w /= w.sum()
            else:
                w = None

            chosen_idx = self.rng.choice(len(strategies), p=w)
            chosen = strategies[chosen_idx]
            masking_params["chosen_strategy"] = chosen

            # Build sub-config: start from mixed config, then apply per-strategy overrides
            strategy_cfgs = masking_strategy_config.get("strategy_configs", {})
            sub_cfg = {**masking_strategy_config, **strategy_cfgs.get(chosen, {})}

            # For healpix-family strategies: sample hl_mask level from the configured list
            if "hl_mask_levels" in masking_strategy_config and "hl_mask" not in strategy_cfgs.get(
                chosen, {}
            ):
                hl_mask_levels = list(masking_strategy_config["hl_mask_levels"])
                hl_mask = hl_mask_levels[self.rng.integers(0, len(hl_mask_levels))]
                sub_cfg["hl_mask"] = hl_mask
                masking_params["hl_mask"] = hl_mask

            # Recursive dispatch — returns a bool tensor; convert back to numpy for the
            # to_bool_tensor() call at the bottom of this method
            sub_mask, sub_params = self._generate_cell_mask(num_cells, chosen, sub_cfg)
            masking_params.update(sub_params)
            mask = sub_mask.numpy()  # sub_mask is CPU bool tensor from recursive call

        else:
            raise NotImplementedError(
                f"Cell selection strategy '{strategy}' not supported for keep mask generation."
            )

        mask = to_bool_tensor(mask)

        return (mask, masking_params)

    def _select_spatially_contiguous_cells(
        self,
        healpix_level: int,
        num_cells: int,
        num_cells_to_select: int,
        num_children_per_parent: int,
        center_cell: int | None = None,
        method: str = "geodesic_disk",
    ) -> NDArray:
        """
        Select spatially contiguous cells on the sphere using neighbor relationships.

        This is the core spatial selection helper used for both masking and cropping.

        Args:
            healpix_level: HEALPix level for selection
            num_cells: Total number of cells at data level
            num_cells_to_select: Number of cells to select
            num_children_per_parent: Number of child cells per parent cell
            center_cell: Starting cell (None = random)
            method: Selection method:
                - "disk": Layer-by-layer neighbor growth (compact regions)
                - "random_walk": Random neighbor selection (irregular shapes)
                - "geodesic_disk": Angular distance selection (circular regions)

        Returns:
            Array of selected cell indices forming a spatially contiguous region

        Examples:
            # Independent crop
            crop1 = _select_spatially_contiguous_cells(0, 9, method="geodesic_disk")
        """

        num_total_cells = 12 * (4**healpix_level)
        nside = 2**healpix_level

        assert num_cells_to_select <= num_total_cells

        # Random starting point. Note we may want overlap here
        # for now we basically control with chosen masking rates
        center_cell = self.rng.integers(0, num_total_cells)

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

        parent_ids = np.array(sorted(selected))

        # Project to data level
        child_offsets = np.arange(num_children_per_parent)
        child_indices = (parent_ids[:, None] * num_children_per_parent + child_offsets).reshape(-1)

        # Create keep mask (True = selected/kept cells at data level).
        mask = np.zeros(num_cells, dtype=bool)
        mask[child_indices] = True

        return mask

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
        Angular distance selection, creates most uniform somewhat circular regions.

        Uses _get_xyz_at_hl_level() so HEALPix coordinate lookups are cached per
        worker process and never repeated across samples.
        """
        hl = int(round(np.log2(nside)))
        all_xyz = self._get_xyz_at_hl_level(hl)  # (num_total_cells, 3) — cached

        center_xyz = all_xyz[center_cell]

        # Compute angular distances and select closest cells
        dot_products = np.clip(np.dot(all_xyz, center_xyz), -1.0, 1.0)
        angular_distances = np.arccos(dot_products)
        selected = np.argsort(angular_distances)[:num_cells_to_select]

        return selected

    def _prepare_healpix_based_masking(self, cfg, keep_rate):
        """
        Prepare healpix masking related attributes.
        """

        hl_data = self.healpix_level_data
        hl_mask = cfg.get("hl_mask")
        assert hl_mask is not None and hl_mask <= hl_data, (
            "For healpix keep mask generation, cfg['hl_mask'] must be set and <= data level."
        )
        num_parent_cells = 12 * (4**hl_mask)
        level_diff = hl_data - hl_mask
        num_children_per_parent = 4**level_diff
        # number of parents to keep
        num_parents_to_keep = int(np.round(keep_rate * num_parent_cells))

        return hl_mask, num_parent_cells, num_children_per_parent, num_parents_to_keep
