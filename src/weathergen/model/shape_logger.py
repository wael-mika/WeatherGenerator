# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

"""
Shape Logger for WeatherGenerator Model

This module provides detailed logging of tensor shapes and dimensions throughout
the model's forward pass to help understand data flow and debug shape mismatches.
"""

import logging
from typing import Any, Optional

import torch

logger = logging.getLogger(__name__)


class ShapeLogger:
    """
    Logs tensor shapes and dimensions throughout the model forward pass.

    Helps understand:
    - How data flows through the model
    - What each dimension represents (batch, cells, queries, features)
    - Where shape transformations occur
    - Potential shape mismatches or issues
    """

    def __init__(self, enabled: bool = False, step_interval: int = 1):
        """
        Args:
            enabled: Whether to enable shape logging
            step_interval: Log shapes every N training steps (default: 1 = every step)
        """
        self.enabled = enabled
        self.step_interval = step_interval
        self.current_step = 0
        self.header_printed = False

    def set_enabled(self, enabled: bool):
        """Enable or disable shape logging"""
        self.enabled = enabled
        if enabled:
            self.header_printed = False

    def increment_step(self):
        """Increment the current step counter"""
        self.current_step += 1

    def should_log(self) -> bool:
        """Determine if we should log on this step"""
        return self.enabled and (self.current_step % self.step_interval == 0)

    def _print_header(self, phase: str):
        """Print section header"""
        if not self.should_log():
            return
        logger.info("=" * 100)
        logger.info(f"SHAPE LOGGING - Step {self.current_step} - {phase}")
        logger.info("=" * 100)

    def _format_shape(self, tensor: torch.Tensor, name: str, description: str = "") -> str:
        """
        Format tensor shape information

        Args:
            tensor: The tensor to describe
            name: Name of the tensor
            description: Human-readable description of what dimensions represent

        Returns:
            Formatted string with shape information
        """
        if tensor is None:
            return f"{name:40s} : None"

        shape_str = " × ".join([f"{s:>6d}" for s in tensor.shape])
        numel = tensor.numel()
        dtype_str = str(tensor.dtype).replace("torch.", "")
        device_str = str(tensor.device)

        # Add explicit shape tuple for clarity
        shape_tuple = str(tuple(tensor.shape))

        info = f"{name:40s} : [{shape_str}] = {numel:>12,} elements ({dtype_str}, {device_str})"
        info += f"\n{' ':40s}   Exact shape: {shape_tuple} (ndim={tensor.ndim})"

        if description:
            info += f"\n{' ':40s}   {description}"

        return info

    def log_embedding_input(self, streams_data, batch_size: int, num_streams: int):
        """Log shapes at embedding input"""
        if not self.should_log():
            return

        self._print_header("EMBEDDING ENGINE - INPUT")
        logger.info(f"Batch size: {batch_size}")
        logger.info(f"Number of streams: {num_streams}")
        logger.info("")

        for i_batch in range(len(streams_data)):
            logger.info(f"Batch {i_batch}:")
            for i_stream in range(len(streams_data[i_batch])):
                stream_data = streams_data[i_batch][i_stream]
                logger.info(f"  Stream {i_stream}:")

                if hasattr(stream_data, 'sources') and stream_data.sources is not None:
                    for j, source in enumerate(stream_data.sources):
                        if source is not None:
                            logger.info(self._format_shape(
                                source,
                                f"    sources[{j}]",
                                "Source data tokens"
                            ))

                if hasattr(stream_data, 'target_coords'):
                    for fstep, coords in enumerate(stream_data.target_coords):
                        if coords is not None and coords.numel() > 0:
                            logger.info(self._format_shape(
                                coords,
                                f"    target_coords[fstep={fstep}]",
                                "Target coordinates for prediction"
                            ))
        logger.info("")

    def log_embedding_output(self, tokens: torch.Tensor, num_healpix_cells: int, pe_embed_shape: tuple):
        """Log shapes after embedding"""
        if not self.should_log():
            return

        logger.info("-" * 100)
        logger.info("EMBEDDING ENGINE - OUTPUT")
        logger.info("-" * 100)
        logger.info(self._format_shape(
            tokens,
            "tokens_all (after embedding)",
            "All embedded tokens from all streams, arranged cell-wise"
        ))
        logger.info(f"Positional embedding shape: {pe_embed_shape}")
        logger.info(f"Number of HEALPix cells: {num_healpix_cells}")
        logger.info("")

    def log_local_assimilation_input(
        self,
        tokens: torch.Tensor,
        cell_lens: torch.Tensor,
        q_cells_shape: tuple,
        pe_global_shape: tuple,
        batch_size: int,
        num_healpix_cells: int
    ):
        """Log shapes at local assimilation input"""
        if not self.should_log():
            return

        self._print_header("LOCAL ASSIMILATION ENGINE - INPUT")
        logger.info(self._format_shape(
            tokens,
            "tokens (input)",
            "Embedded tokens to be processed locally"
        ))
        logger.info(self._format_shape(
            cell_lens,
            "cell_lens",
            "Cumulative lengths for each cell (for varlen attention)"
        ))
        logger.info(f"q_cells shape: {q_cells_shape}")
        logger.info(f"  → Learnable queries: (num_cells or 1, num_queries_per_cell, dim_embed)")
        logger.info(f"pe_global shape: {pe_global_shape}")
        logger.info(f"  → Global positional encoding: (num_cells, num_queries, dim_embed)")
        logger.info(f"Batch size: {batch_size}")
        logger.info(f"Number of HEALPix cells: {num_healpix_cells}")
        logger.info("")

    def log_local_assimilation_blocks(
        self,
        block_idx: int,
        tokens_before: torch.Tensor,
        tokens_after: torch.Tensor,
        block_name: str
    ):
        """Log shapes through local assimilation blocks"""
        if not self.should_log():
            return

        logger.info(f"Local Assimilation Block {block_idx} ({block_name}):")
        logger.info(self._format_shape(
            tokens_before,
            f"  tokens (before block {block_idx})",
            ""
        ))
        logger.info(self._format_shape(
            tokens_after,
            f"  tokens (after block {block_idx})",
            ""
        ))
        logger.info("")

    def log_local_to_global_adapter(
        self,
        tokens_local: torch.Tensor,
        tokens_global_before: torch.Tensor,
        tokens_global_after: torch.Tensor,
        q_cells_lens: torch.Tensor,
        cell_lens: torch.Tensor
    ):
        """Log shapes at local-to-global adapter"""
        if not self.should_log():
            return

        logger.info("-" * 100)
        logger.info("LOCAL-TO-GLOBAL ADAPTER")
        logger.info("-" * 100)
        logger.info(self._format_shape(
            tokens_local,
            "tokens_local (input)",
            "Locally assimilated tokens"
        ))
        logger.info(self._format_shape(
            tokens_global_before,
            "tokens_global (before adapter)",
            "Global queries (initialized from q_cells + pe_global)"
        ))
        logger.info(self._format_shape(
            tokens_global_after,
            "tokens_global (after adapter)",
            "Global tokens after cross-attention with local tokens"
        ))
        logger.info(self._format_shape(
            q_cells_lens,
            "q_cells_lens",
            "Cumulative query lengths per cell"
        ))
        logger.info(self._format_shape(
            cell_lens,
            "cell_lens",
            "Cumulative token lengths per cell"
        ))
        logger.info("")

    def log_local_assimilation_output(
        self,
        tokens_global: torch.Tensor,
        batch_size: int,
        num_healpix_cells: int,
        num_queries: int,
        dim_embed: int
    ):
        """Log shapes after local assimilation (ready for global)"""
        if not self.should_log():
            return

        logger.info("-" * 100)
        logger.info("LOCAL ASSIMILATION ENGINE - OUTPUT (Ready for Global Assimilation)")
        logger.info("-" * 100)
        logger.info(self._format_shape(
            tokens_global,
            "tokens_global (final)",
            f"Expected: [batch={batch_size}, cells*queries={num_healpix_cells}×{num_queries}={num_healpix_cells*num_queries}, dim={dim_embed}]"
        ))
        if tokens_global.ndim == 3:
            logger.info(f"  → Batch dimension PRESERVED: [{batch_size}, {num_healpix_cells * num_queries}, {dim_embed}]")
        elif tokens_global.ndim == 2:
            logger.info(f"  → Batch dimension FLATTENED: [{batch_size * num_healpix_cells * num_queries}, {dim_embed}]")
        logger.info("")

    def log_global_assimilation_input(self, tokens: torch.Tensor):
        """Log shapes at global assimilation input"""
        if not self.should_log():
            return

        self._print_header("GLOBAL ASSIMILATION ENGINE - INPUT")
        logger.info(self._format_shape(
            tokens,
            "tokens (input)",
            "Tokens from local assimilation, ready for global processing"
        ))
        logger.info("")

    def log_global_assimilation_block(
        self,
        block_idx: int,
        block_type: str,
        tokens_before: torch.Tensor,
        tokens_after: torch.Tensor,
        aux_loss: Optional[float] = None,
        is_moe: bool = False
    ):
        """Log shapes through global assimilation blocks"""
        if not self.should_log():
            return

        block_label = f"Block {block_idx} ({block_type})"
        if is_moe:
            block_label += " [MoE]"

        logger.info(f"Global Assimilation {block_label}:")
        logger.info(self._format_shape(
            tokens_before,
            f"  tokens (before)",
            ""
        ))
        logger.info(self._format_shape(
            tokens_after,
            f"  tokens (after)",
            ""
        ))

        if is_moe and aux_loss is not None:
            logger.info(f"  MoE auxiliary loss: {aux_loss:.6f}")

        logger.info("")

    def log_global_assimilation_output(self, tokens: torch.Tensor):
        """Log shapes after global assimilation"""
        if not self.should_log():
            return

        logger.info("-" * 100)
        logger.info("GLOBAL ASSIMILATION ENGINE - OUTPUT")
        logger.info("-" * 100)
        logger.info(self._format_shape(
            tokens,
            "tokens (output)",
            "Globally assimilated latent representation"
        ))
        logger.info("")

    def log_forecast_input(self, tokens: torch.Tensor, fstep: int):
        """Log shapes at forecasting engine input"""
        if not self.should_log():
            return

        self._print_header(f"FORECASTING ENGINE - INPUT (fstep={fstep})")
        logger.info(self._format_shape(
            tokens,
            "tokens (input)",
            "Latent representation to be advanced in time"
        ))
        logger.info(f"Forecast step: {fstep}")
        logger.info("")

    def log_forecast_block(
        self,
        block_idx: int,
        block_type: str,
        tokens_before: torch.Tensor,
        tokens_after: torch.Tensor,
        fstep: int,
        aux_loss: Optional[float] = None,
        is_moe: bool = False
    ):
        """Log shapes through forecasting blocks"""
        if not self.should_log():
            return

        block_label = f"Block {block_idx} ({block_type})"
        if is_moe:
            block_label += " [MoE]"

        logger.info(f"Forecasting {block_label} (fstep={fstep}):")
        logger.info(self._format_shape(
            tokens_before,
            f"  tokens (before)",
            ""
        ))
        logger.info(self._format_shape(
            tokens_after,
            f"  tokens (after)",
            ""
        ))

        if is_moe and aux_loss is not None:
            logger.info(f"  MoE auxiliary loss: {aux_loss:.6f}")

        logger.info("")

    def log_forecast_output(self, tokens: torch.Tensor, fstep: int):
        """Log shapes after forecasting"""
        if not self.should_log():
            return

        logger.info("-" * 100)
        logger.info(f"FORECASTING ENGINE - OUTPUT (fstep={fstep})")
        logger.info("-" * 100)
        logger.info(self._format_shape(
            tokens,
            "tokens (output)",
            "Latent representation advanced by one time step"
        ))
        logger.info("")

    def log_prediction_input(
        self,
        tokens: torch.Tensor,
        tokens_stream: torch.Tensor,
        fstep: int,
        batch_size: int,
        num_healpix_cells: int,
        num_queries: int
    ):
        """Log shapes at prediction input"""
        if not self.should_log():
            return

        self._print_header(f"PREDICTION ENGINE - INPUT (fstep={fstep})")
        logger.info(self._format_shape(
            tokens,
            "tokens (from global/forecast)",
            f"Latent representation: [batch × cells × queries, dim]"
        ))
        logger.info(self._format_shape(
            tokens_stream,
            "tokens_stream (with neighbors)",
            f"Reshaped + neighborhood structure: [batch × cells × queries × (1+neighbors), dim]"
        ))
        logger.info(f"Batch size: {batch_size}")
        logger.info(f"HEALPix cells: {num_healpix_cells}")
        logger.info(f"Queries per cell: {num_queries}")
        logger.info("")

    def log_prediction_stream(
        self,
        stream_idx: int,
        stream_name: str,
        tc_tokens: torch.Tensor,
        tc_tokens_after_tte: torch.Tensor,
        pred_tokens: torch.Tensor,
        target_coords_shape: tuple,
        tcs_lens: torch.Tensor
    ):
        """Log shapes for each prediction stream"""
        if not self.should_log():
            return

        logger.info(f"Prediction Stream {stream_idx} ({stream_name}):")
        logger.info(f"  Target coordinates shape: {target_coords_shape}")
        logger.info(self._format_shape(
            tc_tokens,
            "  tc_tokens (embedded coords)",
            "Embedded target coordinates"
        ))
        logger.info(self._format_shape(
            tc_tokens_after_tte,
            "  tc_tokens (after tte)",
            "After Target Token Engine (cross-attention with latent)"
        ))
        logger.info(self._format_shape(
            pred_tokens,
            "  pred_tokens (final)",
            "Final predictions after prediction head"
        ))
        logger.info(self._format_shape(
            tcs_lens,
            "  tcs_lens",
            "Target coordinate lengths for varlen attention"
        ))
        logger.info("")

    def log_prediction_output(self, preds_all: list, fstep: int):
        """Log shapes of all predictions"""
        if not self.should_log():
            return

        logger.info("-" * 100)
        logger.info(f"PREDICTION ENGINE - OUTPUT (fstep={fstep})")
        logger.info("-" * 100)
        logger.info(f"Number of prediction streams: {len(preds_all)}")
        for i, pred in enumerate(preds_all):
            logger.info(self._format_shape(
                pred,
                f"  preds_all[{i}]",
                "Predictions for one stream"
            ))
        logger.info("")

    def log_forward_complete(self, preds_all: list, posteriors: Any):
        """Log final output shapes"""
        if not self.should_log():
            return

        logger.info("=" * 100)
        logger.info("FORWARD PASS COMPLETE - FINAL OUTPUT")
        logger.info("=" * 100)
        logger.info(f"Total forecast steps: {len(preds_all)}")
        for fstep, preds in enumerate(preds_all):
            logger.info(f"Forecast step {fstep}:")
            for i, pred in enumerate(preds):
                logger.info(self._format_shape(
                    pred,
                    f"  stream[{i}]",
                    ""
                ))

        if posteriors != 0.0 and hasattr(posteriors, '__len__'):
            logger.info(f"Posteriors: {len(posteriors)} elements")

        logger.info("=" * 100)
        logger.info("")


# Global singleton instance
_shape_logger = None


def get_shape_logger(enabled: bool = False, step_interval: int = 1) -> ShapeLogger:
    """
    Get or create the global ShapeLogger instance

    Args:
        enabled: Whether to enable shape logging
        step_interval: Log shapes every N training steps

    Returns:
        ShapeLogger instance
    """
    global _shape_logger
    if _shape_logger is None:
        _shape_logger = ShapeLogger(enabled=enabled, step_interval=step_interval)
    return _shape_logger
