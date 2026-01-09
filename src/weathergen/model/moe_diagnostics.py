"""
MoE Spatial Coherence Diagnostics

This module provides utilities to diagnose whether MoE routing
respects the spatial structure of HEALPix cells in weather data.
"""

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def analyze_routing_spatial_coherence(
    expert_indices: torch.Tensor,
    neighbor_structure: torch.Tensor = None,
    log_details: bool = True
) -> dict:
    """
    Analyze whether MoE routing preserves spatial coherence.

    Args:
        expert_indices: Expert assignments [batch, num_cells, top_k]
        neighbor_structure: Optional [num_cells, num_neighbors] HEALPix neighbor indices
        log_details: Whether to log detailed statistics

    Returns:
        Dictionary with coherence metrics
    """
    batch_size, num_cells, top_k = expert_indices.shape

    # Get top expert for each cell
    top_experts = expert_indices[:, :, 0]  # [batch, num_cells]

    metrics = {}

    # === Metric 1: Sequential Transition Rate ===
    # How often do consecutive cells use different experts?
    # (Lower is better for spatial coherence)

    transitions = (top_experts[:, 1:] != top_experts[:, :-1]).float()
    transition_rate = transitions.mean().item()
    metrics['sequential_transition_rate'] = transition_rate

    # === Metric 2: Expert Clustering ===
    # Are same experts used in contiguous regions?
    # (Higher is better)

    # Count run lengths of same expert
    run_lengths = []
    for b in range(batch_size):
        current_expert = top_experts[b, 0].item()
        current_length = 1

        for i in range(1, num_cells):
            if top_experts[b, i].item() == current_expert:
                current_length += 1
            else:
                run_lengths.append(current_length)
                current_expert = top_experts[b, i].item()
                current_length = 1

        run_lengths.append(current_length)

    avg_run_length = sum(run_lengths) / len(run_lengths)
    max_run_length = max(run_lengths)
    metrics['avg_run_length'] = avg_run_length
    metrics['max_run_length'] = max_run_length

    # === Metric 3: Neighbor Coherence (if neighbor structure provided) ===
    if neighbor_structure is not None:
        # Check how often a cell and its neighbors use the same expert
        neighbor_agreement_rates = []

        for b in range(batch_size):
            for cell_idx in range(num_cells):
                cell_expert = top_experts[b, cell_idx]
                neighbors = neighbor_structure[cell_idx]

                # Remove invalid neighbors (-1 or self-references)
                valid_neighbors = neighbors[(neighbors >= 0) & (neighbors < num_cells) & (neighbors != cell_idx)]

                if len(valid_neighbors) > 0:
                    neighbor_experts = top_experts[b, valid_neighbors]
                    agreement = (neighbor_experts == cell_expert).float().mean().item()
                    neighbor_agreement_rates.append(agreement)

        metrics['neighbor_agreement_rate'] = sum(neighbor_agreement_rates) / len(neighbor_agreement_rates)

    # === Logging ===
    if log_details:
        logger.info("=" * 80)
        logger.info("MoE Spatial Coherence Analysis")
        logger.info("=" * 80)

        logger.info(f"Sequential Transition Rate: {transition_rate:.2%}")
        logger.info(f"  → Fraction of consecutive cells using different experts")
        logger.info(f"  → Lower is better (0% = perfect coherence, 100% = random)")
        logger.info("")

        logger.info(f"Average Run Length: {avg_run_length:.1f} cells")
        logger.info(f"  → Average contiguous region using same expert")
        logger.info(f"  → Higher is better (indicates regional clustering)")
        logger.info(f"Maximum Run Length: {max_run_length} cells")
        logger.info("")

        if 'neighbor_agreement_rate' in metrics:
            logger.info(f"Neighbor Agreement Rate: {metrics['neighbor_agreement_rate']:.2%}")
            logger.info(f"  → Fraction of neighbors using the same expert")
            logger.info(f"  → Higher is better (100% = perfect spatial coherence)")
            logger.info("")

        # Interpretation
        logger.info("Interpretation:")
        if transition_rate > 0.7:
            logger.warning("  ⚠️  POOR spatial coherence (routing is essentially random)")
            logger.warning("  → Consider implementing spatially-aware routing")
        elif transition_rate > 0.4:
            logger.info("  ⚡ MODERATE spatial coherence (some structure preserved)")
            logger.info("  → May benefit from spatial awareness")
        else:
            logger.info("  ✅ GOOD spatial coherence (strong regional structure)")
            logger.info("  → Current routing respects spatial structure")

        logger.info("=" * 80)

    return metrics


def log_expert_assignments(
    expert_indices: torch.Tensor,
    num_samples: int = 20,
    num_experts: int = None
):
    """
    Log first few expert assignments for manual inspection.

    Args:
        expert_indices: Expert assignments [batch, num_cells, top_k]
        num_samples: Number of cells to show
        num_experts: Total number of experts (for histogram)
    """
    batch_size, num_cells, top_k = expert_indices.shape

    logger.info("-" * 80)
    logger.info(f"Expert Assignments (first {num_samples} cells):")
    logger.info("-" * 80)

    for b in range(min(batch_size, 1)):  # Show first batch only
        assignments = expert_indices[b, :num_samples, 0].cpu().numpy()
        logger.info(f"Batch {b}: {assignments}")

        # Check for patterns
        unique, counts = torch.unique(expert_indices[b, :, 0], return_counts=True)
        logger.info(f"\nExpert distribution (batch {b}):")
        for expert_id, count in zip(unique.cpu().numpy(), counts.cpu().numpy()):
            pct = 100 * count / num_cells
            logger.info(f"  Expert {expert_id}: {count:>5} cells ({pct:>5.1f}%)")

    logger.info("-" * 80)


def add_spatial_coherence_hook(moe_block: nn.Module, neighbor_structure: torch.Tensor = None):
    """
    Add a forward hook to a MoE block to log spatial coherence during training.

    Args:
        moe_block: The MoEBlock module to monitor
        neighbor_structure: Optional HEALPix neighbor structure

    Example:
        from weathergen.model.moe_diagnostics import add_spatial_coherence_hook

        # In model initialization
        for name, module in model.named_modules():
            if isinstance(module, MoEBlock):
                add_spatial_coherence_hook(module, model_params.hp_nbours)
    """

    def hook_fn(module, input, output):
        # Only log periodically (e.g., every 100 steps)
        if not hasattr(module, '_coherence_step_count'):
            module._coherence_step_count = 0

        module._coherence_step_count += 1

        # Log every 100 steps
        if module._coherence_step_count % 100 == 0:
            x = input[0]  # Input tensor

            # Get routing decision
            with torch.no_grad():
                router_probs, expert_indices, expert_weights = module.router(x)

                # Analyze spatial coherence
                analyze_routing_spatial_coherence(
                    expert_indices,
                    neighbor_structure=neighbor_structure,
                    log_details=True
                )

                # Log assignments
                log_expert_assignments(expert_indices, num_samples=20, num_experts=module.num_experts)

    moe_block.register_forward_hook(hook_fn)
    logger.info(f"Added spatial coherence hook to {moe_block.__class__.__name__}")
