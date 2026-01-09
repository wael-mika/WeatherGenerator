"""
Router Diagnostics - Analyze why router loss is stagnating

This module provides detailed diagnostics to understand router behavior:
- Are position embeddings being used?
- Is load balancing conflicting with task performance?
- Are features dominating position embeddings?
"""

import logging

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def analyze_router_internals(
    moe_block,
    sample_input: torch.Tensor,
    log_details: bool = True
) -> dict:
    """
    Analyze internal router behavior to diagnose stagnation.

    Args:
        moe_block: The MoEBlock to analyze
        sample_input: Sample input [batch, seq_len, dim_in]
        log_details: Whether to log detailed analysis

    Returns:
        Dictionary with diagnostic metrics
    """
    metrics = {}

    with torch.no_grad():
        batch_size, seq_len, dim_in = sample_input.shape

        # Get router
        router = moe_block.router

        # Check if spatial router
        is_spatial = hasattr(router, 'position_embed')

        if is_spatial:
            # === Spatial Router Diagnostics ===

            # Get position embeddings
            position_ids = torch.arange(seq_len, device=sample_input.device)
            pos_embed = router.position_embed(position_ids)  # [seq_len, pos_dim]

            # Concatenate features + position
            pos_embed_expanded = pos_embed.unsqueeze(0).expand(batch_size, -1, -1)
            x_with_pos = torch.cat([sample_input, pos_embed_expanded], dim=-1)

            # Get router logits
            router_logits = router.router_weights(x_with_pos)  # [batch, seq_len, num_experts]

            # === Metric 1: Position Embedding Usage ===
            # Check if position embeddings contribute to routing decisions

            # Route with position embeddings
            router_probs_with_pos = F.softmax(router_logits, dim=-1)
            router_probs = router_probs_with_pos  # Assign for common metrics

            # Route without position embeddings (features only)
            features_only = sample_input
            # Pad to match router input dimension
            padding = torch.zeros(
                batch_size, seq_len,
                router.router_weights.in_features - dim_in,
                device=sample_input.device
            )
            features_padded = torch.cat([features_only, padding], dim=-1)
            router_logits_no_pos = router.router_weights(features_padded)
            router_probs_no_pos = F.softmax(router_logits_no_pos, dim=-1)

            # Measure difference (KL divergence)
            kl_div = F.kl_div(
                router_probs_no_pos.log(),
                router_probs_with_pos,
                reduction='batchmean'
            )
            metrics['position_contribution_kl'] = kl_div.item()

            # === Metric 2: Position Embedding Variance ===
            # Check if position embeddings are diverse or collapsed
            pos_embed_std = pos_embed.std(dim=0).mean().item()
            pos_embed_mean_norm = pos_embed.norm(dim=-1).mean().item()
            metrics['position_embed_std'] = pos_embed_std
            metrics['position_embed_norm'] = pos_embed_mean_norm

            # === Metric 3: Router Weight Analysis ===
            # Check how much router attends to features vs position
            router_weights = router.router_weights.weight  # [num_experts, dim_in + pos_dim]
            feature_weights = router_weights[:, :dim_in]  # Features part
            position_weights = router_weights[:, dim_in:]  # Position part

            feature_weight_norm = feature_weights.norm(dim=-1).mean().item()
            position_weight_norm = position_weights.norm(dim=-1).mean().item()

            metrics['feature_weight_norm'] = feature_weight_norm
            metrics['position_weight_norm'] = position_weight_norm
            metrics['position_to_feature_ratio'] = position_weight_norm / (feature_weight_norm + 1e-8)

        else:
            # === Basic Router Diagnostics ===
            router_logits = router.router_weights(sample_input)
            router_probs = F.softmax(router_logits, dim=-1)

        # === Common Metrics (both router types) ===

        # Metric 4: Router Entropy (diversity of routing)
        router_entropy = -(router_probs * (router_probs + 1e-10).log()).sum(dim=-1).mean().item()
        metrics['router_entropy'] = router_entropy
        metrics['max_entropy'] = torch.log(torch.tensor(moe_block.num_experts)).item()

        # Metric 5: Router Confidence
        max_probs = router_probs.max(dim=-1)[0]
        metrics['avg_max_prob'] = max_probs.mean().item()
        metrics['routing_confidence'] = max_probs.mean().item()

        # Metric 6: Logit Statistics
        metrics['logit_mean'] = router_logits.mean().item()
        metrics['logit_std'] = router_logits.std().item()
        metrics['logit_max'] = router_logits.max().item()
        metrics['logit_min'] = router_logits.min().item()

        # === Metric 7: Load Balancing Analysis ===
        # Get expert assignments
        expert_indices = router_probs.argmax(dim=-1)  # [batch, seq_len]
        expert_counts = torch.bincount(
            expert_indices.flatten(),
            minlength=moe_block.num_experts
        ).float()
        expert_distribution = expert_counts / expert_counts.sum()

        # Compute load balance loss manually
        router_probs_mean = router_probs.mean(dim=[0, 1])  # [num_experts]
        expert_usage_mean = expert_distribution

        # Load balance loss = num_experts * sum(prob_mean * usage_mean)
        # (penalizes when an expert gets both high routing prob AND high usage)
        lb_loss = moe_block.num_experts * (router_probs_mean * expert_usage_mean).sum()

        metrics['load_balance_loss'] = lb_loss.item()
        metrics['expert_usage_std'] = expert_distribution.std().item()

        # Check if load balancing is perfect (all experts equal)
        perfect_balance = 1.0 / moe_block.num_experts
        balance_deviation = (expert_distribution - perfect_balance).abs().mean().item()
        metrics['balance_deviation'] = balance_deviation

    # === Logging ===
    if log_details:
        logger.info("\n" + "=" * 80)
        logger.info("Router Internal Diagnostics")
        logger.info("=" * 80)

        if is_spatial:
            logger.info("\n🗺️  Spatial Router Analysis:")
            logger.info(f"  Position Contribution (KL divergence): {metrics['position_contribution_kl']:.6f}")
            if metrics['position_contribution_kl'] < 0.001:
                logger.warning(f"    ⚠️  Position embeddings barely affect routing!")
                logger.warning(f"    → Router may be ignoring geographic information")
            elif metrics['position_contribution_kl'] < 0.01:
                logger.info(f"    ⚡ Weak position influence on routing")
            else:
                logger.info(f"    ✅ Position embeddings significantly affect routing")

            logger.info(f"\n  Position Embedding Statistics:")
            logger.info(f"    Std dev: {metrics['position_embed_std']:.4f}")
            logger.info(f"    Mean norm: {metrics['position_embed_norm']:.4f}")

            logger.info(f"\n  Router Weight Analysis:")
            logger.info(f"    Feature weight norm: {metrics['feature_weight_norm']:.4f}")
            logger.info(f"    Position weight norm: {metrics['position_weight_norm']:.4f}")
            logger.info(f"    Position/Feature ratio: {metrics['position_to_feature_ratio']:.4f}")
            if metrics['position_to_feature_ratio'] < 0.01:
                logger.warning(f"    ⚠️  Router weights heavily favor features over position!")
                logger.warning(f"    → Geographic information is being ignored")
            elif metrics['position_to_feature_ratio'] < 0.1:
                logger.info(f"    ⚡ Router moderately attends to position")
            else:
                logger.info(f"    ✅ Router gives significant weight to position")

        logger.info(f"\n📊 Router Output Statistics:")
        logger.info(f"  Entropy: {metrics['router_entropy']:.4f} / {metrics['max_entropy']:.4f}")
        logger.info(f"    → Measures diversity (max = {metrics['max_entropy']:.4f} = all experts equally likely)")
        if metrics['router_entropy'] < 0.5:
            logger.warning(f"    ⚠️  Very low entropy - router is collapsed!")
        elif metrics['router_entropy'] < metrics['max_entropy'] * 0.7:
            logger.info(f"    ⚡ Moderate entropy")
        else:
            logger.info(f"    ✅ High entropy - diverse routing")

        logger.info(f"\n  Routing Confidence: {metrics['routing_confidence']:.2%}")
        logger.info(f"    → Average max probability per token")
        if metrics['routing_confidence'] > 0.9:
            logger.info(f"    ✅ High confidence (strong routing decisions)")
        elif metrics['routing_confidence'] > 0.6:
            logger.info(f"    ⚡ Moderate confidence")
        else:
            logger.warning(f"    ⚠️  Low confidence (uncertain routing)")

        logger.info(f"\n  Logit Statistics:")
        logger.info(f"    Mean: {metrics['logit_mean']:.4f}")
        logger.info(f"    Std:  {metrics['logit_std']:.4f}")
        logger.info(f"    Range: [{metrics['logit_min']:.4f}, {metrics['logit_max']:.4f}]")
        if metrics['logit_std'] < 0.1:
            logger.warning(f"    ⚠️  Very low std - logits are nearly uniform!")

        logger.info(f"\n⚖️  Load Balancing Analysis:")
        logger.info(f"  Load balance loss: {metrics['load_balance_loss']:.6f}")
        logger.info(f"    → Current load balance penalty")
        logger.info(f"  Expert usage std: {metrics['expert_usage_std']:.4f}")
        logger.info(f"  Balance deviation: {metrics['balance_deviation']:.4f}")
        logger.info(f"    → Distance from perfect balance (0 = perfect)")

        if metrics['balance_deviation'] < 0.05:
            logger.info(f"    ✅ Near-perfect load balancing")
            if metrics['load_balance_loss'] > 1.1:
                logger.warning(f"    ⚠️  BUT load balance loss is still high!")
                logger.warning(f"    → Load balancing objective may be too aggressive")
                logger.warning(f"    → Consider reducing ae_global_moe_load_balance_weight")
        elif metrics['balance_deviation'] < 0.1:
            logger.info(f"    ⚡ Good load balancing")
        else:
            logger.warning(f"    ⚠️  Poor load balancing (imbalanced experts)")

        logger.info("=" * 80 + "\n")

    return metrics


def analyze_loss_components(
    moe_block,
    sample_input: torch.Tensor,
    log_details: bool = True
) -> dict:
    """
    Analyze different loss components to see what's dominating.

    Args:
        moe_block: The MoEBlock to analyze
        sample_input: Sample input [batch, seq_len, dim_in]
        log_details: Whether to log detailed analysis

    Returns:
        Dictionary with loss component metrics
    """
    metrics = {}

    with torch.no_grad():
        # Get router output
        router = moe_block.router
        router_probs, expert_indices, expert_weights = router(sample_input)

        # Compute load balance loss
        batch_size, seq_len, num_experts = router_probs.shape

        # Expert probability (mean routing probability to each expert)
        expert_probs_mean = router_probs.mean(dim=[0, 1])  # [num_experts]

        # Expert usage (fraction of tokens actually routed to each expert)
        expert_assignments = expert_indices[:, :, 0]  # Top-1 expert [batch, seq_len]
        expert_counts = torch.bincount(
            expert_assignments.flatten(),
            minlength=num_experts
        ).float()
        expert_usage = expert_counts / expert_counts.sum()

        # Load balance loss = num_experts * sum(p_i * f_i)
        # Where p_i = mean routing prob to expert i, f_i = fraction of tokens routed to expert i
        lb_loss = num_experts * (expert_probs_mean * expert_usage).sum()

        metrics['load_balance_loss_raw'] = lb_loss.item()
        metrics['load_balance_loss_weighted'] = (lb_loss * moe_block.load_balance_loss.weight).item()

        # Estimate routing loss (how much router wants to route differently)
        # This is approximate - real routing loss would need labels
        routing_confidence = expert_weights.max(dim=-1)[0].mean()
        metrics['routing_confidence'] = routing_confidence.item()

    if log_details:
        logger.info("\n" + "=" * 80)
        logger.info("MoE Loss Components Analysis")
        logger.info("=" * 80)

        logger.info(f"\n📊 Loss Breakdown:")
        logger.info(f"  Load balance loss (raw): {metrics['load_balance_loss_raw']:.6f}")
        logger.info(f"  Load balance weight: {moe_block.load_balance_loss.weight:.6f}")
        logger.info(f"  Load balance loss (weighted): {metrics['load_balance_loss_weighted']:.6f}")
        logger.info(f"    → This is added to total loss")

        logger.info(f"\n  Routing confidence: {metrics['routing_confidence']:.2%}")
        logger.info(f"    → How confident router is in top expert choice")

        # Diagnosis
        logger.info(f"\n🔍 Diagnosis:")
        if metrics['load_balance_loss_weighted'] > 0.05:
            logger.warning(f"  ⚠️  Load balance loss is HIGH (>5% of typical main loss)")
            logger.warning(f"  → This penalty is forcing equal distribution")
            logger.warning(f"  → May prevent router from learning better patterns")
            logger.warning(f"  → Consider reducing ae_global_moe_load_balance_weight from {moe_block.load_balance_loss.weight:.6f}")
            logger.warning(f"  → Suggested: 0.005 - 0.01 (current: {moe_block.load_balance_loss.weight:.6f})")
        elif metrics['load_balance_loss_weighted'] > 0.02:
            logger.info(f"  ⚡ Load balance loss is moderate")
            logger.info(f"  → May be limiting router learning slightly")
        else:
            logger.info(f"  ✅ Load balance loss is reasonable")

        logger.info("=" * 80 + "\n")

    return metrics
