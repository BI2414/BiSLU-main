import torch
import torch.nn.functional as F


def scl_intent_loss_func(intent_vectors, intent_labels, temperature=0.07):
    """
    Supervised Contrastive Loss for intent detection (multi-label)

    Args:
        intent_vectors: Tensor of shape [batch_size, num_views, feature_dim]
        intent_labels: Tensor of shape [batch_size, num_intents] (multi-label)
        temperature: Temperature scaling parameter
    """
    batch_size, num_views, feat_dim = intent_vectors.shape
    device = intent_vectors.device

    # Normalize feature vectors
    intent_vectors = F.normalize(intent_vectors, dim=-1)

    # Reshape to combine batch and views dimensions
    features = intent_vectors.reshape(-1, feat_dim)  # [batch_size * num_views, feat_dim]

    # Expand labels to match the multi-view structure
    expanded_labels = intent_labels.repeat_interleave(num_views, dim=0)  # [batch_size * num_views, num_intents]

    # Compute similarity matrix
    similarity_matrix = torch.matmul(features, features.T) / temperature  # [N, N], N = batch_size * num_views

    # Create mask for positive pairs:
    # 1. Same sample in different views (always positive)
    same_sample_mask = torch.zeros((batch_size, batch_size), dtype=torch.bool, device=device)
    for i in range(batch_size):
        same_sample_mask[i, i] = True
    same_sample_mask = same_sample_mask.repeat_interleave(num_views, dim=0).repeat_interleave(num_views, dim=1)

    # 2. Different samples with at least one common intent label
    common_intent_mask = torch.matmul(expanded_labels.float(), expanded_labels.float().T) > 0
    positive_mask = same_sample_mask | common_intent_mask

    # Exclude self-comparison
    identity_mask = torch.eye(batch_size * num_views, dtype=torch.bool, device=device)
    positive_mask = positive_mask & (~identity_mask)

    # Compute logits
    exp_sim = torch.exp(similarity_matrix)

    # Compute numerator (positive pairs)
    numerator = torch.sum(exp_sim * positive_mask.float(), dim=1)

    # Compute denominator (all pairs except self)
    denominator = torch.sum(exp_sim, dim=1) - torch.diag(exp_sim)

    # Compute loss
    losses = -torch.log(numerator / (denominator + 1e-8))
    valid_losses = losses[torch.any(positive_mask, dim=1)]  # Filter anchors with no positive pairs

    if len(valid_losses) == 0:
        return torch.tensor(0.0, device=device)

    return valid_losses.mean()


def scl_slot_loss_func(slot_vectors, slot_labels, temperature=0.07):
    """
    Supervised Contrastive Loss for slot filling

    Args:
        slot_vectors: Tensor of shape [num_spans, num_views, feature_dim]
        slot_labels: Tensor of shape [num_spans] (integer labels)
        temperature: Temperature scaling parameter
    """
    num_spans, num_views, feat_dim = slot_vectors.shape
    device = slot_vectors.device

    # Normalize feature vectors
    slot_vectors = F.normalize(slot_vectors, dim=-1)

    # Reshape to combine span and views dimensions
    features = slot_vectors.reshape(-1, feat_dim)  # [num_spans * num_views, feat_dim]

    # Expand labels to match the multi-view structure
    expanded_labels = slot_labels.repeat_interleave(num_views, dim=0)  # [num_spans * num_views]

    # Compute similarity matrix
    similarity_matrix = torch.matmul(features, features.T) / temperature  # [N, N], N = num_spans * num_views

    # Create mask for positive pairs:
    # 1. Same span in different views (always positive)
    same_span_mask = torch.zeros((num_spans, num_spans), dtype=torch.bool, device=device)
    for i in range(num_spans):
        same_span_mask[i, i] = True
    same_span_mask = same_span_mask.repeat_interleave(num_views, dim=0).repeat_interleave(num_views, dim=1)

    # 2. Different spans with the same slot label
    same_label_mask = expanded_labels.unsqueeze(0) == expanded_labels.unsqueeze(1)

    # Combine masks
    positive_mask = same_span_mask | same_label_mask

    # Exclude self-comparison
    identity_mask = torch.eye(num_spans * num_views, dtype=torch.bool, device=device)
    positive_mask = positive_mask & (~identity_mask)

    # Compute logits
    exp_sim = torch.exp(similarity_matrix)

    # Compute numerator (positive pairs)
    numerator = torch.sum(exp_sim * positive_mask.float(), dim=1)

    # Compute denominator (all pairs except self)
    denominator = torch.sum(exp_sim, dim=1) - torch.diag(exp_sim)

    # Compute loss
    losses = -torch.log(numerator / (denominator + 1e-8))
    valid_losses = losses[torch.any(positive_mask, dim=1)]  # Filter anchors with no positive pairs

    if len(valid_losses) == 0:
        return torch.tensor(0.0, device=device)

    return valid_losses.mean()