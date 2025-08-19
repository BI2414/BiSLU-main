import torch
import torch.nn as nn
import torch.nn.functional as F

#新增方法
class MLD(nn.Module):
    """
    Multi-Label Distillation loss for self-distillation
    as described in the BiSLU paper (Section 3.3)
    """

    def __init__(self, temperature=1.0, reduction='mean'):
        """
        Args:
            temperature: Temperature parameter for distillation
            reduction: Specifies the reduction to apply to the output
        """
        super(MLD, self).__init__()
        self.temperature = temperature
        self.reduction = reduction

    def forward(self, student_logits, teacher_logits):
        """
        Compute multi-label distillation loss

        Args:
            student_logits: Logits from student model (intermediate intents)
            teacher_logits: Logits from teacher model (final intents)

        Returns:
            loss: Multi-label distillation loss
        """
        # Convert logits to probabilities using sigmoid
        p_S = torch.sigmoid(student_logits / self.temperature)
        p_T = torch.sigmoid(teacher_logits / self.temperature)

        # Avoid numerical instability by clipping probabilities
        p_S = torch.clamp(p_S, min=1e-7, max=1.0 - 1e-7)
        p_T = torch.clamp(p_T, min=1e-7, max=1.0 - 1e-7)

        # Compute positive term KL(p_S || p_T)
        pos_term = p_S * (torch.log(p_S) - torch.log(p_T))

        # Compute negative term KL(1-p_S || 1-p_T)
        neg_term = (1 - p_S) * (torch.log(1 - p_S) - torch.log(1 - p_T))

        # Sum over all labels and average over batch
        loss = (pos_term + neg_term).sum(dim=1)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss