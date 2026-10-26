"""Knowledge distillation into ternary students.

loss = alpha * CE(student, labels) + (1 - alpha) * T^2 * KL(softmax(teacher/T) || softmax(student/T))

The teacher (usually the dense or native-BitNet model the student was derived from) runs in
eval mode without gradients. Teacher and student must share a tokenizer. The KL term is
computed only on positions where loss_mask is 1 (response tokens during SFT).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def kd_loss(student_logits, teacher_logits, targets, loss_mask=None, alpha=0.5, temperature=2.0,
            chunk: int = 1024):
    from ..architectures.transformer import lm_loss
    ce = lm_loss(student_logits, targets, loss_mask)
    V = student_logits.size(-1)
    s = student_logits.reshape(-1, V)
    t = teacher_logits.reshape(-1, V)
    valid = (targets.reshape(-1) != -100).float()
    if loss_mask is not None:
        valid = valid * loss_mask.reshape(-1).float()
    kl_sum = s.new_zeros((), dtype=torch.float32)
    for i in range(0, s.shape[0], chunk):  # chunked to bound fp32 memory with large vocabularies
        ls = F.log_softmax(s[i:i + chunk].float() / temperature, -1)
        lt = F.log_softmax(t[i:i + chunk].float() / temperature, -1)
        kl = (lt.exp() * (lt - ls)).sum(-1)
        kl_sum = kl_sum + (kl * valid[i:i + chunk]).sum()
    kl = kl_sum / valid.sum().clamp(min=1.0) * temperature ** 2
    return alpha * ce + (1 - alpha) * kl, {"ce": ce.detach(), "kl": kl.detach()}
