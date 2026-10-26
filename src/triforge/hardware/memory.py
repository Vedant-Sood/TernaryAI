"""Training / inference memory estimator.

Deliberately conservative, and it never changes the model: it only reports whether a given
(model, strategy, micro-batch, sequence length) fits and where the bytes go.

Low-bit *inference* memory says little about *training* memory: during QAT every trainable
ternary weight still needs a high-precision latent copy, a gradient and optimizer state, and
activations are stored in bf16/fp32. Only frozen layers can live as packed 2-bit codes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

GiB = 1024 ** 3

OPTIM_BYTES = {          # optimizer state bytes per trainable parameter (on the GPU)
    "adamw": 8,          # fp32 exp_avg + exp_avg_sq
    "adamw8bit": 2,      # bitsandbytes 8-bit states
    "adamw_cpu": 0,      # states kept in system RAM
    "sgd": 0,
}


@dataclass
class MemoryEstimate:
    weights_trainable: float
    weights_frozen: float
    grads: float
    optimizer: float
    master_fp32: float
    activations: float
    logits: float
    overhead: float
    cpu_ram: float

    @property
    def total(self):
        return (self.weights_trainable + self.weights_frozen + self.grads + self.optimizer
                + self.master_fp32 + self.activations + self.logits + self.overhead)

    def as_gib(self):
        d = {k: round(v / GiB, 3) for k, v in asdict(self).items()}
        d["total_gpu"] = round(self.total / GiB, 3)
        return d


def estimate_training_memory(counts: dict, d_model: int, d_ff: int, n_layers: int, vocab: int,
                             micro_batch: int, seq_len: int, *, n_trainable_layers: int | None = None,
                             train_embeddings: bool = True, lora_rank: int = 0, param_bytes: int = 2,
                             optimizer: str = "adamw", fp32_master: bool = False,
                             grad_checkpointing: bool = True, frozen_format: str = "packed",
                             chunked_loss: bool = True, cuda_overhead_gib: float = 0.6) -> MemoryEstimate:
    """Bytes needed on the accelerator for one optimizer step.

    counts: TriForgeConfig.param_counts(). n_trainable_layers=None means all layers train.
    frozen_format: "packed" (2-bit codes, 0.25 B/weight), "int8" (1 B) or "bf16" (2 B).
    lora_rank > 0: ternary-preserving LoRA; latent weights frozen in bf16, only adapters train.
    """
    L = n_layers
    nt = L if n_trainable_layers is None else n_trainable_layers
    per_layer = counts["per_layer_linear"]
    emb = counts["embedding"] + counts["head"]
    norms = counts["norms"]

    if lora_rank:
        # every adapted linear gets r*(in+out); approximate with 7 projections per layer
        lora_params = L * lora_rank * (4 * d_model + 2 * d_model + 2 * (d_model + d_ff) + (d_ff + d_model))
        trainable = lora_params
        frozen_bytes = L * per_layer * 2 + emb * 2 + norms * 2  # latent weights must stay bf16
        trainable_bytes = lora_params * param_bytes
    else:
        trainable = nt * per_layer + norms + (emb if train_embeddings else 0)
        frozen_lin = (L - nt) * per_layer
        fb = {"packed": 0.25, "int8": 1.0, "bf16": 2.0}[frozen_format]
        frozen_bytes = frozen_lin * fb + (0 if train_embeddings else emb * 2)
        trainable_bytes = trainable * param_bytes

    grads = trainable * param_bytes
    optim = trainable * OPTIM_BYTES[optimizer]
    master = trainable * 4 if fp32_master and param_bytes == 2 else 0

    tok = micro_batch * seq_len
    a = 2  # bf16 activations
    # Per layer: attention in/out, q,k,v, MLP in, gate/up/hidden, plus quantized copies of inputs.
    per_layer_act = tok * (10 * d_model + 4 * d_ff) * a
    # Backward only needs activations of layers whose inputs need grad: with all layers frozen
    # below the first trainable block, those layers keep nothing.
    layers_with_grad = L if (train_embeddings or lora_rank) else nt
    if grad_checkpointing:
        acts = layers_with_grad * tok * d_model * a + per_layer_act
    else:
        acts = layers_with_grad * per_layer_act
    logits = tok * vocab * 4 * (0.25 if chunked_loss else 3)  # fp32 logits + grad (+ softmax)
    overhead = cuda_overhead_gib * GiB + 0.08 * (trainable_bytes + frozen_bytes + grads + optim + acts)
    cpu = trainable * 12 if optimizer == "adamw_cpu" else 0
    return MemoryEstimate(trainable_bytes, frozen_bytes, grads, optim, master, acts, logits, overhead, cpu)


def estimate_inference_memory(counts: dict, d_model: int, n_layers: int, n_kv_heads: int, head_dim: int,
                              seq_len: int, batch: int = 1, weight_format: str = "int8-cache",
                              cuda_overhead_gib: float = 0.4) -> dict:
    """weight_format: "bf16" | "packed" (2-bit, unpacked per matmul) | "int8-cache" (codes cached as int8)."""
    lin = counts["linear"]
    fb = {"bf16": 2.0, "packed": 0.25, "int8-cache": 1.25}[weight_format]
    weights = lin * fb + (counts["embedding"] + counts["head"] + counts["norms"]) * 2
    kv = 2 * n_layers * batch * seq_len * n_kv_heads * head_dim * 2
    total = weights + kv + cuda_overhead_gib * GiB
    return {"weights_gib": weights / GiB, "kv_cache_gib": kv / GiB, "total_gib": total / GiB}
