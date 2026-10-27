# Training the 2B specialists on cloud GPUs (Modal, Colab, Kaggle)

Both platforms give a free NVIDIA T4 (16 GB) for time-limited sessions:

| | Colab (free) | Kaggle |
| --- | --- | --- |
| Needs | a Google account | a phone-verified account (for GPU and internet) |
| GPU time | not guaranteed, varies | ~30 h/week (shown in account settings) |
| Session | up to ~12 h, ends early if the tab is closed or idle | 12 h, runs in the background |
| Checkpoints kept in | Google Drive (`MyDrive/TriForge`), copied during training | the notebook version's output |

A full `bitnet2b_partial` run is estimated at 11–22 hours (not measured yet), so it is a chain of
sessions that each resume from the newest checkpoint. Start every specialist with `PRESET = "quick"`
(at most 30 minutes of training): its `tokens_per_s` replaces the estimate.

## What is different from the RTX 4060 presets

`configs/<specialist>/bitnet2b_{quick,partial}_t4.yaml` inherit the RTX 4060 preset and change only
the runtime (`configs/base/runtime_t4.yaml`):

* **fp16 autocast with loss scaling** (`runtime.precision: fp16`). The T4 (compute 7.5) has no native
  bf16. The scaler state is saved in each checkpoint, so resuming is exact.
* **fp32 trainable weights** (`runtime.param_dtype: fp32`). Loss scaling needs fp32 gradients, and fp32
  latent weights also keep the small QAT updates (lr 1e-5) that bf16 would round away. Cost: about
  2.2 GB more GPU memory for the last 8 blocks, well within 16 GB. Frozen packed blocks and the
  embedding are unchanged.
* The ReLU² GLU is computed in fp32 when its input is fp16 (`ReLU(g)² · up` can exceed 65504).
* `keep_last: 1` and a checkpoint every 50 steps. One checkpoint is roughly 4–5 GB (estimate).
* One GPU only (no multi-GPU training). A T4 is roughly as fast as the laptop's RTX 4060, so the cloud
  adds GPU hours rather than speed.

Output directories end in `_t4`, so laptop and cloud runs never mix.

`scripts/cloud_session.py` runs one session (`--profile t4` or `l4` picks the presets): it restores the newest checkpoint and the tokenized data
from `--input-root`, prepares the data if there is none, trains with `--resume auto` until
`--save-margin-min` (20) minutes before `--budget-hours` (11.5), and with `--sync-dir` copies every
new checkpoint there while training. `--export-when-done` exports the model once the schedule is finished.

## Modal

Modal's Starter plan includes $30 of compute per month (usage beyond that is billed, so set a
spending limit in the Modal dashboard). On-demand GPUs, runs keep going with the laptop closed, and
one function call can last up to 24 hours. `cloud/modal_train.py` uses an **L4** (24 GB, native bf16)
by default with the `bitnet2b_{quick,partial}_l4.yaml` presets (`configs/base/runtime_l4.yaml`: bf16
autocast, fp32 trainable weights, micro-batch 2 × half the accumulation, same effective batch).

```bash
pip install modal
modal setup                                       # once: sign in from the browser
cd TernaryAI                                      # run from the repo root (it is copied into the image)
modal run --detach cloud/modal_train.py --specialist trimath --preset quick
modal app logs triforge                           # follow a detached run (or use the dashboard)
modal volume ls triforge checkpoints/trimath
modal volume get triforge checkpoints/exports ./exports
```

* Checkpoints, tokenized data and the Hugging Face cache are on the Volume `triforge`, so the model is
  downloaded once and a run that hits the 24 h limit resumes with the same `modal run` command.
* `TRIFORGE_GPU=A10G` picks an A10G (same presets); `TRIFORGE_GPU=T4` a T4 (fp16 `_t4` presets).
* Cost estimate (not measured): Modal lists the L4 at roughly $0.80/h, so the 30-minute quick run is
  well under $1 and a full `partial` run (speed unknown until the quick run reports `tokens_per_s`)
  probably costs a few dollars to ~$15. Check current prices on modal.com/pricing.
* Not tested on Modal yet: the first quick run is also the test of the image and the Volume.

## Colab

1. Open `colab/triforge_colab.ipynb`: in Colab, **File → Open notebook → GitHub**, tick
   *Include private repos*, pick Vedant-Sood/TernaryAI, branch `claude/project-thread-psmmpb`
   (or **File → Upload notebook** with the file).
2. **Runtime → Change runtime type → T4 GPU.**
3. **Secrets** (key icon, left sidebar): add `GITHUB_TOKEN` with *Notebook access* on. Use a fine-grained
   GitHub token for this repo with Contents: read-only. Not needed if the repo is public.
4. In the second cell choose `SPECIALIST` and `PRESET`, then **Runtime → Run all** and allow Drive access.
5. Keep the tab open. To continue after a disconnect or the session limit, open the notebook and
   **Run all** again.

Google Drive storage: the free 15 GB holds one checkpoint plus the data. Replaced checkpoints are
emptied before deletion, so the Drive trash stays small, but check **Storage** if Drive reports it is full.
Colab's free VM has ~12 GB RAM; if loading the 2B model runs out of memory, report it (the loader
streams weights, but this has not been tested on Colab).

## Kaggle

1. **Create → New Notebook → File → Import Notebook**: GitHub tab, Vedant-Sood/TernaryAI, branch
   `claude/project-thread-psmmpb`, file `kaggle/triforge_kaggle.ipynb` (or upload the file).
2. Settings: **Accelerator: GPU T4 x2**, **Internet: on** (both need phone verification).
3. **Add-ons → Secrets**: `GITHUB_TOKEN` as above.
4. **Save Version → Save & Run All (Commit)**. Logs are in the version's **Logs** tab, and the checkpoint
   and data are in its **Output**.
5. Next session: **Add Input → Your Work →** this notebook's output (or **Output → New Dataset** and add
   that), then **Save & Run All** again.

With the Kaggle CLI (`pip install kaggle`, token from Settings → API): set `id` in
`kaggle/kernel-metadata.json` to `<your-username>/triforge-2b-ternary`, then `kaggle kernels push -p kaggle`.
The secret still has to be added once in the web editor.
