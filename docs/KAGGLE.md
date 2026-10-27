# Training the 2B specialists on Kaggle (free T4 GPUs)

Kaggle Notebooks give about 30 GPU hours a week (the quota is shown in your account settings), on
2× T4 (16 GB each) or 1× P100, with sessions of at most 12 hours. A full `bitnet2b_partial` run is
estimated at 11–22 hours (not measured yet), so it is run as a chain of sessions that each resume from
the previous session's checkpoint.

## What is different from the RTX 4060 presets

`configs/<specialist>/bitnet2b_{quick,partial}_kaggle_t4.yaml` inherit the RTX 4060 preset and change
only the runtime (`configs/base/runtime_kaggle_t4.yaml`):

* **fp16 autocast with loss scaling** (`runtime.precision: fp16`). The T4 (compute 7.5) has no native
  bf16. The scaler state is saved in each checkpoint, so resuming is exact.
* **fp32 trainable weights** (`runtime.param_dtype: fp32`). Loss scaling needs fp32 gradients, and fp32
  latent weights also keep the small QAT updates (lr 1e-5) that bf16 would round away. Cost: about
  2.2 GB more GPU memory for the last 8 blocks, well within 16 GB. Frozen packed blocks and the
  embedding are unchanged.
* The ReLU² GLU is computed in fp32 when its input is fp16 (`ReLU(g)² · up` can exceed 65504).
* `keep_last: 1` and a checkpoint every 50 steps, to stay under Kaggle's 20 GB output limit.
* Only one of the two T4s is used (no multi-GPU training yet). A T4 is roughly as fast as the laptop's
  RTX 4060, so Kaggle adds GPU hours rather than speed.

Output directories end in `_kaggle`, so laptop and Kaggle runs never mix.

## First session

1. Kaggle → **Create → New Notebook → File → Import Notebook**, and upload `kaggle/triforge_kaggle.ipynb`.
2. Right-hand panel: **Accelerator: GPU T4 x2**, **Internet: on** (needs a phone-verified account).
3. **Add-ons → Secrets**: add `GITHUB_TOKEN`, a GitHub token with read access to this repo (fine-grained,
   Contents: read-only). Not needed if the repo is public. `HF_TOKEN` is optional (the base model and
   datasets are public).
4. In the second cell, choose `SPECIALIST` and `PRESET`. Start with `PRESET = "quick"`: it trains for
   at most 30 minutes and logs the real `tokens_per_s`, which replaces the 11–22 hour estimate.
5. **Save Version → Save & Run All (Commit)**. The run continues with the browser closed. Logs are under
   the version's **Logs** tab, and the checkpoint and tokenized data are in its **Output**.

## Next sessions

1. In the notebook editor: **Add Input → Your Work → Notebooks →** this notebook (its latest version's
   output). If it is not listed, open the finished version, **Output → New Dataset**, and add that dataset.
2. **Save & Run All** again. `scripts/kaggle_session.py` looks under `/kaggle/input` for the newest
   checkpoint of the same run, copies it (and the tokenized data, so it is not rebuilt) into
   `/kaggle/working`, and trains with `--resume auto` until about 20 minutes before the 12-hour limit.
3. Repeat until the log says `schedule finished`. The model is then exported to
   `checkpoints/exports/<specialist>-2b-<preset>-kaggle/`.

## From a computer with the Kaggle CLI

`kaggle/kernel-metadata.json` describes the same notebook (private, T4, internet on). Set `id` to
`<your-kaggle-username>/triforge-2b-ternary`, then:

```bash
pip install kaggle                       # API token: kaggle.com → Settings → API → Create New Token
kaggle kernels push -p kaggle            # uploads and starts a run
kaggle kernels status <user>/triforge-2b-ternary
kaggle kernels output <user>/triforge-2b-ternary -p out/   # download the output
```

The secrets (`GITHUB_TOKEN`) still have to be added once in the web editor. For a later session, add
the previous output to `kernel_sources` (or a dataset made from it to `dataset_sources`) and push again.

## Running the driver elsewhere

`scripts/kaggle_session.py` works on any machine:

```bash
python scripts/kaggle_session.py --specialist trimath --preset quick --input-root /path/to/previous/outputs
```

`--budget-hours` (default 11.5) and `--save-margin-min` (default 20) set the wall-clock cap.
