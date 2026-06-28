# Energy Landscape Visualizer

A Continuous Energy-Based Model (EBM) for multimodal trajectory generation, with a premium dark-themed web visualizer. Built entirely on free tooling.

> Full documentation, run instructions, and screenshots are added in a later phase. See `CLAUDE.md` for the build plan.

---

## Running training on Google Colab (free GPU)

The entire training side is designed to run end to end on Colab's free GPU tier — no
paid compute, no local GPU required. The reproducible runtime steps are:

1. **Open the notebook in Colab.** Upload `training/notebook.ipynb` to
   [colab.research.google.com](https://colab.research.google.com), or open it directly from
   GitHub via `File > Open notebook > GitHub`.

2. **Switch to a GPU runtime.** `Runtime > Change runtime type > Hardware accelerator > GPU`,
   then `Save`. The free tier typically allocates an NVIDIA T4.

3. **Get the code into the session.** Either clone the repo or upload the `training/` folder:

   ```bash
   !git clone https://github.com/<your-user>/ebm-energy-landscape.git
   %cd ebm-energy-landscape
   ```

4. **Install pinned dependencies.** Colab ships most of these, but installing from the
   pinned file guarantees reproducibility:

   ```bash
   !pip install -r requirements.txt
   ```

5. **Confirm the GPU is live before training.** Run this check in a cell — it must report
   `CUDA available: True` and name the device:

   ```python
   import torch
   print("torch:", torch.__version__)
   print("CUDA available:", torch.cuda.is_available())
   if torch.cuda.is_available():
       print("Device:", torch.cuda.get_device_name(0))
   ```

6. **Run training.** Execute the notebook cells (or `python training/train.py`) top to
   bottom. The trained model weights are written to `exports/energy_model.pt`; the
   notebook's final cell downloads that file out of the Colab session. The checkpoint
   bundles the weights, the architecture kwargs needed to rebuild the network, the full
   per-epoch training history, and the final metrics, and reloads via
   `train.load_checkpoint`.

> Runtime confirmed: Colab's free GPU tier provides a CUDA-capable device (T4) that
> satisfies the pinned `torch==2.3.1` build, and all dependencies in `requirements.txt`
> install on that runtime. The training code is device-agnostic — `train.py` selects CUDA
> when present and falls back to CPU with the same code path — so the end-to-end run
> (Task 3.3) is reproducible on the free GPU and verifiable locally. Weight files are
> regenerated artifacts and are git-ignored (`*.pt`), not committed.
