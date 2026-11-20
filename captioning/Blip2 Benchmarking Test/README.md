# BLIP-2 Karpathy Benchmark — Local VS Code Edition

This project converts `blip2_karpathy_benchmark_debug_fixed.py` from Google Colab into a normal Python command-line script for VS Code. No Colab notebook, Google Drive mount, or `!apt-get` commands are needed.

It uses the same MS COCO **2014 Karpathy test split**, pretrained checkpoint (`Salesforce/blip2-flan-t5-xl`), `Describe this image.` prompt, 4-beam decoding, 64-new-token cap, `do_sample=False`, seed 42 for sample selection, and 8-bit quantized model loading. Image bytes remain unmodified and the checkpoint processor handles resizing/cropping. Choose the COCO-fine-tuned model explicitly to benchmark that separate configuration.

## 1. Requirements

- **Python 3.11**, 64-bit, recommended for easiest package compatibility.
- VS Code with the Python extension (use **Python: Select Interpreter** to choose `.venv`).
- An **NVIDIA GPU with CUDA support**, ideally RTX 20-series/T4 or newer, and sufficient GPU RAM. The 8-bit Flan-T5-XL checkpoint is large; a low-memory GPU may still fail. Local 8-bit inference is not practical using CPU-only PyTorch or unsupported integrated GPUs. Run `--check-setup` to inspect your machine.
- Install **Java JDK or JRE** and verify `java -version` works in a new terminal. The COCO PTB tokenizer requires Java even for CIDEr; METEOR and SPICE also use Java. For Windows, use a Java distribution with `java` on your PATH.
- Internet connection for model weights, Karpathy metadata, and COCO JPEGs unless cached locally.

## 2. Set up in VS Code (Windows PowerShell)

Open this project folder in VS Code (File > Open Folder), then Terminal > New Terminal:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

If PowerShell blocks activation, skip the activation command and call `.\.venv\Scripts\python.exe` in place of `python` below. You can also use Command Prompt: `.venv\Scripts\activate.bat`.

**Important:** Install the appropriate **CUDA-enabled PyTorch** wheel from the selector at https://pytorch.org/get-started/locally/ for your NVIDIA driver and Windows setup, then:

```powershell
python -m pip install -r requirements.txt
python blip2_karpathy_local.py --check-setup
java -version
```

`CUDA available: True` and a GPU name should appear. If CUDA is false, **do not** start a 5,000-image run; first repair your PyTorch/CUDA environment. A CUDA Toolkit installation is not automatically required for standard PyTorch wheels; an appropriate NVIDIA driver and compatible GPU are essential.

On Linux/macOS, replace the venv activation step with `source .venv/bin/activate`. macOS CPU-only setups will not run the BLIP-2 8-bit generation stage.

## 3. Run a small test before the full benchmark

```powershell
python blip2_karpathy_local.py --num-images 5
```

This loads the Karpathy metadata, takes a reproducible five-image sample (seed 42), downloads those images, loads BLIP-2, generates captions, and computes CIDEr. It writes to:

`results/project_pretrained_5_seed42/`

If you want to troubleshoot an SSL/image-host problem **before downloading model weights**, run:

```powershell
python blip2_karpathy_local.py --num-images 5 --download-only
```

The image download attempts official COCO HTTPS endpoints with normal SSL validation. It **does not** disable TLS certificate checks. If downloads fail, obtain and extract the original COCO 2014 validation JPEGs, then use:

```powershell
python blip2_karpathy_local.py --num-images 5 --local-image-dir "D:\Datasets\COCO\val2014"
```

The path above is an example; replace it with your own local folder.

After the smoke test, try a 50-image verification:

```powershell
python blip2_karpathy_local.py --num-images 50
```

The standard **full test** is 5,000 images. Only run it when system capacity, image access, and the short test are satisfactory:

```powershell
python blip2_karpathy_local.py --num-images 5000
```

To use the **COCO-fine-tuned checkpoint**, which is a different experiment:

```powershell
python blip2_karpathy_local.py --experiment coco_finetuned_reference --num-images 50
```

The script adopts the original notebook's fine-tuned prompt `a photo of`. To claim exact paper reproduction, independently verify all checkpoint and evaluation settings match the paper; using a subset cannot reproduce the full-test score.

## 4. Metrics and saved files

By default, **CIDEr only** is computed to keep troubleshooting and runtime manageable. To also compute BLEU-1/2/3/4, METEOR, and SPICE:

```powershell
python blip2_karpathy_local.py --num-images 50 --score-only --metrics all
```

METEOR/SPICE have Java dependencies and SPICE may download additional resources. Errors in optional scorers are reported; CIDEr is required.

Outputs are saved under `results/<experiment>_<N>_seed<seed>/`:

- `run_config.json` — reproducibility settings and sample ID hash
- `predictions.jsonl` — a completed caption on each line (incrementally saved)
- `metrics.json` — aggregate results and x100 paper-scale formatting
- `prediction_audit.csv` — prediction, five references, per-image CIDEr
- `coco_predictions.json` — COCO-style predictions
- `per_image_cider_raw.json` — per-image CIDEr

The image cache is under `image_cache_val2014/`. Re-running the **same** command will skip existing completed captions and reuse cached images. Different numbers of images use separate results folders. If code crashes, leave `predictions.jsonl` unchanged and rerun.

### Publication caveats

The standard paper score (e.g., CIDEr 144.5) was obtained using a COCO-fine-tuned BLIP-2 variant. Running the **pretrained** checkpoint with this project's VizWiz prompt is only a *pipeline check*. CIDEr on a tiny subset is also **not directly comparable** with the published 5,000-image CIDEr score, because CIDEr's document frequencies and other experimental conditions differ. Report exact model checkpoint, sample count, seed, metric version, prompt and decoding settings.

## 5. Common problems

| Message | What to do |
|---|---|
| `CUDA unavailable` | Install compatible NVIDIA drivers and CUDA-enabled PyTorch; verify `torch.cuda.is_available()` |
| `CUDA out of memory` | Free GPU memory, close other GPU apps, or use a GPU with more VRAM; 8-bit weights are still large |
| `ModuleNotFoundError` | Confirm VS Code uses `.venv` and reinstall `requirements.txt` |
| `Java not found` / tokenizer failure | Install Java; open a new terminal and rerun `java -version` |
| SSL / COCO image download failure | Try `--download-only`; use `--local-image-dir` with original COCO 2014 validation JPEGs |
| `Only X/N captions saved` | Run generation without `--score-only` to finish, then retry metric scoring |
| METEOR or SPICE errors | Use default CIDEr first; optional scorers can be run later |

**Validation performed for this delivery:** Python syntax compilation and local unit tests. Full model inference and online image retrieval have **not** been run here; they require your GPU, downloads and local environment.

## 6. Run directly with VS Code's ▶ / F5 button

This project also includes `.vscode/launch.json`. After selecting your `.venv` interpreter, open the **Run and Debug** view in VS Code (Ctrl+Shift+D), choose **BLIP-2: check local setup** or **BLIP-2: test 5 COCO images**, and press F5. The *download only* configuration is useful if the previous Colab SSL/network problem recurs locally.
