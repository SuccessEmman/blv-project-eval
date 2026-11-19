# BLV caption evaluation — code and run guide

This is the starting guide for the Naija Jollof project deliverables. It documents the image-selection workflow, caption generation, human review, Gemini judging, CIDEr evaluation and final statistical analysis. Human-score comparisons use `BLV_Combined_Human_Ratings.xlsx` in **Final Statistical Analysis**.

The Drive item named `finalProjectCODE-NaijaJollof.zip` is currently a folder, not a downloadable ZIP archive. The code folders do not contain the full image dataset. Follow the source links below to obtain the required data.

## Choose what you want to reproduce

- **Reproduce the reported statistics:** start with Stage 6. Its workbook, saved judge responses and CIDEr inputs allow offline analysis; you do not need to regenerate captions or make API calls.
- **Trace how the dataset was prepared:** follow Stages 1–3. The sampling-history notebook contains old paths, intermediate attempts and manual-review steps. It is an audit record, not an unchanged, one-click pipeline.
- **Generate new model outputs:** use Stages 2, 4 and 5 in a separate working copy. New generations can differ from the saved experiment. Retain the published inputs and results.
- **Explore failure categories:** Stage 7 is optional and exploratory; it is not fully validated human coding.

## Data locations in Google Drive

Use these links rather than assuming that an old notebook path still matches the reorganized Drive.

| Required material | Location and contents |
|---|---|
| Source images | [Analysis/raw_data](https://drive.google.com/drive/folders/1XYLWNWiQT3smasoU_S08zAyuc01RTW69): `train`, `val`, and `test`. The experiment uses train and validation images only. |
| Caption annotations | [Caption annotation folder](https://drive.google.com/drive/folders/1gwNeti35zf0he2-mR-MeQ5tzhRZB17gq). Its displayed name is literally `raw_data/caption_annotations`; do not assume this represents two nested folders. |
| Image-quality annotations | [quality_annotations](https://drive.google.com/drive/folders/1c_wB-IGl9iBOWTiKvuQ7zujuJR40c4om). Use this extracted folder, not the similarly named `raw_data/quality_annotations` folder without first checking its contents. |
| Metadata and human caption references | [Analysis/outputs](https://drive.google.com/drive/folders/1uyWiTonE8E0vRqWswyadc1jpO7BKXU-e): `master_metadata.csv` and `captions_long.csv`. The latter contains human reference captions, not model generations. |
| Final selected images | [outputs/final_sample](https://drive.google.com/drive/folders/1GhNVmcYylO2LkDTsiCjDcRgwUOiJdcIJ): `sample_manifest.csv`, `sample_manifest_with_blur.csv`, `blur_settings.json`, `blur_inspection_notes.csv`. Manifests list images; they are not image files. |
| Separate blur pilot | [outputs/blur_calibration](https://drive.google.com/drive/folders/195M_TbohqfNhFupa_wkGV9fJ8RJQqKMI): `pilot_manifest.csv`, `pilot_with_blur_scores.csv`, `selected_threshold.json`, `threshold_search.csv`, `threshold_check_results.csv`. |
| Text-selection review trail | [outputs/text_review_round_4](https://drive.google.com/drive/folders/11oz_Azi-Qvl9Qs934rCs1k2HpNxklWZj) and [outputs/Archive](https://drive.google.com/drive/folders/1luuJT-hAVJiKx71mcdKnB68KpfjWpvEj). These preserve selection decisions and intermediate rounds. |
| Generated captions | [outputs/generated_captions](https://drive.google.com/drive/folders/1d3JFcrtaIzRO4rZRBZ6VCCZ6R3bRY4QV): `blip2_final_results.csv`, `llava_final_results.csv`, `combined_500_captions.csv`, and the final configuration JSON files. |
| Human-review source materials | [human_review](https://drive.google.com/drive/folders/1eO27WPoV-bhN7vEDamj2fcWRvjgaAgS3). Use the frozen workbook packaged in Final Statistical Analysis for reproducible numerical analysis. |
| Analysis-ready human scores | [BLV_Combined_Human_Ratings.xlsx](https://drive.google.com/file/d/1WGbYR5uXZeyZQhWVdfOtxU3pD9w7AAdM/view), now beside the Final Statistical Analysis scripts. |
| Saved analysis inputs and results | [Final Statistical Analysis](https://drive.google.com/drive/folders/1ls8oA-aD-acKjlhMStdkyr2BNpGkB5qt): `inputs/`, `results/`, notebook, scripts, workbook, requirements and its detailed README. |

## Before running the earlier notebooks: adapt the paths

1. Download or mount the source data, keeping train and validation image filenames unchanged. For a local working layout, use `raw_data/train`, `raw_data/val`, `raw_data/caption_annotations`, `raw_data/quality_annotations` and `outputs/`. Download the caption-annotation folder into the simple local name `caption_annotations` to avoid its literal slash in Drive.
2. Point each notebook's project-root variable to the directory that actually contains `raw_data` and `outputs`. This is separate from the directory containing the notebook itself.
3. Search the entire notebook for `/content/drive`, `MyDrive`, `/Users/a/`, `PROJECT_FOLDER`, `CAPTION_FOLDER`, `QUALITY_FOLDER`, `RAW_DATA`, `IMAGE_ROOT`, `BASE` and `OUTPUT_FOLDER`. Several later cells reset paths. Changing only the first setup cell is insufficient.
4. In Colab, mount Drive and use the mounted location shown by your own Drive. In VS Code, omit `google.colab` and `drive.mount(...)` cells and use local filesystem paths. A Drive web link is not a local filesystem path.
5. For the proposed local layout, point caption loading at `raw_data/caption_annotations` and quality loading at `raw_data/quality_annotations`. The historical sampling notebook instead contains `raw_data caption_annotations`; change that variable to your actual extracted folder.
6. Choose the matching Python notebook kernel. Install each stage's dependencies in its environment; the final-analysis requirements do not install the large captioning models. The caption-generation notebooks require the hardware and quantization setup used in their model-loading cells; do not assume the Mac's local CPU environment can run them unchanged.
7. Use a working copy for reruns. Check output destinations before executing cells that write CSV/JSON files. Keep the frozen manifests, captions and human-rating workbook intact.

## Run order

### 1. Image selection and blur analysis

File: **Image selection and blur analysis/BLV Project_history.ipynb**.

After adapting paths, work through the relevant stages in order:

1. Load train and validation caption annotations and quality annotations. Caption JSON files are dictionaries with image records; quality JSON files are lists with flaw votes. Do not interchange these files merely because both are named `train.json` or `val.json`.
2. Match quality records to image filenames, create `master_metadata.csv`, and collect valid reference captions in `captions_long.csv`.
3. Apply the recorded mutually exclusive sampling rules and random seed. The final quotas are 70 clear, 70 blurry, 60 low-light and 50 text images.
4. Preserve the text-image approval/rejection decisions made by Ebere and Samuel. Replacement rounds are manual selection steps; do not run all historical rounds blindly or recreate approvals automatically. Use the saved final manifest when reproducing the reported experiment.
5. Verify that all 250 selected images open, then calculate Laplacian-variance blur scores using the exact procedure recorded in `blur_settings.json`.
6. Use the separate 200-image blur pilot: 100 images to choose the threshold and 100 reserved images to check it, with 50 blurry and 50 clear in each half. Verify zero overlap with the final 250-image sample.

Expected saved threshold: approximately **164.678108**, with calibration balanced accuracy **77%** and reserved-set balanced accuracy **71%**. Blur scores supplement the annotation-based image groups; they do not replace them. This threshold was not validated for the low-light or text groups.

For local image validation, the final sample and separate pilot together contain 450 unique images. The history notebook is retained to document development; path adaptation and review of its intermediate cells remain necessary before a full replay.

### 2. Generate model captions

Files: **BLIP_LLAVA Caption Generation/BLIP2_LLaVA_Caption_Generation.ipynb** and **LLaVA_Caption_Generation.ipynb**.

- Point `PROJECT_FOLDER`, `RAW_DATA`, `OUTPUT_FOLDER` and `MANIFEST_FILE` to the prepared data layout, including later cells that redefine them.
- Use the same final `sample_manifest.csv` for both models. Verify 250 unique filenames and readable images before generation.
- Run setup, model loading, image checks, pilot and full-generation sections in their recorded order. Retain the final instruction and generation settings recorded by the experiment.
- Save each model's final CSV and configuration JSON, then use the combination section to create the 500-caption table: two captions for each of 250 images.
- Existing saved outputs are in the generated-captions folder linked above. Do not substitute pilot generations for final results.

### 3. Human review and score assembly

Four reviewers score the same caption-image pairs using rubric v1.2. Ebere and Samuel's earlier text-image selection review is a separate task from caption scoring. Preserve caption IDs, filenames, caption text, scores, reasons and TR applicability when combining reviews; never match rows solely by their position after sorting.

The main analysis reads `BLV_Combined_Human_Ratings.xlsx`. N/A is not zero. Numerical TR analysis uses the common subset where all four reviewers consider text applicable. Do not replace human scores with Gemini scores or exploratory taxonomy labels.

### 4. Gemini judging — only if new responses are needed

Files: **Gemini Judge/BLV Gemini Judge.ipynb**, `judge_run.py`, `judge_inputs.json`, `requirements.txt`.

- Keep those files together, install that folder's requirements and ensure the notebook's `PACKAGE` points to the folder containing `judge_run.py`.
- Set `IMAGE_ROOT` to the real image directory; the supplied notebook's original path points to Ebere's local `blv-project/raw_data` folder.
- Load the API key privately through the notebook prompt or an environment variable. Do not put it in shared notebook cells, GitHub or the deliverable files.
- Run the preflight checks, then the small pilot and full-run cells as needed. The configured judge is `gemini-3.1-flash-lite`. Resume from saved responses rather than deleting completed work after a quota interruption.
- The judge sees the image, caption, rubric and recorded reference text; human scores are not supplied in its prompt.
- For the reported statistics, reuse `Final Statistical Analysis/inputs/judge_results.jsonl` and `experiment_config.json`. Do not invoke the old `judge_run.analyze()` method: it is disabled in this deliverable. Human-score comparisons belong in Stage 6.

### 5. CIDEr evaluation

File: **CIDEr Evaluation/cider_evaluation.ipynb**.

Set `BASE` to the actual `outputs` directory, `REFS` to `captions_long.csv`, and model inputs to the two final caption CSVs in `generated_captions`. Adapt the Colab mount and installation cells if running locally. Run reference filtering, caption matching, scoring and export in order.

Only 217 of the 250 images have valid human reference captions, giving 434 scored captions. Exclude the other 33 images from CIDEr comparisons; do not assign them zero. Keep per-caption full-precision scores for correlation analysis and distinguish raw scores from any display multiplied by 100.

The standalone statistical package already contains the aligned `inputs/cider_scores.csv`. Reuse this for the recorded experiment. A new CIDEr run must be aligned by filename, model and caption ID before replacing that input; renaming an arbitrary CSV is insufficient.

### 6. Final Statistical Analysis — the main human-score results

Download this entire folder with the workbook, `inputs/` and `results/` still inside it. Keep `refresh_analysis.py`, `prepare_deliverables.py`, `agreement_methods.py`, `requirements.txt` and the notebook together.

In a terminal opened in that downloaded folder:

```bash
python -m pip install -r requirements.txt
python refresh_analysis.py
python prepare_deliverables.py
```

Alternatively, open **BLV Human Evaluation.ipynb**, select the prepared Python kernel and run its cells in order. See this folder's own README for full methods and environment details. This stage runs offline after installation and writes generated outputs to its `results/` folder.

It computes Spearman (judge versus human), the six reviewer-pair Cohen kappa comparisons, Kendall tau-b (CIDEr versus human), model summaries, confidence intervals and validation checks. Team kappa is the mean of six pairwise values, not one four-reviewer Cohen kappa. Caption-level Kendall correlation is not itself a two-model ranking test.

Sanity checks for the packaged human-score analysis:

- 500 captions, 250 images and 2,000 EH ratings.
- Common numerical TR subset: 102 captions from 51 images; 408 ratings from four reviewers. The seven captions with differing applicability judgments remain outside this common subset.
- Main Spearman: EH approximately 0.633686; TR approximately 0.714366.
- Main CIDEr–human Kendall tau-b: EH approximately 0.210937; TR approximately 0.075625.

The workbook is the human-score source. Generated result tables are outputs, not substitute input workbooks. Earlier snapshots and the separate text-review sensitivity analysis must not be mixed into this main run.

### 7. Failure Taxonomy — exploratory, optional

Folder: **Failure Taxonomy - Exploratory**. Read its `README.txt` before using `failure_taxonomy.py` or the saved CSVs/chart.

This is an automated keyword-rule analysis of captions and written explanations, including saved Gemini explanations for some rules. It is **not fully validated human coding**, does not independently establish what an image contains, and does not alter EH/TR scores. Treat counts as exploratory. In particular, an EH score of 1 is not always over-hedging, missing terminal punctuation is not proof of token-limit truncation, and the saved table's “All photos” totals actually count captions. A spot-check workbook alone does not establish that all 500 captions were validated.

If included in the report, describe it as an exploratory automated failure taxonomy requiring manual validation. It is not required to reproduce Stage 6.

## Completion checks

Confirm that required files exist, caption IDs/filenames match across inputs, final and pilot samples do not overlap, both models use the same final sample, and N/A remains distinct from zero. Review `results/validation.json` after the statistical run. Access to linked data is governed by the existing Drive permissions; this documentation does not change sharing settings.
