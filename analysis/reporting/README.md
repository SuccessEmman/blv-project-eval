# Report tables and figures

This folder generates report material from the human-score workbook and saved analysis in `analysis/blv_current_evaluation`. It also reads the sampling outputs and the exploratory taxonomy. It does not call Gemini or rerun captioning, and it does not change the inputs.

## Start in VS Code

Open the **blv-project-team** folder in VS Code. In the **terminal**, run these commands once to install the reporting packages:

```bash
conda activate blv-project
cd /Users/a/Desktop/blv-project-team
python -m pip install -r analysis/reporting/requirements.txt
```

Teammates should replace the `cd` path with their own repository location. This command uses the same statistical package versions as the final analysis, plus Matplotlib and Jinja2. Python 3.12 or later is required.

Open **analysis/reporting/BLV Report Tables and Figures.ipynb**, select the **blv-project** Python kernel, and run its cells in order. The notebook calls `make_report.py` and previews the results. No image files, GPU, API key, or internet connection are needed after installing the dependencies.

Alternatively, run this in the terminal from the repository root:

```bash
python analysis/reporting/make_report.py
```

To regenerate the underlying statistical analysis after deliberately changing the human workbook, follow the README in `analysis/blv_current_evaluation` first. This reporting script stops when the workbook no longer matches the saved statistics.

## Where your report material is saved

- `report_outputs/index.html`: a gallery of all figures and tables. Open this file in a browser, or use the notebook previews.
- `report_outputs/figures/`: each figure in PNG, PDF, and SVG. PNG is convenient for Word and slides; PDF is a vector format suitable for Overleaf.
- `report_outputs/tables/`: each table in full-precision CSV and a four-decimal LaTeX export. CSV opens in Excel. LaTeX needs `\usepackage{booktabs}`. Wide appendix tables may need a landscape page or a smaller selection of columns.
- `report_outputs/CAPTIONS.md`: figure captions and table notes to adapt for your report.
- `report_outputs/report_manifest.json`: source hashes, verification results, and the methods used for the exports.

Running the script again replaces its own exported files in `report_outputs`. It does not modify the final analysis package.

## Suggested main report material

Start with these five figures:

| Figure | Purpose |
| --- | --- |
| `01_workflow` | Explains the sample and which captions enter each analysis. |
| `02_blur_analysis` | Shows blur-score distributions, threshold selection, and the reserved-set confusion matrix. |
| `03_human_scores` | Compares the models' EH and applicable TR scores across image conditions. |
| `04_reviewer_agreement_unweighted` | Shows all six reviewer pairs and the mean unweighted Cohen's kappa. |
| `07_correlations` | Shows Spearman and Kendall estimates with confidence intervals. |

Use tables `01_dataset`, `03_blur_performance`, `05_coverage`, `07_human_model_scores`, `10_team_agreement`, and `11_correlations` when you need precise numbers. The other tables and figures are supporting material; you do not need to put every export in the main report.

## What the numbers mean

- EH is assessed on 500 captions from 250 images. The four reviewers supply 2,000 individual EH ratings.
- There are 415 individual numerical TR ratings. The common TR subset uses 102 captions from 51 images, giving 408 ratings. Seven captions with differing applicability judgments stay in the source data but are excluded from that common subset. N/A is never converted to zero.
- The common TR subset contains 50 text-group images and one blurry image. Clear and low-light TR cells are N/A. The single blurry image is shown descriptively, without an interval or a claim about the blurry group as a whole.
- CIDEr is available for 434 captions from 217 images. The 33 images without valid references are excluded, not assigned zero. The common TR/CIDEr comparison has 100 captions from 50 images.
- The blur threshold is calibrated on 100 pilot images and checked on a different 100. The 200 pilot images have no overlap with the final 250. The heuristic did not replace annotation-based group labels.
- Cohen's kappa is calculated separately for each reviewer pair. The team summaries average six pairwise values. Unweighted and linear-weighted results are explicitly distinguished. These are the human scores in the final workbook, not a reconstruction of initial independent ratings.
- Spearman compares the Gemini judge with human means. Kendall tau-b compares CIDEr with human means on matched captions. Neither measures exact agreement or percentage accuracy. Model-specific correlation differences are not a formal significance test.
- Saved correlation confidence intervals use 1,000 image-cluster bootstrap replicates. The new descriptive **mean-score** intervals use 2,000 image-bootstrap replicates with seeds beginning at 20261119. They are separate from the correlation intervals, and their creation does not change the saved analysis.
- The sensitivity figure excludes generation-flagged caption C0104. The separate post-review text-sensitivity analysis is outside this main package and is not mixed into these exports.
- Failure categories are exploratory outputs of keyword rules, including rules using judge reasons. They are not fully validated human coding. The majority-EH1 category can include colour-only errors and must not automatically be interpreted as over-hedging.
- No Wilcoxon result is created here: the main analysis package contains no saved Wilcoxon result to reproduce. Do not describe the descriptive model differences as statistically significant on that basis.

## Validation and source code

The script verifies the workbook, per-reviewer scores, caption IDs, image mappings, judge scores, CIDEr values, and sampling separation. It independently recalculates 61 correlation and kappa estimates before drawing the figures. It checks source hashes again after generation.

`make_report.py` contains these sections:

- `load`: input checks and statistical cross-checks.
- `dataset_and_blur`: sample tables and blur plots.
- `coverage`: analysis counts, workflow, and applicability.
- `model_scores`: human means, intervals, and rating distributions.
- `agreement`: reviewer-pair kappa tables and heatmaps.
- `correlations`: Spearman, Kendall, score pairs, and C0104 sensitivity.
- `cider_and_taxonomy`: matched model summaries and exploratory categories.
- `finish`: gallery, captions, and reproducibility manifest.

This reporting folder can be committed to your own branch after review. Creating it locally does not upload it to GitHub.
