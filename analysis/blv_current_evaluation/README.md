# BLV human evaluation — main analysis

This package reproduces the main analysis using the human scores in
`BLV_Combined_Human_Ratings.xlsx`, the 17 November 2026 snapshot.
It contains the four reviewers' scores and reasons, the saved Gemini judge
responses, CIDEr scores, analysis scripts, a notebook, and generated results.
Earlier human-rating snapshots, change logs, old-versus-new comparisons, and
the separate post-review adjudication analysis are not included.

## Which scripts to use

1. **refresh_analysis.py** reads the workbook and bundled inputs, verifies
   caption IDs, filenames, and captions, then computes Spearman correlations,
   six pairwise Cohen kappa comparisons, Kendall tau-b comparisons, model
   summaries, and confidence intervals.
2. **prepare_deliverables.py** checks the generated results, independently
   checks kappa against scikit-learn, and writes `results/summary.csv` and
   `results/validation.json`.
3. **agreement_methods.py** provides the kappa and image-bootstrap calculations
   used by the first script. Keep this file beside the other scripts; it does
   not need to be run separately.
4. **BLV Human Evaluation.ipynb** runs these scripts and displays their results.

## Run in VS Code

Extract the entire ZIP first. Open the extracted folder in VS Code, then select
your project Python environment. The package was tested with Python 3.13;
Python 3.12 or later can install the pinned dependencies in `requirements.txt`.

In the terminal, with this extracted folder as the current directory:

```bash
python -m pip install -r requirements.txt
python refresh_analysis.py
python prepare_deliverables.py
```

Alternatively, open **BLV Human Evaluation.ipynb**, select that same Python
environment as the kernel, and run its cells in order. Once dependencies are
installed, the analysis runs offline. No Gemini key, Google Drive mount,
image download, or new model call is required. Running the scripts replaces
only the generated files in this package's `results` folder.

## Inputs

- `BLV_Combined_Human_Ratings.xlsx`: authoritative human scores and reasons;
  500 captions, four reviewers (2,000 EH ratings). It also preserves the
  workbook's review checks. Any helper-script names mentioned inside the
  workbook refer to the team's wider project; use the scripts listed above
  to reproduce this package.
- `inputs/judge_inputs.json`: caption IDs, filenames, captions, image groups,
  and reference text used by the judge.
- `inputs/model_lookup.json`: caption-to-model mapping.
- `inputs/judge_results.jsonl`: 500 saved judge responses and run metadata.
- `inputs/experiment_config.json`: recorded judge model, prompt, settings,
  input/image hashes, and experiment fingerprint. It contains no API key.
- `inputs/cider_scores.csv`: 434 CIDEr scores with caption ID, filename, and
  captioning model. No human-score columns are present in this input.

The workbook is the sole human-score input. The scripts regenerate JSON and
CSV representations from it; editing a generated CSV will not change the
workbook. Canonical caption text and filenames are checked before scores are
combined. The workbook contains formula results; if editing it in a spreadsheet
program, recalculate and save it before rerunning this analysis.

## Analysis rules

- EH and TR scores use the 0–2 rubric. N/A is never converted to zero.
- EH comparisons use the mean of the four human scores for each caption.
- Primary numerical TR comparisons require all four reviewers to mark text
  applicable. This gives 102 captions from 51 images. Seven captions with
  differing applicability judgments remain in the source data but do not
  enter that common TR subset. Spearman additionally requires a numeric judge
  TR score; all 102 eligible captions have one.
- Cohen kappa is calculated separately for all six reviewer pairs, both
  unweighted and **linear-weighted**. TR-score kappa uses the captions that
  both members of each pair mark applicable; applicability kappa is a separate
  Yes/No comparison. The reported team mean is the mean of six pairwise kappas,
  not a single four-rater Cohen kappa. These describe this human-rating
  snapshot, not a reconstruction of first-pass reviewer independence.
- Spearman compares human means with saved judge scores. Kendall tau-b
  compares CIDEr with human means at caption level, pooled and by model.
  These are associations, not accuracy percentages or proof that an LLM can
  replace reviewers. Caption-level Kendall is not a two-model ranking test.
- CIDEr is available for 434 captions from 217 images. The 33 images without
  valid human reference captions are excluded, never assigned zero. The TR
  overlap for Kendall is 100 captions from 50 images.
- Confidence intervals are percentile intervals from 1,000 image-cluster
  bootstrap samples, keeping both captions for an image together. Seeds:
  20261116 for Spearman; 20261117 for Kendall and kappa. Average ranks handle
  Spearman ties; Kendall uses tau-b. Constant-score bootstrap samples are
  excluded, with a minimum of 800 valid samples required for intervals.
- C0104 remains in primary analyses. Its recorded generation flag is handled
  separately in EH sensitivity results that exclude this caption.
- The scripts reuse recorded judge responses and CIDEr scores; they do not
  generate captions, rerun the judge, recompute CIDEr, or label failure types.

## Main results to expect

| Statistic | EH | TR |
|---|---:|---:|
| Spearman rho: judge versus human | 0.633686 | 0.714366 |
| Mean pairwise linear-weighted Cohen kappa | 0.694257 | 0.873457 |
| Kendall tau-b: CIDEr versus human | 0.210937 | 0.075625 |

See `results/summary.csv` for the compact summary and the detailed CSV/JSON
files for sample sizes, model-specific results, pairwise kappas, and intervals.
`results/analysis_report.json` records input hashes and methods.
`package_manifest.json` records hashes of the distributed files; rerunning
analysis under another software environment can change generated metadata.

## Scope

This is an evaluation-data and code package, not the full image dataset.
It does not contain image files, model weights, API credentials, or the
separate post-review sensitivity results. Keep the original project archive
for the research audit trail. Existing historical archives are unchanged.

Human workbook source:
https://docs.google.com/spreadsheets/d/1C49HiFCjuVMTiACyT-5jtqgAtTHDLiv8/edit
