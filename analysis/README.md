# Analysis entry points

Use **blv_current_evaluation** for the reported human-score comparisons.
It contains the final combined workbook, frozen judge responses and CIDEr
inputs, all required scripts, and the generated results. The folder name has
no date. Earlier human-score snapshots are not included in this package.

- [Run the final analysis](blv_current_evaluation/README.md): Spearman,
  six-pair Cohen kappa (unweighted and linear-weighted), Kendall tau-b,
  image-cluster confidence intervals and model summaries.
- [Run the judge](blv_judge_run/README.md): only needed for new Gemini responses.
  The study's 500 saved responses are already in the final-analysis inputs.
- [Explore failure categories](failure_taxonomy_exploratory/README.txt):
  optional keyword-rule analysis, not fully validated human coding.

Ebere's original local `human_agreement`, `kendall_analysis`, and
`updated_evaluation_20261117` folders are development versions. Their maintained
statistical functionality is consolidated in `blv_current_evaluation`; do not
combine older result files with its human-rating workbook.

The full [run guide](../docs/DRIVE_DELIVERABLES_RUN_GUIDE.md) documents the
earlier stages and links to images, annotations and review materials.
