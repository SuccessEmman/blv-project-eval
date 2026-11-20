FAILURE TAXONOMY — EXPLORATORY
BLV caption evaluation, Group #6 Naija Jollof

STATUS AND SCOPE
This folder contains an exploratory automated, rule-based classification of caption failures. It is NOT fully validated human coding. The rules use caption text, human written reasons and, for some categories, saved Gemini EH explanations. They do not inspect the images directly and are not independent of the judge. No API calls are made by failure_taxonomy.py.

Use Final Statistical Analysis and its BLV_Combined_Human_Ratings.xlsx for the main EH/TR, agreement, Spearman and Kendall results. Taxonomy labels do not change those scores. This optional analysis is not required to reproduce the main results.

FILES
failure_taxonomy.py     Rule-based classifier.
taxonomy_captions.csv   One row per caption: model, stratum, scores, primary and matching labels, TR outcome and evidence.
taxonomy_counts.csv     Exploratory category counts by stratum and model.
taxonomy.png            Saved chart; interpret under the limitations below.
taxonomy_spot_check.xlsx Review worksheet for a 30-caption spot check. Its presence alone does not establish completed validation or validation of all 500 captions.

HOW TO RUN
Install pandas in your Python environment. Run in a separate working directory: the script writes taxonomy_captions.csv and taxonomy_counts.csv into the current directory and can overwrite files with those names.

  python failure_taxonomy.py <reviewer_csv_folder> combined_500_captions.csv caption_level_comparison.csv

Use quoted absolute paths if the script or inputs are elsewhere or their paths contain spaces. Required inputs:
1. A folder containing Success.csv, Ebere.csv, Yvonne.csv and Samuel.csv, exported from the intended human-score source. Required columns used by this script: caption_id, file_name, caption, eh_score, eh_reason, tr_score, tr_reason. Keep N/A non-numeric; never turn it into zero.
2. A caption CSV containing file_name, caption, model and stratum. The script matches filename and normalized caption text to recover model/group metadata.
3. A judge-comparison CSV containing caption_id and judge_eh_reason. The filename is only an example: the required columns must actually be present. Do not pass raw judge_results.jsonl or rename a file to simulate the required schema.

Before running, verify unique and matching caption IDs, filename-caption pairs and the intended human scores across all four exports. The script uses Success's caption IDs as its row list and does not perform the comprehensive validation in Final Statistical Analysis. Do not silently mix earlier reviewer exports with the main workbook.

The script regenerates the two CSV files, not the chart or spot-check workbook. Document the source files used for any rerun and keep new outputs separate from saved deliverables until checked.

WHAT THE EXISTING RULES MEAN
- EH majority failure: at least three of four reviewers gave EH 0. This is a majority, not necessarily unanimous agreement. The saved run has 255 such captions.
- Contested: exactly two reviewers gave EH 0; reported separately, not given an EH0 failure category. The saved run has 59 such captions.
- The label "Vague or over-hedged (3+ of 4 gave 1)" is assigned automatically to 25 captions with at least three EH1 ratings. It does NOT prove vagueness: the team's rubric also assigns EH1 to colour-only errors. Interpret these as majority-EH1 cases pending inspection.
- Primary label is the first matching keyword rule, in order: degenerate output; confident description of an unreadable photo; wrong text/brand/label; invented person/body part; invented setting/furniture; wrong colour/count/position; wrong or invented main object. Unmatched majority-EH0 cases receive "Other unsupported claim". Multiple matches are retained separately.
- TR categories use rows with four numerical TR scores (102 in the saved run): mean 2 -> "Text read correctly"; mean below 0.5 -> "Text ignored"; matching misreading keywords -> "Text misread"; otherwise -> "Text partly read". These are rule-generated category names, not independently confirmed observations. The script uses numeric-score availability and does not separately validate applicability flags.

IMPORTANT LIMITATIONS FOR REPORTING
1. Keyword matches can misinterpret explanations, negation and overlapping errors. Priority order influences the primary category.
2. Gemini reasons help assign some categories. These labels cannot serve as independent human validation of the same judge.
3. The EH1 category conflates rubric reasons, including colour-only errors; do not present it as a validated count of over-hedging.
4. The cut_off column checks missing terminal punctuation in LLaVA captions. The script's printed statement about a "64-token limit" is stronger than its test supports. Neither that column nor the message proves truncation at the generation limit.
5. "All photos" in taxonomy_counts.csv counts CAPTIONS, not unique images. Two model captions can refer to the same image. EH primary-label totals also include the majority-EH1 category, not just majority-EH0 failures.
6. A mean TR score below 0.5 does not require all four reviewers to give zero. The "ignored", "misread" and "partly read" labels require image/caption inspection before being asserted as validated failure types.
7. A 30-caption spot check, even when completed, does not validate every label in a 500-caption dataset. Full validation would require documented manual review and resolution of disagreements.

SUGGESTED REPORT WORDING
"We conducted an exploratory automated failure taxonomy using keyword rules over captions and written human and judge explanations. Categories and counts are provisional and were not established by fully validated human coding. They supplement, rather than replace, the rubric-based EH/TR analyses."

This documentation labels the existing method and its limitations. It does not change the classifier, scores, saved CSVs, chart or review worksheet.
