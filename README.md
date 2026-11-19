# BLV Caption Evaluation

AI final project: do BLIP-2 and LLaVA-1.5 captions stay honest on degraded photos
(Error Honesty) and read the text in them (Text Recognition)? And does CIDEr rank the
models the same way our rubric does?

**Team:** Success, Yvonne, Ebere, Samuel

## Start here: final human-score analysis

The complete reproducible analysis is in
[`analysis/blv_current_evaluation`](analysis/blv_current_evaluation/README.md).
It includes the human-rating workbook, saved Gemini responses, CIDEr inputs,
scripts, notebook and validated results. No API key or images are needed to
reproduce the statistics.

```bash
cd analysis/blv_current_evaluation
python -m pip install -r requirements.txt
python refresh_analysis.py
python prepare_deliverables.py
```

See [`results/summary.csv`](analysis/blv_current_evaluation/results/summary.csv)
and [`results/validation.json`](analysis/blv_current_evaluation/results/validation.json).
The primary judge–human Spearman correlations are EH **0.633686** and TR
**0.714366**, using the packaged human scores. N/A is never scored as zero.

| Work | Location |
| --- | --- |
| Image selection, text review and blur pilot | [`sampling/`](sampling/README.md) |
| Original caption generations and settings | [`captioning/outputs/generated_captions/`](captioning/outputs/generated_captions/) |
| Gemini 3.1 Flash-Lite experiment runner | [`analysis/blv_judge_run/`](analysis/blv_judge_run/README.md) |
| Human agreement, Spearman, Kendall and model summaries | [`analysis/blv_current_evaluation/`](analysis/blv_current_evaluation/README.md) |
| Optional exploratory failure taxonomy | [`analysis/failure_taxonomy_exploratory/`](analysis/failure_taxonomy_exploratory/README.txt) |
| Earlier stages and shared data locations | [`docs/DRIVE_DELIVERABLES_RUN_GUIDE.md`](docs/DRIVE_DELIVERABLES_RUN_GUIDE.md) |

`judge/llm_judge.py` is the team's original judge implementation. The adapted
runner above implements the experiment rubric and model; its obsolete
`analyze()` entry point is disabled so that human comparisons use the final
workbook. The sampling-history notebook retains historical paths and manual
review steps; consult the run guide before replaying it.

The report, poster and presentation remain in the team's
[Drive deliverables](https://drive.google.com/drive/folders/19oK13yRaaIY4WaRHa359wiKn8d1-uusU).

## Folders

| Folder        | Contents                                      | Owner         |
| ------------- | --------------------------------------------- | ------------- |
| `sampling/`   | 250-image sample, text review, blur scores    | Ebere, Samuel |
| `captioning/` | BLIP-2 + LLaVA notebooks, captions CSV        | Yvonne        |
| `judge/`      | LLM-as-judge (EH/TR scoring)                  | Success       |
| `analysis/`   | Cohen's kappa, Spearman, CIDEr, Kendall's tau | Samuel, Ebere |
| `demo/`       | Accessible web demo                           | Success       |
| `report/`     | Report, slides, poster                        | Everyone      |

Images and VizWiz data live in the shared Google Drive, not in this repo.

## Shared file formats

- Captions: `file_name, model, caption`
- Sample manifest: `file_name, stratum, expected_text`

## How we work

1. `git pull` before you start
2. Work on your own branch: `yourname/topic`
3. Push, then open a pull request into `main`
4. Never commit API keys, `.env` files, or images
