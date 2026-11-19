# Gemini judge — experiment runner

This is the adapted version of the team's original `judge/llm_judge.py`.
It uses **gemini-3.1-flash-lite**, rubric v1.2, and the team's colour-only EH1
rule. The model receives the original image, generated caption, rubric and
available reference text. Human scores, human reasons, reviewer names and
captioning-model identities are not sent to Gemini.

## Reproduce the reported comparisons

The 500 responses are already saved in
[`../blv_current_evaluation/inputs/`](../blv_current_evaluation/inputs/).
Use [BLV Human Evaluation.ipynb](../blv_current_evaluation/BLV%20Human%20Evaluation.ipynb)
or that package's scripts. No Gemini calls are needed. `judge_run.analyze()`
is deliberately disabled because its earlier implementation read a separate
human-rating snapshot.

## Generate new responses

1. Install this folder's `requirements.txt` in your Python environment.
2. Open `BLV Gemini Judge.ipynb` in VS Code and select that environment.
3. Set `IMAGE_ROOT` to your downloaded VizWiz image folder. The example path
   points to Ebere's original `Desktop/blv-project/raw_data` directory.
4. Run preflight, then enter your Gemini key through the hidden prompt or set
   `GEMINI_API_KEY` in your environment. Never save a literal key in a cell.
5. Run the eight-caption technical check, then the full-run cell. Rerunning
   that cell resumes from saved results. The eight responses count toward
   the complete run; they are not a separate rubric-calibration dataset.

Responses are saved in this folder's ignored `results/` directory. Do not
replace the frozen study responses with a fresh run: model outputs can vary.
The runner records input/image hashes, settings and response metadata, and
rejects mixed configurations. Quota/access errors stop the run while keeping
completed responses. Temperature 0 reduces variation but does not guarantee
identical responses.

## Changes from the original script

- Expanded EH/TR prompt to match the agreed rubric; substantive unsupported
  claims get EH0, isolated colour errors EH1, and N/A TR stays null.
- Structured responses and stricter score/reason validation.
- Checks for 500 unique captions and 250 readable images, with two captions
  per image; stable caption IDs track progress.
- Recorded experiment fingerprints, fixed request order, and response metadata.
- Revised timeout, pacing and stopping behavior; saving/resuming/retries were
  already features of the original and were strengthened here.
- Human comparisons are maintained in the separate final-analysis package.

Reference-text assistance also existed in the original script. Describe this
as a reference-assisted judge, not an unassisted OCR experiment.

Offline runner checks (no API calls):

```bash
python -m unittest discover -s analysis/blv_judge_run -p 'test_*.py'
```
