# BLV Caption Evaluation

AI final project: do BLIP-2 and LLaVA-1.5 captions stay honest on degraded photos
(Error Honesty) and read the text in them (Text Recognition)? And does CIDEr rank the
models the same way our rubric does?

**Team:** Success, Yvonne, Ebere, Samuel

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
