# Web demo: Evaluating AI-Generated Image Captions for Blind and Low-Vision Users
Group #6, Naija Jollof (CSCI 7130, Fall 2026)

A static, screen-reader-friendly website. No server, no framework, no build tools.

## Run it
1. Copy the 250 sample photos into `images/`, keeping their file names
   (e.g. `images/VizWiz_train_00000025.jpg`). They are in the shared Drive under raw_data/train and raw_data/val.
2. Open `index.html` in a browser (double-click), or run `python -m http.server` here and go to http://localhost:8000.
3. To publish: push this folder to GitHub and turn on GitHub Pages for it.

## What it shows
- Results: reviewer agreement (kappa), judge vs humans (Spearman), CIDEr vs humans (Kendall),
  text recognition outcomes, failure taxonomy and honesty by photo group. The two tables are
  computed live from data.js, so they always match the explorer.
- Explorer: every photo with both captions, each reviewer's EH/TR scores, the Gemini judge's
  scores and reasons, the CIDEr score, the failure type, and a Read aloud button.
  Filters: photo group, and "Show" (judge disagrees with reviewers, or any failure type).

## Rebuild the data
`python build_site_data.py` reads data_in/ and writes data.js. Inputs:
- Success.csv, Ebere.csv, Yvonne.csv, Samuel.csv: final review sheets (Review tab as CSV)
- image_references.csv: agreed key text per photo (51 photos)
- judge_results.csv: saved Gemini 3.1 Flash-Lite results (caption_level_comparison.csv)
- cider_scores.csv: saved per-caption CIDEr (matched_caption_scores.csv; 217 photos have references)
- taxonomy_captions.csv: output of failure_taxonomy.py

## Accessibility
Semantic headings and landmarks, skip link, full keyboard use, live-region announcements when the
photo changes, every score written as text with a spoken label (never colour alone), a colour pair
for the two models that stays distinct for common colour-vision deficiencies, dark mode, and text
that scales with browser zoom. Test with VoiceOver (Cmd+F5 on Mac) and NVDA on Windows before presenting.
