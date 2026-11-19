# build_site_data.py
# Puts all our results into one file, data.js, which is the only data the website loads.
# Run it again whenever one of the score files changes:   python build_site_data.py
#
# What it reads (all saved as CSV in data_in/):
#   Success.csv, Ebere.csv, Yvonne.csv, Samuel.csv   our 4 final review sheets
#   image_references.csv                             agreed key text for the 51 text photos
#   judge_results.csv                                Gemini 3.1 Flash-Lite scores and reasons
#   cider_scores.csv                                 CIDEr for each caption (217 photos have references)
#   taxonomy_captions.csv                            photo group, model and failure type per caption
#
# The photos aren't inside data.js. The site loads them from images/ (see copy_images.py).
import json
import pandas as pd

# reviewer order, the same order the website uses
RATERS = ["Success", "Ebere", "Yvonne", "Samuel"]
# turns a cell into a whole number, or None if it's empty (like TR on a non-text photo)
num = lambda v: None if pd.isna(v) else int(v)

# load every file and index it by caption_id (file_name for the references).
# We match everything by ID, never by row order, because sorting a sheet moves rows
# around and would attach scores to the wrong caption.
# dtype=str and keep_default_na=False keep cells exactly as typed, so "N/A" stays text.
sheets = {r: pd.read_csv(f"data_in/{r}.csv", dtype=str, keep_default_na=False).set_index("caption_id") for r in RATERS}
# any sheet works as the list of all 500 captions, we use the first one
base = sheets["Success"]
refs = pd.read_csv("data_in/image_references.csv", dtype=str, keep_default_na=False).set_index("file_name")
judge = pd.read_csv("data_in/judge_results.csv").set_index("caption_id")
cider = pd.read_csv("data_in/cider_scores.csv").set_index("caption_id")
tax = pd.read_csv("data_in/taxonomy_captions.csv", keep_default_na=False).set_index("caption_id")

# group the 500 captions under their 250 photos
photos = {}
for cid in base.index:
    f = base.loc[cid, "file_name"]
    t = tax.loc[cid]
# the first time we see a photo, create its entry. After that, reuse it.
    p = photos.setdefault(f, {"file_name": f, "stratum": t["stratum"], "image": f"images/{f}",
                              "expected_text": refs.loc[f, "expected_text"] or None, "captions": []})
# each reviewer's Error Honesty and Text Recognition score for this caption.
# errors="coerce" turns N/A into empty, so captions on non-text photos get None.
    eh = [num(pd.to_numeric(sheets[r].loc[cid, "eh_score"])) for r in RATERS]
    tr = [num(pd.to_numeric(sheets[r].loc[cid, "tr_score"], errors="coerce")) for r in RATERS]
    j = judge.loc[cid]
    p["captions"].append({
        "id": cid,
        "model": t["model"],
        "caption": base.loc[cid, "caption"],
# TR only counts when all 4 of us scored it (the 102 captions on text photos)
        "human": {"eh": eh, "tr": tr if all(v is not None for v in tr) else None},
        "judge": {"eh": num(j.judge_eh), "tr": num(j.judge_tr),
                  "eh_reason": j.judge_eh_reason if isinstance(j.judge_eh_reason, str) else "",
                  "tr_reason": j.judge_tr_reason if isinstance(j.judge_tr_reason, str) else ""},
# 33 photos have no CIDEr (no usable reference captions), so those stay None
        "cider": None if cid not in cider.index else round(float(cider.loc[cid, "cider"]), 3),
        "failure": t["primary_label"] or None,
        "tr_outcome": t["tr_class"] or None,
        "cut_off": bool(t["cut_off"]),
    })

# sort photos by group (clear, blurry, low light, text), then by file name,
# and put BLIP-2 before LLaVA in each photo, so the explorer order never changes
order = {"clear": 0, "blurry": 1, "low_light": 2, "text": 3}
data = sorted(photos.values(), key=lambda p: (order[p["stratum"]], p["file_name"]))
for p in data:
    p["captions"].sort(key=lambda c: c["model"])
# save it as a JavaScript file (window.DEMO_DATA = {...}) instead of plain JSON.
# Browsers block loading a .json file when you open index.html straight from the folder,
# but they allow a normal <script>, so the site works with or without a server.
with open("data.js", "w", encoding="utf-8") as fh:
    fh.write("window.DEMO_DATA = ")
    json.dump({"photos": data, "raters": len(RATERS)}, fh, ensure_ascii=False)
    fh.write(";\n")
# quick check at the end. It should say 250 photos, 500 captions,
# 102 with TR from all 4, and 434 with CIDEr.
caps = [c for p in data for c in p["captions"]]
print(f"wrote data.js: {len(data)} photos, {len(caps)} captions, "
      f"{sum(c['human']['tr'] is not None for c in caps)} with TR from all 4, "
      f"{sum(c['cider'] is not None for c in caps)} with CIDEr")
