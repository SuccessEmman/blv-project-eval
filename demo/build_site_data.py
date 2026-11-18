# build_site_data.py
# BLV caption study (Group #6, Naija Jollof) - builds data.js, the one file the website reads.
#
# Inputs (all in data_in/):
#   Success.csv Ebere.csv Yvonne.csv Samuel.csv   final review sheets (Review tab saved as CSV)
#   image_references.csv                          agreed reference text per photo (51 photos)
#   judge_results.csv                             saved Gemini 3.1 Flash-Lite judge results
#   cider_scores.csv                              saved per-caption CIDEr (217 photos have references)
#   taxonomy_captions.csv                         output of failure_taxonomy.py
#
#   python build_site_data.py           -> writes data.js
# Photos are not bundled: copy the 250 sample photos into images/ (file names unchanged).
import json
import pandas as pd

RATERS = ["Success", "Ebere", "Yvonne", "Samuel"]
num = lambda v: None if pd.isna(v) else int(v)

sheets = {r: pd.read_csv(f"data_in/{r}.csv", dtype=str, keep_default_na=False).set_index("caption_id") for r in RATERS}
base = sheets["Success"]
refs = pd.read_csv("data_in/image_references.csv", dtype=str, keep_default_na=False).set_index("file_name")
judge = pd.read_csv("data_in/judge_results.csv").set_index("caption_id")
cider = pd.read_csv("data_in/cider_scores.csv").set_index("caption_id")
tax = pd.read_csv("data_in/taxonomy_captions.csv", keep_default_na=False).set_index("caption_id")

photos = {}
for cid in base.index:
    f = base.loc[cid, "file_name"]
    t = tax.loc[cid]
    p = photos.setdefault(f, {"file_name": f, "stratum": t["stratum"], "image": f"images/{f}",
                              "expected_text": refs.loc[f, "expected_text"] or None, "captions": []})
    eh = [num(pd.to_numeric(sheets[r].loc[cid, "eh_score"])) for r in RATERS]
    tr = [num(pd.to_numeric(sheets[r].loc[cid, "tr_score"], errors="coerce")) for r in RATERS]
    j = judge.loc[cid]
    p["captions"].append({
        "id": cid,
        "model": t["model"],
        "caption": base.loc[cid, "caption"],
        "human": {"eh": eh, "tr": tr if all(v is not None for v in tr) else None},
        "judge": {"eh": num(j.judge_eh), "tr": num(j.judge_tr),
                  "eh_reason": j.judge_eh_reason if isinstance(j.judge_eh_reason, str) else "",
                  "tr_reason": j.judge_tr_reason if isinstance(j.judge_tr_reason, str) else ""},
        "cider": None if cid not in cider.index else round(float(cider.loc[cid, "cider"]), 3),
        "failure": t["primary_label"] or None,
        "tr_outcome": t["tr_class"] or None,
        "cut_off": bool(t["cut_off"]),
    })

order = {"clear": 0, "blurry": 1, "low_light": 2, "text": 3}
data = sorted(photos.values(), key=lambda p: (order[p["stratum"]], p["file_name"]))
for p in data:
    p["captions"].sort(key=lambda c: c["model"])
with open("data.js", "w", encoding="utf-8") as fh:
    fh.write("window.DEMO_DATA = ")
    json.dump({"photos": data, "raters": len(RATERS)}, fh, ensure_ascii=False)
    fh.write(";\n")
caps = [c for p in data for c in p["captions"]]
print(f"wrote data.js: {len(data)} photos, {len(caps)} captions, "
      f"{sum(c['human']['tr'] is not None for c in caps)} with TR from all 4, "
      f"{sum(c['cider'] is not None for c in caps)} with CIDEr")
