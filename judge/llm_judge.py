# llm_judge.py
# BLV captioning project - LLM judge
#
# What this does:
#   Gives Gemini a photo + a caption and asks it to grade the caption with our
#   rubric (Error Honesty 0-2, Text Recognition 0-2). Saves every score to a CSV
#   so we can compare Gemini's grades with ours later (Spearman's rho, section IV.E).
#
# Before you run it:
#   1. Get a free Gemini key: https://aistudio.google.com/apikey
#   2. pip install google-genai
#   3. In terminal:  export GEMINI_API_KEY="your key here"
#      (don't paste your key into this file or the group chat!)
#
# How to run it:
#   Quick test with fake scores, doesn't call Gemini at all:
#       python llm_judge.py --captions captions.csv --images sample/images --mock
#
#   Test on 5 real photos (uses the VizWiz human captions since we don't
#   have BLIP-2/LLaVA captions yet):
#       python llm_judge.py --vizwiz-json val.json --images images --limit 5
#
#   The real run once Samuel and Yvonne send the captions:
#       python llm_judge.py --captions captions.csv --sample sample_manifest.csv --images sample/images
#
# If it stops halfway (wifi drops, Gemini says slow down, laptop sleeps), just
# run the exact same command again. It skips whatever is already in the CSV.

import argparse
import csv
import json
import os
import random
import sys
import time


# This is the prompt Gemini gets for every caption.
# The score descriptions are copied from our proposal (section IV.C) so the
# judge is grading with the same rubric we use. If we change the rubric
# anchors as a team, UPDATE THIS TOO or the comparison won't be fair.
# {caption} and {text_note} get filled in for each image further down.
RUBRIC_PROMPT = """You are grading image captions written for blind and low-vision users.
These photos were taken by blind people, so a lot of them are blurry, dark, or badly framed.
Look at the image, then grade the caption below.

Caption: "{caption}"
{text_note}
ERROR HONESTY (EH) - does the caption avoid making things up?
0 = it confidently describes things that are wrong or can't actually be seen
1 = it hedges a little, or it's mostly right but has at least one confident wrong detail
2 = it clearly says what it can't make out (like "too blurry to tell"), or it describes the image correctly
Grade EH for every image. A correct caption on a clear photo gets a 2.

TEXT RECOGNITION (TR) - does the caption read the text in the photo?
First decide if there's readable text that matters (a label, sign, document, screen, packaging).
If there's no readable text: text_present = false and tr_score = null.
If there is text:
0 = the caption ignores it
1 = it says there's text but doesn't read it (or reads it mostly wrong)
2 = it reads the important text correctly (small typos are ok)

Reply with only this JSON and nothing else:
{{"eh_score": 0, "eh_reason": "one sentence", "text_present": true, "tr_score": 0, "tr_reason": "one sentence"}}
"""

# For the text images, Samuel and Ebere are writing down what the text actually says.
# If that's in the sample CSV, we hand it to Gemini so it can check the caption's spelling
# against the real thing instead of guessing.
TEXT_NOTE = '\nNote: a person on our team checked this photo. The important text in it is: "{text}"\n'

# Columns in the output CSV, in this order.
COLUMNS = ["file_name", "model", "stratum", "caption",
           "eh_score", "eh_reason", "text_present", "tr_score", "tr_reason",
           "judge_model", "error"]


def read_captions(path):
    # Reads the captions CSV from Samuel/Yvonne.
    # Needs at least these columns: file_name, model, caption
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # catch a wrong column name now instead of 2 hours into the run
    for col in ["file_name", "model", "caption"]:
        if rows and col not in rows[0]:
            sys.exit(f"captions file is missing a '{col}' column. columns I found: {list(rows[0])}")
    return rows


def captions_from_vizwiz(json_path, image_folder):
    # Test mode only.
    # Instead of model captions, take the first human caption VizWiz has for each photo.
    # Only uses photos that are actually sitting in image_folder.
    data = json.load(open(json_path, encoding="utf-8"))
    photos_we_have = set(os.listdir(image_folder))

    # VizWiz links captions to photos by an id number, so build id -> file name first
    id_to_file = {}
    for img in data["images"]:
        if img["file_name"] in photos_we_have:
            id_to_file[img["id"]] = img["file_name"]

    rows = []
    used = set()
    for cap in data["annotations"]:
        file_name = id_to_file.get(cap["image_id"])
        # skip photos we don't have, photos we already took a caption for,
        # and captions VizWiz marked as rejected
        if not file_name or file_name in used or cap.get("is_rejected"):
            continue
        used.add(file_name)
        rows.append({"file_name": file_name, "model": "vizwiz_human", "caption": cap["caption"]})
    return rows


def read_sample(path):
    # The sample manifest tells us each photo's group (clear/blurry/low_light/text)
    # and the expected text for the text photos. Optional, the script works without it.
    # Returns a dict so we can look photos up by file name: sample["VizWiz_val_...jpg"]
    if not path:
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        return {row["file_name"]: row for row in csv.DictReader(f)}


def already_scored(out_path):
    # This is what makes the script resumable.
    # Looks at the output CSV and returns every (photo, model) pair that already has a score,
    # so we don't pay (in time) for them twice. Rows that errored out get retried.
    if not os.path.exists(out_path):
        return set()
    done = set()
    with open(out_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if not row["error"]:
                done.add((row["file_name"], row["model"]))
    return done


def make_prompt(caption, expected_text):
    # Fills the caption (and the expected text, if we have it) into the rubric prompt.
    # Swapping " for ' so a quote inside a caption doesn't mess up the prompt.
    note = TEXT_NOTE.format(text=expected_text) if expected_text else ""
    return RUBRIC_PROMPT.format(caption=caption.replace('"', "'"), text_note=note)


def read_gemini_reply(text):
    # Gemini is told to send back only JSON, but sometimes it wraps it in ```json ... ```
    # so grab everything from the first { to the last } and parse that.
    start = text.find("{")
    end = text.rfind("}")
    return json.loads(text[start:end + 1])


def check_scores(reply):
    # Make sure Gemini gave us real scores (0, 1 or 2) before we save anything.
    # If it sends something weird like a 3, this throws an error and we retry.
    def valid(score):
        if score is None or score == "null":
            return ""
        score = int(score)
        if score not in (0, 1, 2):
            raise ValueError(f"got a score of {score}, should be 0, 1 or 2")
        return score

    has_text = reply.get("text_present") in (True, "true", "True")
    return {
        "eh_score": valid(reply.get("eh_score")),
        "eh_reason": reply.get("eh_reason", ""),
        "text_present": has_text,
        # TR only counts when there's text in the photo, otherwise leave it blank
        "tr_score": valid(reply.get("tr_score")) if has_text else "",
        "tr_reason": reply.get("tr_reason", "") if has_text else "",
    }


def connect_to_gemini(model_name):
    # Sets up the Gemini connection and hands back a function we can call
    # with (photo bytes, prompt) to get Gemini's grade.
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        sys.exit("google-genai isn't installed. run:  pip install google-genai")

    if not os.environ.get("GEMINI_API_KEY"):
        sys.exit("no API key found. run:  export GEMINI_API_KEY=\"your key\"  first")

    client = genai.Client()  # picks up GEMINI_API_KEY automatically

    # temperature 0 = less randomness, so the same caption gets the same grade if we rerun it.
    # response_mime_type asks Gemini to reply in JSON.
    settings = types.GenerateContentConfig(temperature=0, response_mime_type="application/json")

    def ask_gemini(photo_bytes, prompt):
        photo = types.Part.from_bytes(data=photo_bytes, mime_type="image/jpeg")
        reply = client.models.generate_content(model=model_name, contents=[photo, prompt], config=settings)
        return read_gemini_reply(reply.text)

    return ask_gemini


def fake_gemini(photo_bytes, prompt):
    # Random scores for --mock mode. Lets us test the script (and build the web demo)
    # without using up our free Gemini requests. NEVER use these numbers in the report.
    has_text = random.random() < 0.5
    return {"eh_score": random.choice([0, 1, 2]), "eh_reason": "MOCK",
            "text_present": has_text,
            "tr_score": random.choice([0, 1, 2]) if has_text else None,
            "tr_reason": "MOCK" if has_text else ""}


def grade_with_retries(ask, photo_bytes, prompt, tries=6):
    # The free tier only allows ~5 requests a minute. If we go over, Gemini says
    # "429 / RESOURCE_EXHAUSTED". When that happens we wait and try again,
    # doubling the wait each time (20s, 40s, 80s...) up to 5 min.
    wait = 20
    for attempt in range(1, tries + 1):
        try:
            return check_scores(ask(photo_bytes, prompt))
        except Exception as e:
            if attempt == tries:
                raise  # gave it 6 shots, give up on this one and move on

            msg = str(e)
            if any(code in msg for code in ["429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE"]) or "quota" in msg.lower():
                print(f"    gemini says slow down, waiting {wait}s...")
                time.sleep(wait)
                wait = min(wait * 2, 300)
            else:
                # some other hiccup (bad JSON, network blip), short pause then retry
                print(f"    error: {msg[:100]} - trying again")
                time.sleep(5)


def main():
    # ---- command line options (see the examples at the top) ----
    parser = argparse.ArgumentParser(description="Grade captions with Gemini using our EH/TR rubric")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--captions", help="CSV with file_name, model, caption")
    source.add_argument("--vizwiz-json", help="test mode: grade VizWiz human captions instead")
    parser.add_argument("--images", required=True, help="folder with the photos")
    parser.add_argument("--sample", help="sample manifest CSV (adds stratum + expected_text)")
    parser.add_argument("--out", default="judge_scores.csv", help="where to save the scores")
    parser.add_argument("--model", default="gemini-flash-latest", help="which Gemini model to use")
    parser.add_argument("--delay", type=float, default=13,
                        help="seconds to wait between photos. 13s keeps us under the free limit")
    parser.add_argument("--limit", type=int, help="only do the first N (for testing)")
    parser.add_argument("--mock", action="store_true", help="fake scores, doesn't call Gemini")
    args = parser.parse_args()

    # ---- load everything ----
    if args.captions:
        rows = read_captions(args.captions)
    else:
        rows = captions_from_vizwiz(args.vizwiz_json, args.images)

    sample = read_sample(args.sample)
    if args.limit:
        rows = rows[:args.limit]

    # skip anything we already graded on an earlier run
    done = already_scored(args.out)
    to_do = [r for r in rows if (r["file_name"], r["model"]) not in done]
    print(f"{len(rows)} captions, {len(done)} already done, {len(to_do)} left")
    if not to_do:
        return

    # ---- pick real Gemini or fake ----
    if args.mock:
        ask, judge_name, delay = fake_gemini, "mock", 0
    else:
        ask, judge_name, delay = connect_to_gemini(args.model), args.model, args.delay
        print(f"this should take about {len(to_do) * delay / 60:.0f} min")

    # ---- grade each caption ----
    # Opening in "a" (append) mode so a rerun adds to the file instead of wiping it.
    first_time = not os.path.exists(args.out)
    with open(args.out, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if first_time:
            writer.writeheader()

        for i, row in enumerate(to_do, start=1):
            file_name = row["file_name"]
            info = sample.get(file_name, {})  # group + expected text, if we have the manifest

            result = {"file_name": file_name, "model": row["model"], "caption": row["caption"],
                      "stratum": info.get("stratum", ""), "judge_model": judge_name, "error": ""}

            try:
                with open(os.path.join(args.images, file_name), "rb") as img:
                    photo_bytes = img.read()
                prompt = make_prompt(row["caption"], info.get("expected_text", "").strip())
                result.update(grade_with_retries(ask, photo_bytes, prompt))
                print(f"[{i}/{len(to_do)}] {file_name} ({row['model']}): EH={result['eh_score']}  TR={result['tr_score'] or '-'}")
            except Exception as e:
                # don't crash the whole run over one photo. log it and keep going,
                # it'll get retried next time we run the script
                result["error"] = str(e)[:300]
                print(f"[{i}/{len(to_do)}] {file_name} ({row['model']}): FAILED - {result['error'][:80]}")

            writer.writerow(result)
            f.flush()  # save to disk right away so nothing is lost if it crashes

            if i < len(to_do):
                time.sleep(delay)  # stay under the free tier limit

    print(f"\ndone! scores saved to {args.out}")
    print("any row with something in the 'error' column will be retried next time you run this")


if __name__ == "__main__":
    main()
