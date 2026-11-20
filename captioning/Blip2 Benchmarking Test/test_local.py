"""Offline smoke tests; run `python -m unittest test_local.py`."""
import argparse
import json
import tempfile
import unittest
from pathlib import Path
from blip2_karpathy_local import (parse_args, normalize_refs, candidate_urls,
    load_existing_predictions, validate_resume_config)


class LocalBenchmarkTests(unittest.TestCase):
    def test_cli_defaults_and_finetuned(self):
        opts = parse_args([])
        self.assertEqual(opts.num_images, 5)
        self.assertEqual(opts.model, "Salesforce/blip2-flan-t5-xl")
        opts = parse_args(["--experiment", "coco_finetuned_reference", "--num-images", "50"])
        self.assertEqual(opts.prompt, "a photo of")
        self.assertEqual(opts.num_images, 50)

    def test_normalize_refs(self):
        self.assertEqual(normalize_refs([{"raw": f"sentence {i}"} for i in range(5)])[0], "sentence 0")
        with self.assertRaises(ValueError):
            normalize_refs(["short"])

    def test_urls_stay_https(self):
        urls = candidate_urls({"filename": "COCO_val2014_000000000001.jpg", "filepath": "val2014",
                               "url": "http://images.cocodataset.org/val2014/COCO_val2014_000000000001.jpg"})
        self.assertTrue(urls)
        self.assertTrue(all(url.startswith("https://") for url in urls))

    def test_incremental_predictions_and_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "predictions.jsonl"
            path.write_text(json.dumps({"image_id": 1, "caption": "a cat"}) + "\n", encoding="utf-8")
            self.assertEqual(load_existing_predictions(path, {1, 2}), {1: "a cat"})
            config = root / "run_config.json"
            current = {"dataset": "demo", "num_images": 5}
            validate_resume_config(config, current)
            self.assertTrue(config.exists())

if __name__ == "__main__":
    unittest.main()
