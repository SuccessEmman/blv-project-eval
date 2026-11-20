#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Any

PRETRAINED = "Salesforce/blip2-flan-t5-xl"
COCO_FINETUNED = "Salesforce/blip2-flan-t5-xl-coco"
PROJECT_PROMPT = "Describe this image."
FINETUNED_PROMPT = "a photo of"
DATASET = "yerevann/coco-karpathy"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run BLIP-2 on MS COCO Karpathy test (5k) from Windows/macOS/Linux.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--experiment", choices=["project_pretrained", "coco_finetuned_reference"],
                        default="project_pretrained")
    parser.add_argument("--model", default=None, help="Explicit checkpoint override")
    parser.add_argument("--prompt", default=None, help="Explicit prompt override")
    parser.add_argument("--num-images", type=int, default=5000,
                        help="5000 = complete Karpathy test set; use 5 or 50 for a quick check")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--output-root", type=Path, default=Path("results"))
    parser.add_argument("--cache-dir", type=Path, default=Path("image_cache_val2014"))
    parser.add_argument("--local-image-dir", type=Path, default=None,
                        help="Optional extracted COCO val2014 image directory (avoids web image downloads)")
    parser.add_argument("--score-only", action="store_true",
                        help="Score already-generated predictions, without a GPU/model")
    parser.add_argument("--download-only", action="store_true",
                        help="Download/check source images without loading BLIP-2")
    parser.add_argument("--metrics", choices=["cider", "all"], default="cider",
                        help="CIDEr alone or BLEU/CIDEr/METEOR/SPICE (Java needed for all)")
    parser.add_argument("--check-setup", action="store_true",
                        help="Check Python/GPU/Java and exit; no dataset or model download")
    parser.add_argument("--java", type=Path, default=None,
                        help="Path to java(.exe) or a Java folder (normally found or downloaded automatically)")
    parser.add_argument("--no-java-download", action="store_true",
                        help="Never download a portable Java automatically")
    parser.add_argument("--tokenizer", choices=["auto", "ptb", "python"], default="auto",
                        help="auto = official Java tokenizer if Java is found, else the Python one; "
                             "ptb = official only (stop if no Java); python = never use Java")
    args = parser.parse_args(argv)
    if args.tokenizer == "python" and args.metrics == "all":
        parser.error("--metrics all needs Java (METEOR/SPICE); use --tokenizer ptb")
    if not 1 <= args.num_images <= 5000:
        parser.error("--num-images must be between 1 and 5000")
    if args.num_beams < 1 or args.max_new_tokens < 1:
        parser.error("--num-beams and --max-new-tokens must be >= 1")
    if args.score_only and args.download_only:
        parser.error("--score-only and --download-only are mutually exclusive")
    if args.model is None:
        args.model = PRETRAINED if args.experiment == "project_pretrained" else COCO_FINETUNED
    if args.prompt is None:
        args.prompt = PROJECT_PROMPT if args.experiment == "project_pretrained" else FINETUNED_PROMPT
    args.output_root = args.output_root.expanduser().resolve()
    args.cache_dir = args.cache_dir.expanduser().resolve()
    if args.local_image_dir:
        args.local_image_dir = args.local_image_dir.expanduser().resolve()
    args.output_dir = args.output_root / f"{args.experiment}_{args.num_images}_seed{args.seed}"
    return args


JAVA_FEATURE_VERSION = 21
ADOPTIUM_API = "https://api.adoptium.net/v3"
JAVA_DOWNLOAD_DIR = Path(__file__).resolve().parent  # portable Java lives next to this script

JAVA_HELP = (
    "Java is unavailable, and the automatic download did not work (see the message above).\n"
    "The official COCO tokenizer (and METEOR/SPICE) runs Java. Options:\n"
    "  - Check the internet connection/proxy and run again; the download is retried.\n"
    "  - Unzip any Java 11+ into this script's folder (a jdk*/jre* folder is found automatically),\n"
    "    or pass --java \"C:\\path\\to\\java.exe\".\n"
    "  - --tokenizer python scores CIDEr without Java (separate, unofficial metrics file).\n"
    "Generated captions are kept; rerun with --score-only."
)


def _adoptium_platform() -> tuple[str, str]:
    import platform
    system, machine = platform.system(), platform.machine().lower()
    os_name = {"Windows": "windows", "Linux": "linux", "Darwin": "mac"}.get(system)
    arch = {"amd64": "x64", "x86_64": "x64", "arm64": "aarch64", "aarch64": "aarch64"}.get(machine)
    if os_name is None or arch is None:
        raise RuntimeError(f"no automatic Java download for {system}/{machine}")
    return os_name, arch


def _adoptium_package(session: Any, os_name: str, arch: str) -> tuple[str, str, str]:
    """Return (download link, sha256, file name) of the latest Temurin JRE."""
    try:
        response = session.get(
            f"{ADOPTIUM_API}/assets/feature_releases/{JAVA_FEATURE_VERSION}/ga",
            params={"architecture": arch, "image_type": "jre", "os": os_name, "jvm_impl": "hotspot",
                    "heap_size": "normal", "vendor": "eclipse", "page_size": 1, "sort_order": "DESC"},
            timeout=(15, 60))
        response.raise_for_status()
        for release in response.json():
            for binary in release.get("binaries", []):
                package = binary.get("package") or {}
                if ((binary.get("os"), binary.get("architecture"), binary.get("image_type")) == (os_name, arch, "jre")
                        and package.get("link") and package.get("checksum")):
                    return package["link"], package["checksum"].strip().lower(), package.get("name") or package["link"].rsplit("/", 1)[-1]
        raise RuntimeError("release list had no matching JRE")
    except Exception as exc:  # fall back to the redirect endpoint + published .sha256.txt
        print(f"Adoptium release list unavailable ({type(exc).__name__}); trying direct link...", flush=True)
    redirect = session.get(
        f"{ADOPTIUM_API}/binary/latest/{JAVA_FEATURE_VERSION}/ga/{os_name}/{arch}/jre/hotspot/normal/eclipse",
        allow_redirects=False, timeout=(15, 60))
    link = redirect.headers.get("Location")
    if redirect.status_code not in (301, 302, 303, 307, 308) or not link:
        raise RuntimeError(f"Adoptium download link not available (HTTP {redirect.status_code})")
    checksum_response = session.get(link + ".sha256.txt", timeout=(15, 60))
    checksum_response.raise_for_status()
    checksum = checksum_response.text.split()[0].strip().lower()
    if len(checksum) != 64:
        raise RuntimeError("Adoptium checksum file not understood")
    return link, checksum, link.rsplit("/", 1)[-1]


def _safe_extract(archive: Path, destination: Path) -> None:
    import tarfile
    import zipfile
    destination.mkdir(parents=True)
    root = destination.resolve()
    if archive.name.endswith(".zip"):
        with zipfile.ZipFile(archive) as zf:
            for member in zf.namelist():
                if not (root / member).resolve().is_relative_to(root):
                    raise RuntimeError(f"unsafe path in Java archive: {member}")
            zf.extractall(destination)
    else:
        with tarfile.open(archive, "r:gz") as tf:
            if hasattr(tarfile, "data_filter"):
                tf.extractall(destination, filter="data")
            else:
                for member in tf.getmembers():
                    if not (root / member.name).resolve().is_relative_to(root) or member.isdev():
                        raise RuntimeError(f"unsafe entry in Java archive: {member.name}")
                tf.extractall(destination)


def download_portable_java(destination: Path = JAVA_DOWNLOAD_DIR) -> Path:
    """Download and unpack a portable Eclipse Temurin JRE (no installer, no admin rights)."""
    import tempfile
    from tqdm.auto import tqdm
    os_name, arch = _adoptium_platform()
    session = make_session()
    link, checksum, name = _adoptium_package(session, os_name, arch)
    print(f"Java not found: downloading portable Java {JAVA_FEATURE_VERSION} ({name}) into\n"
          f"  {destination}\n  One-time download of about 45 MB; later runs reuse it.", flush=True)
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination, prefix=".java_download_") as tmp:
        archive = Path(tmp) / name
        digest = hashlib.sha256()
        with session.get(link, stream=True, timeout=(15, 120)) as response:
            response.raise_for_status()
            total = int(response.headers.get("content-length") or 0) or None
            with archive.open("wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc="Java download") as bar:
                for chunk in response.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
                    digest.update(chunk)
                    bar.update(len(chunk))
        if digest.hexdigest() != checksum:
            raise RuntimeError("downloaded Java failed its SHA-256 check; nothing was installed")
        unpacked = Path(tmp) / "unpacked"
        _safe_extract(archive, unpacked)
        folders = [p for p in unpacked.iterdir() if p.is_dir()]
        if len(folders) != 1:
            raise RuntimeError("unexpected Java archive layout")
        target = destination / folders[0].name
        if target.exists():  # an earlier, broken copy: keep it, install beside it
            target = destination / (folders[0].name + "-redownload")
            shutil.rmtree(target, ignore_errors=True)
        os.replace(folders[0], target)
    print(f"Portable Java installed: {target}", flush=True)
    return target


_JAVA_RESULT: dict[str, Path | None] = {}


def ensure_java(explicit: Path | None = None, allow_download: bool = True) -> Path | None:
    """Find Java; if none exists, download a portable copy once. Result is cached per run."""
    if "java" in _JAVA_RESULT:
        return _JAVA_RESULT["java"]
    java = locate_java(explicit)
    if java is None and explicit is None and allow_download:
        try:
            download_portable_java()
            java = locate_java(None)
        except Exception as exc:
            print(f"Automatic Java download failed: {type(exc).__name__}: {exc}", flush=True)
    _JAVA_RESULT["java"] = java
    return java


def _java_candidates(explicit: Path | None) -> list[Path]:
    exe = "java.exe" if os.name == "nt" else "java"
    found: list[Path] = []
    if explicit is not None:
        explicit = explicit.expanduser()
        found += [explicit, explicit / exe, explicit / "bin" / exe] if explicit.is_dir() else [explicit]
        return found  # an explicit choice is never second-guessed
    on_path = shutil.which("java")
    if on_path:
        found.append(Path(on_path))
    if os.environ.get("JAVA_HOME"):
        found.append(Path(os.environ["JAVA_HOME"]) / "bin" / exe)
    # A portable Java unzipped next to this script or in the current folder,
    # e.g. .\jdk-21.0.5+11-jre\bin\java.exe (no installer or admin rights needed).
    for base in dict.fromkeys([Path(__file__).resolve().parent, Path.cwd()]):
        for pattern in ("jdk*", "jre*", "java*", "openjdk*", "OpenJDK*"):
            for folder in sorted((p for p in base.glob(pattern) if p.is_dir()), reverse=True):
                found += [folder / "bin" / exe, folder / "Contents" / "Home" / "bin" / exe]  # macOS layout
    if os.name == "nt":
        import glob
        for root in filter(None, {os.environ.get("ProgramFiles"), os.environ.get("ProgramFiles(x86)"),
                                  os.environ.get("LOCALAPPDATA") and os.path.join(os.environ["LOCALAPPDATA"], "Programs")}):
            for vendor in ("Eclipse Adoptium", "Java", "Microsoft", "Zulu", "Amazon Corretto", "BellSoft", "Common Files\\Oracle\\Java"):
                found += sorted((Path(p) for p in glob.glob(os.path.join(root, vendor, "*", "bin", exe))), reverse=True)
                found += [Path(p) for p in glob.glob(os.path.join(root, vendor, "javapath", exe))]
    return found


def locate_java(explicit: Path | None = None) -> Path | None:
    """Find a working java and put its folder first on PATH for pycocoevalcap.

    pycocoevalcap starts the bare command 'java', so the folder must be on PATH.
    """
    import subprocess
    for candidate in _java_candidates(explicit):
        if not candidate.is_file():
            continue
        # Absolute, because pycocoevalcap starts Java from its own folder.
        candidate = candidate.resolve()
        try:
            subprocess.run([str(candidate), "-version"], capture_output=True, timeout=30, check=True)
        except (OSError, subprocess.SubprocessError):
            continue
        bin_dir = str(candidate.parent)
        if os.path.normcase(bin_dir) not in {os.path.normcase(p) for p in os.environ.get("PATH", "").split(os.pathsep)}:
            os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
        return candidate
    return None


def check_environment(java: Path | None = None) -> None:
    print("Python:", sys.version.split()[0], "| Executable:", sys.executable, flush=True)
    java_path = locate_java(java)
    print("Java:", java_path or "not found yet (a portable copy is downloaded automatically when scoring)", flush=True)
    try:
        import torch
        print("PyTorch:", torch.__version__, flush=True)
        print("CUDA available:", torch.cuda.is_available(), flush=True)
        if torch.cuda.is_available():
            print("GPU:", torch.cuda.get_device_name(0), flush=True)
            props = torch.cuda.get_device_properties(0)
            print("GPU memory (GiB):", round(props.total_memory / 1024**3, 1), flush=True)
        else:
            print("BLIP-2 inference requires a compatible NVIDIA CUDA GPU for this 8-bit configuration.", flush=True)
            print("--download-only and --score-only do not require a GPU.", flush=True)
    except ImportError:
        print("PyTorch not installed; follow README setup.", flush=True)
    from importlib.metadata import PackageNotFoundError, version
    for package in ("transformers", "accelerate", "bitsandbytes", "datasets", "pycocoevalcap"):
        try:
            print(f"{package}:", version(package), flush=True)
        except PackageNotFoundError:
            print(f"{package}: NOT INSTALLED", flush=True)


def write_json_atomic(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(temp, path)


def normalize_refs(value: Any) -> list[str]:
    if not isinstance(value, list):
        raise ValueError("Expected a list of reference captions")
    refs = [(str(s.get("raw", s.get("caption", ""))) if isinstance(s, dict) else str(s)).strip() for s in value]
    if len(refs) < 5 or any(not s for s in refs):
        raise ValueError(f"Expected at least 5 nonempty captions, got {len(refs)}")
    return refs


def load_test_rows(n: int, seed: int) -> list[dict[str, Any]]:
    from datasets import load_dataset
    print(f"Loading {DATASET} test metadata...", flush=True)
    ds = load_dataset(DATASET, split="test")
    if len(ds) != 5000:
        raise ValueError(f"Expected 5000 Karpathy test images; found {len(ds)}")
    rows: list[dict[str, Any]] = []
    for item in ds:
        if item["split"] != "test" or item["filepath"] != "val2014":
            raise ValueError(f"Unexpected test split/2014 image: {item['filename']}")
        rows.append({
            "image_id": int(item["cocoid"]),
            "filename": str(item["filename"]),
            "filepath": str(item["filepath"]),
            "url": str(item["url"]),
            "references": normalize_refs(item["sentences"]),
        })
    rows.sort(key=lambda r: r["image_id"])
    if len({row["image_id"] for row in rows}) != 5000:
        raise ValueError("Duplicate COCO image IDs in metadata")
    if n < 5000:
        rows = sorted(random.Random(seed).sample(rows, n), key=lambda r: r["image_id"])
    return rows


def settings_dict(args: argparse.Namespace, rows: list[dict[str, Any]]) -> dict[str, Any]:
    from importlib.metadata import PackageNotFoundError, version
    versions: dict[str, str] = {}
    for package in ("torch", "transformers", "accelerate", "datasets", "bitsandbytes", "pycocoevalcap"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "not installed"
    ids = ",".join(str(r["image_id"]) for r in rows)
    return {
        "dataset": DATASET,
        "split": "test",
        "num_images": len(rows),
        "full_test": len(rows) == 5000,
        "subset_method": "all" if len(rows) == 5000 else "sample_without_replacement",
        "seed": args.seed,
        "selected_ids_sha256": hashlib.sha256(ids.encode("utf-8")).hexdigest(),
        "model": args.model,
        "prompt": args.prompt,
        "num_beams": args.num_beams,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "quantization": "bitsandbytes_int8",
        "image_preprocessing": "Original JPEG -> Pillow RGB -> Blip2Processor; no enhancement",
        "package_versions": versions,
    }


def validate_resume_config(path: Path, current: dict[str, Any]) -> None:
    if not path.exists():
        write_json_atomic(path, current)
        return
    old = json.loads(path.read_text(encoding="utf-8"))
    required = ("dataset", "split", "num_images", "seed", "selected_ids_sha256",
                "model", "prompt", "num_beams", "max_new_tokens", "do_sample", "quantization")
    changed = [key for key in required if old.get(key) != current.get(key)]
    if changed:
        raise RuntimeError(f"Results folder has a different experiment ({changed}). Choose a new --output-root.")


def load_existing_predictions(path: Path, allowed_ids: set[int],
                              repair_partial_tail: bool = False) -> dict[int, str]:
    """Read saved captions.

    A crash or power loss mid-write can leave an unterminated last line. With
    repair_partial_tail=True (generation mode) that line is dropped, or given
    its missing newline, so the run can resume instead of failing.
    """
    existing: dict[int, str] = {}
    if not path.exists():
        return existing
    data = path.read_bytes()
    if data and not data.endswith(b"\n"):
        cut = data.rfind(b"\n") + 1
        try:
            json.loads(data[cut:].decode("utf-8"))
            tail_is_complete = True
        except ValueError:  # includes UnicodeDecodeError / JSONDecodeError
            tail_is_complete = False
        if repair_partial_tail:
            if tail_is_complete:
                with path.open("ab") as f:
                    f.write(b"\n")
                data += b"\n"
            else:
                print(f"Dropping incomplete last line of {path.name} (interrupted write).", flush=True)
                with path.open("r+b") as f:
                    f.truncate(cut)
                data = data[:cut]
        elif not tail_is_complete:
            raise ValueError(f"Last line of {path} is incomplete; rerun generation to repair it.")
    for line_no, line in enumerate(data.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            image_id = int(record["image_id"])
            caption = str(record["caption"]).strip()
        except (ValueError, TypeError, KeyError) as exc:
            raise ValueError(f"Invalid prediction JSONL line {line_no}; fix/truncate it: {path}") from exc
        if image_id not in allowed_ids or image_id in existing or not caption:
            raise ValueError(f"Unexpected/duplicate/empty caption on line {line_no}: {path}")
        existing[image_id] = caption
    return existing


def candidate_urls(row: dict[str, Any]) -> list[str]:
    """Return HTTPS candidates for one COCO image, most reliable first.

    images.cocodataset.org is an S3 bucket whose name contains dots, so its
    HTTPS certificate does not match the hostname. The path-style S3 URL serves
    the same bucket with a valid certificate, so it is tried first.
    """
    folder = row.get("filepath", "val2014")
    filename = row["filename"]
    original = row.get("url", "")
    options = [f"https://s3.amazonaws.com/images.cocodataset.org/{folder}/{filename}"]
    if original.startswith("https://"):
        options.append(original)
    elif original.startswith("http://images.cocodataset.org/"):
        # Try the HTTPS form; deliberately avoid disabling SSL verification.
        options.append("https://" + original[len("http://"):])
    options.append(f"https://images.cocodataset.org/{folder}/{filename}")
    return list(dict.fromkeys(options))


def open_original_image(row: dict[str, Any], session: Any, args: argparse.Namespace):
    from PIL import Image
    import requests

    name = Path(row["filename"]).name
    if name != row["filename"]:
        raise ValueError(f"Unexpected image filename: {row['filename']}")
    # (path, is_user_supplied): user-supplied files are never deleted.
    candidates: list[tuple[Path, bool]] = []
    if args.local_image_dir is not None:
        candidates.extend([(args.local_image_dir / name, True),
                           (args.local_image_dir / row.get("filepath", "val2014") / name, True)])
    candidates.append((args.cache_dir / name, False))
    for path, is_user_supplied in candidates:
        if path.is_file():
            try:
                with Image.open(path) as image:
                    image.load()
                    return image.convert("RGB")
            except Exception as exc:
                if is_user_supplied:
                    raise RuntimeError(f"Local image corrupted: {path}") from exc
                print(f"Corrupt cached image; re-downloading: {path}", flush=True)
                path.unlink(missing_ok=True)

    errors: list[str] = []
    for url in candidate_urls(row):
        try:
            response = session.get(url, timeout=(15, 45))
            response.raise_for_status()
            data = response.content
            with Image.open(io.BytesIO(data)) as image:
                image.load()
                rgb = image.convert("RGB")
            args.cache_dir.mkdir(parents=True, exist_ok=True)
            destination = args.cache_dir / name
            temp = destination.with_suffix(destination.suffix + ".tmp")
            temp.write_bytes(data)
            os.replace(temp, destination)
            return rgb
        except (requests.RequestException, OSError, ValueError) as exc:
            errors.append(f"{url}: {type(exc).__name__} {str(exc)[:120]}")

    detail = "\n".join(errors)
    raise RuntimeError(
        f"Cannot download COCO image {name}.\n{detail}\n"
        "If the COCO image server is inaccessible, download/extract COCO val2014 "
        "images locally and pass --local-image-dir PATH. Do not set verify=False."
    )


def make_session():
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    session = requests.Session()
    retry = Retry(total=2, connect=2, read=2, backoff_factor=0.5,
                  status_forcelist=[429, 500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def download_images(args: argparse.Namespace, rows: list[dict[str, Any]]) -> None:
    from tqdm.auto import tqdm
    session = make_session()
    for row in tqdm(rows, desc="Checking COCO images"):
        open_original_image(row, session, args)
    print(f"Image test finished: {len(rows)} images accessible.", flush=True)


def dtype_kwarg_name() -> str:
    """Transformers >= 4.56 takes ``dtype``; older releases only know ``torch_dtype``."""
    import transformers
    from packaging.version import Version
    return "dtype" if Version(transformers.__version__).release >= (4, 56) else "torch_dtype"


def ensure_image_token_setup(processor: Any, model: Any) -> list[str]:
    """Give older BLIP-2 checkpoints the <image> placeholder setup newer Transformers needs.

    Recent Transformers versions put ``num_query_tokens`` <image> tokens in front
    of the prompt and overwrite them with the image features; they no longer
    fall back to plain concatenation. Checkpoints saved before that change (such
    as Salesforce/blip2-flan-t5-xl-coco) have no ``image_token_index`` in their
    config and no ``num_query_tokens`` in their processor, which makes generate()
    fail with "'bool' object has no attribute 'unsqueeze'". This fills both in,
    exactly as Salesforce's updated checkpoints do. Returns what was added.
    """
    from transformers import AddedToken

    config, tokenizer = model.config, processor.tokenizer
    added: list[str] = []
    image_token = getattr(processor, "image_token", None) or getattr(tokenizer, "image_token", None) or "<image>"
    image_token = getattr(image_token, "content", image_token)
    token_id = tokenizer.convert_tokens_to_ids(image_token)
    if token_id is None or token_id == tokenizer.unk_token_id:
        tokenizer.add_tokens([AddedToken(image_token, normalized=False, special=True)], special_tokens=True)
        token_id = tokenizer.convert_tokens_to_ids(image_token)
        added.append(f"tokenizer token {image_token}")

    current = getattr(config, "image_token_index", None)
    if current is None:
        current = getattr(config, "image_token_id", None)
    if current is None:
        rows = model.get_input_embeddings().num_embeddings
        if token_id >= rows:
            raise RuntimeError(f"Cannot add {image_token}: token id {token_id} is outside the "
                               f"model's {rows}-row embedding table.")
        config.image_token_index = token_id
        if getattr(config, "image_token_id", token_id) != token_id:  # versions without the alias
            config.image_token_id = token_id
        added.append(f"image_token_index={token_id}")
    elif current != token_id:
        raise RuntimeError(f"Checkpoint expects image token id {current}, but the tokenizer gives "
                           f"{image_token} id {token_id}; processor and model files do not match.")

    if getattr(processor, "num_query_tokens", None) is None:
        processor.num_query_tokens = config.num_query_tokens
        added.append(f"num_query_tokens={config.num_query_tokens}")
    return added


def generate_predictions(args: argparse.Namespace, rows: list[dict[str, Any]], path: Path) -> None:
    import torch
    from tqdm.auto import tqdm
    from transformers import BitsAndBytesConfig, Blip2ForConditionalGeneration, Blip2Processor

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. This 8-bit BLIP-2 experiment requires an NVIDIA CUDA GPU. "
                           "Check 'nvidia-smi', CUDA PyTorch, then --check-setup. "
                           "You can run --download-only or --score-only without CUDA.")
    if torch.cuda.get_device_capability(0)[0] < 7:
        raise RuntimeError("8-bit LLM quantization generally requires a newer NVIDIA GPU (e.g. RTX 20 series/T4 or newer).")

    allowed = {row["image_id"] for row in rows}
    old = load_existing_predictions(path, allowed, repair_partial_tail=True)
    remaining = [row for row in rows if row["image_id"] not in old]
    print(f"Already completed {len(old)}/{len(rows)}; remaining {len(remaining)}.", flush=True)
    if not remaining:
        return
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    print(f"Loading {args.model} with 8-bit weights...", flush=True)
    processor = Blip2Processor.from_pretrained(args.model)
    model = Blip2ForConditionalGeneration.from_pretrained(
        args.model,
        device_map="auto",
        quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        **{dtype_kwarg_name(): torch.float16},
    )
    model.eval()
    added = ensure_image_token_setup(processor, model)
    if added:
        print("Note: this checkpoint predates BLIP-2's <image> placeholder tokens; added "
              + ", ".join(added) + " (as in the updated Salesforce BLIP-2 checkpoints).", flush=True)
        config_path = args.output_dir / "run_config.json"
        if config_path.exists():
            saved = json.loads(config_path.read_text(encoding="utf-8"))
            saved["image_token_setup_added"] = added
            write_json_atomic(config_path, saved)
    # With device_map="auto" the input embeddings may not be on cuda:0.
    input_device = model.get_input_embeddings().weight.device
    session = make_session()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output, torch.inference_mode():
        for row in tqdm(remaining, desc="Generating captions"):
            image = open_original_image(row, session, args)
            inputs = processor(images=image, text=args.prompt, return_tensors="pt")
            inputs = {
                name: (tensor.to(device=input_device, dtype=torch.float16) if tensor.is_floating_point()
                       else tensor.to(device=input_device))
                for name, tensor in inputs.items()
            }
            generated = model.generate(**inputs, num_beams=args.num_beams,
                                       max_new_tokens=args.max_new_tokens, do_sample=False)
            caption = processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
            if not caption:
                raise RuntimeError(f"Empty caption from BLIP-2 for image {row['image_id']}")
            output.write(json.dumps({"image_id": row["image_id"], "caption": caption}, ensure_ascii=False) + "\n")
            output.flush()


# Tokens pycocoevalcap's PTBTokenizer drops after Stanford tokenization. Its list
# has "-LRB-" etc. in upper case, but output is lower-cased, so brackets survive
# as "-lrb-"/"-rrb-"; the approximation reproduces that.
_COCO_PUNCTUATION = {"''", "'", "``", "`", ".", "?", "!", ",", ":", "-", "--", "...", ";", '"'}
_BRACKETS = {"(": "-lrb-", ")": "-rrb-", "[": "-lsb-", "]": "-rsb-", "{": "-lcb-", "}": "-rcb-"}
_CONTRACTION = __import__("re").compile(r"(?i)(\w)(n't|'s|'re|'ve|'ll|'d|'m)\b")
_LONE_LETTER_DOT = __import__("re").compile(r"(?<![\w'\-])([a-z])\.(?=\s|$)")
_SPLIT = __import__("re").compile(r"(?<!\d)([.,])|([.,])(?!\d)|([^\w\s'\-/.,\x00])")


def python_tokenize(captions: dict[str, list[dict[str, str]]]) -> dict[str, list[str]]:
    """Java-free approximation of the COCO PTBTokenizer (lowercase, split, drop punctuation)."""
    out: dict[str, list[str]] = {}
    for key, items in captions.items():
        tokenized = []
        for item in items:
            text = item["caption"].lower().replace("\n", " ")
            # Stanford keeps a lone letter + period ("a.") as one abbreviation token.
            text = _LONE_LETTER_DOT.sub("\\1\x00", text)
            text = _SPLIT.sub(lambda mt: f" {mt.group(0)} ", text).replace("\x00", ".")
            text = _CONTRACTION.sub(r"\1 \2", text)
            tokens = []
            for tok in text.split():
                if not tok.startswith(("'s", "'re", "'ve", "'ll", "'d", "'m")) or len(tok) > 4:
                    tok = tok.strip("'")
                tok = _BRACKETS.get(tok, tok.strip("-") if set(tok) - {"-"} else tok)
                if tok and tok not in _COCO_PUNCTUATION:
                    tokens.append(tok)
            tokenized.append(" ".join(tokens))
        out[key] = tokenized
    return out


def resolve_tokenizer(args: argparse.Namespace) -> str:
    """Return 'ptb' or 'python' for this scoring run."""
    if args.tokenizer == "python":
        return "python"
    java = ensure_java(getattr(args, "java", None), not getattr(args, "no_java_download", False))
    if java is not None:
        print("Using Java:", java, flush=True)
        return "ptb"
    if args.tokenizer == "ptb":
        raise RuntimeError(JAVA_HELP)
    print("\nWARNING: " + JAVA_HELP + "\nScoring with the Python tokenizer instead "
          "(metrics_python_tokenizer.json).\n", flush=True)
    return "python"


def score_predictions(args: argparse.Namespace, rows: list[dict[str, Any]], path: Path) -> dict[str, Any]:
    from pycocoevalcap.cider.cider import Cider

    allowed = {row["image_id"] for row in rows}
    captions = load_existing_predictions(path, allowed)
    if len(captions) != len(rows):
        raise RuntimeError(f"Only {len(captions)}/{len(rows)} captions saved. "
                           "Complete generation first; --score-only cannot generate captions.")
    ordered = sorted(rows, key=lambda item: item["image_id"])
    refs = {str(row["image_id"]): [{"caption": s} for s in row["references"]] for row in ordered}
    preds = {str(row["image_id"]): [{"caption": captions[row["image_id"]]}] for row in ordered}
    tokenizer_used = resolve_tokenizer(args)
    if tokenizer_used == "ptb":
        from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
        print("Tokenizing with COCO PTBTokenizer (Java)...", flush=True)
        tokenizer = PTBTokenizer()
        refs = tokenizer.tokenize(refs)
        preds = tokenizer.tokenize(preds)
    else:
        print("Tokenizing with the Python approximation (no Java; CIDEr is approximate).", flush=True)
        refs = python_tokenize(refs)
        preds = python_tokenize(preds)

    scorers: list[tuple[str, Any, list[str]]] = [("CIDEr", Cider, ["CIDEr"])]
    if args.metrics == "all":
        from pycocoevalcap.bleu.bleu import Bleu
        from pycocoevalcap.meteor.meteor import Meteor
        from pycocoevalcap.spice.spice import Spice
        scorers = [
            ("BLEU", lambda: Bleu(4), ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4"]),
            ("CIDEr", Cider, ["CIDEr"]),
            ("METEOR", Meteor, ["METEOR"]),
            ("SPICE", Spice, ["SPICE"]),
        ]
    metrics_raw: dict[str, float] = {}
    errors: dict[str, str] = {}
    per_image_cider: dict[str, float] = {}
    for label, scorer_type, names in scorers:
        scorer = None
        try:
            print(f"Calculating {label}...", flush=True)
            scorer = scorer_type()
            aggregate, details = scorer.compute_score(refs, preds)
            if isinstance(aggregate, (tuple, list)):
                metrics_raw.update({name: float(value) for name, value in zip(names, aggregate)})
            else:
                metrics_raw[names[0]] = float(aggregate)
            if label == "CIDEr":
                # Cider returns per-image scores in the key order of the refs dict.
                per_image_cider = {image_id: float(v) for image_id, v in zip(refs.keys(), details)}
        except Exception as exc:
            if label == "CIDEr":
                raise RuntimeError("CIDEr computation failed (required metric). "
                                   "Check pycocoevalcap and Java installation.") from exc
            errors[label] = f"{type(exc).__name__}: {exc}"
            print(f"Optional {label} unavailable: {errors[label]}", flush=True)
        finally:
            if label == "METEOR" and scorer is not None:
                try:
                    if hasattr(scorer, "close"):  # pycocoevalcap >= 1.2
                        scorer.close()
                    elif getattr(scorer, "meteor_p", None) is not None:
                        scorer.meteor_p.stdin.close()
                        scorer.meteor_p.terminate()
                        scorer.meteor_p.wait(timeout=10)
                except Exception:
                    pass

    result = {
        "model": args.model,
        "prompt": args.prompt,
        "split": "MS COCO 2014 Karpathy test",
        "num_evaluated": len(rows),
        "full_test_set": len(rows) == 5000,
        "tokenizer": ("pycocoevalcap PTBTokenizer (official)" if tokenizer_used == "ptb"
                      else "Python approximation (NOT official; scores differ slightly)"),
        "scores_raw": metrics_raw,
        "scores_paper_scale_x100": {name: round(value * 100, 4) for name, value in metrics_raw.items()},
        "metric_errors": errors,
        "published_reference_only": {
            "description": "BLIP-2 Flan-T5-XL COCO-fine-tuned (paper Table 3)",
            "CIDEr_paper_scale": 144.5,
            "BLEU4_paper_scale": 42.4,
            "warning": "Do not equate pretrained/subset results with the paper's full fine-tuned benchmark.",
        },
    }
    output_dir = args.output_dir
    metrics_name = "metrics.json" if tokenizer_used == "ptb" else "metrics_python_tokenizer.json"
    write_json_atomic(output_dir / metrics_name, result)
    write_json_atomic(output_dir / "coco_predictions.json", [
        {"image_id": row["image_id"], "caption": captions[row["image_id"]]} for row in ordered
    ])
    if per_image_cider:
        write_json_atomic(output_dir / "per_image_cider_raw.json", per_image_cider)
    with (output_dir / "prediction_audit.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "filename", "predicted_caption", "reference_1", "reference_2",
                         "reference_3", "reference_4", "reference_5", "cider_raw"])
        for row in ordered:
            writer.writerow([row["image_id"], row["filename"], captions[row["image_id"]],
                             *row["references"][:5], per_image_cider.get(str(row["image_id"]), "")])
    print("\nRESULTS (paper display scale, metric * 100):", flush=True)
    print(json.dumps(result["scores_paper_scale_x100"], indent=2), flush=True)
    if errors:
        print("Optional scorer errors:", errors, flush=True)
    print("Saved output:", output_dir, flush=True)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.check_setup:
        check_environment(args.java)
        return 0
    if not args.download_only and args.tokenizer != "python":
        # Set Java up first (finding or downloading it), not after hours of generation.
        java = ensure_java(args.java, not args.no_java_download)
        if java is None and args.tokenizer == "ptb":
            print(JAVA_HELP, file=sys.stderr)
            return 2
        if java is None:
            print("NOTE: " + JAVA_HELP + "\nScoring will use the Python tokenizer "
                  "(metrics_python_tokenizer.json).\n", flush=True)
    print("Benchmark:", args.experiment)
    print("Checkpoint:", args.model)
    print("Karpathy images:", args.num_images)
    print("Output:", args.output_dir)
    print("Cache:", args.cache_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_test_rows(args.num_images, args.seed)
    validate_resume_config(args.output_dir / "run_config.json", settings_dict(args, rows))
    prediction_path = args.output_dir / "predictions.jsonl"
    if args.download_only:
        download_images(args, rows)
        return 0
    if not args.score_only:
        generate_predictions(args, rows, prediction_path)
    score_predictions(args, rows, prediction_path)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted. Already-generated captions remain in predictions.jsonl.", file=sys.stderr)
        raise SystemExit(130)
