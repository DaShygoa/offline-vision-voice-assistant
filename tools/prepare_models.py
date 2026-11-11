#!/usr/bin/env python
"""One-time model preparation for the classroom release (internet needed ONCE).

Downloads / converts the runtime models:
  1. ResNet18 (ImageNet pretrained) → ONNX        — 512-d feature extraction
  2. Depth Anything V2 Small → ONNX               — zero-shot localization
  3. sherpa-onnx Paraformer INT8 + Silero VAD     — Chinese speech input
  4. sherpa-onnx VITS                              — offline Chinese speech output

Offline Chinese speech uses a sherpa-onnx VITS model.

Usage:
    python tools/prepare_models.py            # all models (recommended)
    python tools/prepare_models.py --resnet18 --depth --asr
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import urllib.request
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, "models")
os.makedirs(MODELS, exist_ok=True)

# ------------------------------------------------------------------ mirrors
# Domestic (CN) mirrors are preferred so a classroom/offline release can be
# fetched quickly. Each entry is a LIST of candidate URLs tried in order; the
# upstream official URL is always kept as the final fallback so a dead mirror
# never blocks deployment. Override the mirror base with env vars if needed.
_HF_MIRROR = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com").rstrip("/")
_GH_PROXY = os.environ.get("GH_PROXY", "https://gh-proxy.com/").rstrip("/") + "/"

_SHERPA_RELEASE_BASE = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models"
_PARAFORMER_TAR = "sherpa-onnx-paraformer-zh-2023-09-14.tar.bz2"
_VAD_ONNX = "silero_vad.onnx"

# sherpa-onnx Paraformer Chinese (int8 e2e) + silero VAD release bundles.
# gh-proxy / ghproxy.net mirror GitHub release assets from mainland China.
SHERPA_PARAFORMER_INT8_URLS = [
    f"{_GH_PROXY}{_SHERPA_RELEASE_BASE}/{_PARAFORMER_TAR}",
    f"https://ghproxy.net/{_SHERPA_RELEASE_BASE}/{_PARAFORMER_TAR}",
    f"{_SHERPA_RELEASE_BASE}/{_PARAFORMER_TAR}",
]
SILERO_VAD_URLS = [
    f"{_GH_PROXY}{_SHERPA_RELEASE_BASE}/{_VAD_ONNX}",
    f"https://ghproxy.net/{_SHERPA_RELEASE_BASE}/{_VAD_ONNX}",
    f"{_SHERPA_RELEASE_BASE}/{_VAD_ONNX}",
]
# Kept as single-URL aliases for backwards compatibility.
SHERPA_PARAFORMER_INT8_URL = SHERPA_PARAFORMER_INT8_URLS[0]
SILERO_VAD_URL = SILERO_VAD_URLS[0]


def _download(url: str | list[str], dest: str, desc: str = "") -> str:
    """Download *dest* from the first reachable URL.

    ``url`` may be a single string or an ordered list of mirrors; the first
    candidate that returns bytes wins, so a domestic mirror is tried before
    the (often slow/blocked) upstream host.
    """
    urls = [url] if isinstance(url, str) else list(url)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.isfile(dest) and os.path.getsize(dest) > 1024:
        print(f"[skip] {dest} exists")
        return dest
    print(f"[get ] {desc or urls[0]}\n       → {dest}")
    tmp = dest + ".part"
    last_err: Exception | None = None
    for candidate in urls:
        try:
            print(f"       mirror: {candidate}")
            urllib.request.urlretrieve(candidate, tmp)
            if os.path.getsize(tmp) <= 1024:
                raise OSError("downloaded file is too small (<1 KiB)")
            os.replace(tmp, dest)
            return dest
        except Exception as e:  # try the next mirror / upstream
            last_err = e
            print(f"       [warn] failed ({e}); trying next source …")
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass
    raise last_err if last_err else RuntimeError("no download URL provided")


def _extract(archive: str, out_dir: str) -> str:
    print(f"[unz] {archive} → {out_dir}")
    if archive.endswith(".zip"):
        with zipfile.ZipFile(archive, "r") as z:
            z.extractall(out_dir)
    elif archive.endswith((".tar.bz2", ".tbz2", ".tar")):
        import tarfile

        with tarfile.open(archive, "r:*") as t:
            t.extractall(out_dir)
    return out_dir


# ----------------------------------------------------------------- ResNet18
# torchvision pulls this exact filename from download.pytorch.org when the
# weight is not already in torch's hub cache. Pre-seeding the cache lets a
# domestic mirror serve it (and keeps the run fully offline afterwards).
# No public CN mirror is guaranteed to host the *torchvision-format* file, so
# the mirror is configurable: set RESNET18_WEIGHT_MIRROR to a trusted URL.
RESNET18_WEIGHT_FILE = "resnet18-f37072fd.pth"
_resnet_mirror = os.environ.get("RESNET18_WEIGHT_MIRROR", "").strip()
RESNET18_WEIGHT_URLS = (
    ([_resnet_mirror] if _resnet_mirror else [])
    + [f"https://download.pytorch.org/models/{RESNET18_WEIGHT_FILE}"]
)


def _seed_resnet18_cache(torch) -> None:
    """Place resnet18-f37072fd.pth in torch's hub cache so torchvision is offline."""
    cache_dir = os.path.join(torch.hub.get_dir(), "checkpoints")
    cached = os.path.join(cache_dir, RESNET18_WEIGHT_FILE)
    if os.path.isfile(cached) and os.path.getsize(cached) > 10_000_000:
        print(f"[skip] torchvision weight cached: {cached}")
        return
    try:
        _download(RESNET18_WEIGHT_URLS, cached, "ResNet18 ImageNet weight")
    except Exception as e:
        # Non-fatal: torchvision will retry its own official download.
        print(f"[warn] weight mirror pre-seed failed ({e}); torchvision will retry upstream.")


def prepare_resnet18() -> str:
    """Export ImageNet-pretrained ResNet18 (avgpool 512-d) → ONNX."""
    rn_dir = os.path.join(MODELS, "resnet18")
    os.makedirs(rn_dir, exist_ok=True)
    onnx_path = os.path.join(rn_dir, "resnet18.onnx")

    if not (os.path.isfile(onnx_path) and os.path.getsize(onnx_path) > 10_000_000):
        print("[exp ] export ResNet18 (ImageNet pretrained, fc→Identity) via torchvision …")
        try:
            import torch
            from torchvision.models import ResNet18_Weights, resnet18

            _seed_resnet18_cache(torch)
            model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            model.fc = torch.nn.Identity()  # 512-d avgpool feature
            model.eval()
            dummy = torch.randn(1, 3, 224, 224)
            torch.onnx.export(
                model,
                dummy,
                onnx_path,
                input_names=["input"],
                output_names=["feature"],
                opset_version=17,
                dynamo=False,
                do_constant_folding=True,
            )
            print(f"[ok ] {onnx_path} ({os.path.getsize(onnx_path)} bytes)")
        except Exception as e:
            print(f"[err] ResNet18 torch export failed: {e}")
            print(
                "      Manual: place ResNet18 ONNX (input 1x3x224x224 → feature 1x512) at:\n"
                f"      {onnx_path}"
            )
            return ""

    if os.path.isfile(onnx_path):
        print(f"[done] ResNet18 ONNX ready: {onnx_path}")
    return onnx_path


# ------------------------------------------------------------------ Depth
# Depth Anything V2 Small ONNX — zero-shot object localization.
# hf-mirror.com is the domestic mirror of huggingface.co (same repo path).
# https://huggingface.co/onnx-community/depth-anything-v2-small
_DEPTH_REPO_PATH = "onnx-community/depth-anything-v2-small/resolve/main/onnx/model.onnx"
DEPTH_V2_SMALL_ONNX_URLS = [
    f"{_HF_MIRROR}/{_DEPTH_REPO_PATH}",
    f"https://huggingface.co/{_DEPTH_REPO_PATH}",
]
DEPTH_V2_SMALL_ONNX_URL = DEPTH_V2_SMALL_ONNX_URLS[0]


def prepare_depth() -> str:
    """Download Depth Anything V2 Small ONNX for zero-shot localization."""
    depth_dir = os.path.join(MODELS, "depth_anything_v2")
    os.makedirs(depth_dir, exist_ok=True)
    onnx_path = os.path.join(depth_dir, "depth_anything_v2_vits.onnx")

    if os.path.isfile(onnx_path) and os.path.getsize(onnx_path) > 50_000_000:
        print(f"[skip] {onnx_path}")
        return onnx_path

    try:
        _download(DEPTH_V2_SMALL_ONNX_URLS, onnx_path, "Depth Anything V2 Small ONNX")
        if os.path.getsize(onnx_path) < 50_000_000:
            print(f"[warn] file seems too small ({os.path.getsize(onnx_path)} bytes), download may be incomplete")
        else:
            print(f"[ok ] Depth Anything V2 Small → {onnx_path}")
    except Exception as e:
        print(f"[err] Depth model download failed: {e}")
        print("      Manual download (try in this order):")
        for url in DEPTH_V2_SMALL_ONNX_URLS:
            print(f"      {url}")
        print(f"      Save as: {onnx_path}")
        print("      Alternative: export from HuggingFace transformers:")
        print("        pip install optimum[onnxruntime]")
        print("        optimum-cli export onnx --model onnx-community/depth-anything-v2-small --task depth-estimation <out_dir>")
        return ""
    return onnx_path


# ---------------------------------------------------------------------- ASR
def prepare_asr() -> str:
    asr_dir = os.path.join(MODELS, "sherpa-onnx-paraformer-zh-int8")
    os.makedirs(asr_dir, exist_ok=True)
    tokens = os.path.join(asr_dir, "tokens.txt")
    if os.path.isfile(tokens):
        print(f"[skip] {asr_dir}")
        _prepare_vad()
        return asr_dir

    try:
        # The downloaded archive and extracted tree are build intermediates,
        # not runtime models; do not leave a second copy of the weights behind.
        with tempfile.TemporaryDirectory(prefix="vision_asr_prep_") as tmp:
            archive = os.path.join(tmp, "paraformer_zh.tar.bz2")
            extracted = os.path.join(tmp, "extracted")
            _download(SHERPA_PARAFORMER_INT8_URLS, archive, "sherpa-onnx Paraformer zh")
            _extract(archive, extracted)
            found_root = None
            for root, dirs, files in os.walk(extracted):
                if "tokens.txt" in files:
                    found_root = root
                    break
            if found_root:
                import shutil

                for fn in os.listdir(found_root):
                    src = os.path.join(found_root, fn)
                    dst = os.path.join(asr_dir, fn)
                    if os.path.isfile(src):
                        shutil.copy2(src, dst)
                _alias_int8(asr_dir, extracted)
                print(f"[ok ] ASR models → {asr_dir}")
            else:
                print("[err] tokens.txt not found in archive")
    except Exception as e:
        print(f"[err] ASR download failed: {e}")
        print(f"      Manual: put tokens.txt / model.onnx / decoder.onnx under {asr_dir}")
    _prepare_vad()
    return asr_dir


def _alias_int8(asr_dir: str, extracted_dir: str | None = None) -> None:
    """Normalize names for config; e2e path only needs model.int8.onnx + tokens + am.mvn."""
    import shutil

    files = os.listdir(asr_dir)
    enc_int8 = next((f for f in files if f in ("model.int8.onnx", "encoder.int8.onnx")), None)
    enc = next((f for f in files if f in ("model.onnx", "encoder.onnx")), None)
    dec_int8 = next((f for f in files if f in ("decoder.int8.onnx",)), None)
    dec = next((f for f in files if f in ("decoder.onnx",)), None)
    if enc_int8 and enc_int8 != "model.int8.onnx":
        shutil.copy2(os.path.join(asr_dir, enc_int8), os.path.join(asr_dir, "model.int8.onnx"))
    elif enc:
        shutil.copy2(os.path.join(asr_dir, enc), os.path.join(asr_dir, "model.int8.onnx"))
    if dec_int8:
        shutil.copy2(os.path.join(asr_dir, dec_int8), os.path.join(asr_dir, "decoder.int8.onnx"))
    elif dec:
        shutil.copy2(os.path.join(asr_dir, dec), os.path.join(asr_dir, "decoder.int8.onnx"))
    # copy a real speech sample for benchmark ASR
    for root, _, fs in os.walk(asr_dir):
        for fn in fs:
            if fn.endswith(".wav"):
                src = os.path.join(root, fn)
                dst = os.path.join(ROOT, "data", "asr_sample.wav")
                if not os.path.isfile(dst):
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    shutil.copy2(src, dst)
                    print(f"[ok ] asr sample → {dst}")
                return
    # also check extract dir
    for root, _, fs in os.walk(extracted_dir or ""):
        for fn in fs:
            if fn.endswith(".wav"):
                src = os.path.join(root, fn)
                dst = os.path.join(ROOT, "data", "asr_sample.wav")
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(src, dst)
                print(f"[ok ] asr sample → {dst}")
                return


def _prepare_vad() -> str:
    vad_dir = os.path.join(MODELS, "silero_vad")
    os.makedirs(vad_dir, exist_ok=True)
    vad_path = os.path.join(vad_dir, "silero_vad.onnx")
    if os.path.isfile(vad_path):
        print(f"[skip] {vad_path}")
        return vad_path
    try:
        _download(SILERO_VAD_URLS, vad_path, "Silero VAD ONNX")
    except Exception as e:
        print(f"[err] VAD download failed: {e}")
        print("      Manual download (try in this order):")
        for url in SILERO_VAD_URLS:
            print(f"      {url}")
        print(f"      Save as: {vad_path}")
    return vad_path


def prepare_tts() -> str:
    name = "sherpa-onnx-vits-zh-ll"
    target = os.path.join(MODELS, name)
    required = ("model.onnx", "tokens.txt", "lexicon.txt", "phone.fst", "date.fst", "number.fst")
    if all(os.path.isfile(os.path.join(target, f)) for f in required):
        print(f"[skip] {target} exists")
        return target
    url = f"https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/{name}.tar.bz2"
    archive = os.path.join(MODELS, name + ".tar.bz2")
    _download([url, f"{_GH_PROXY}{url}", f"https://ghproxy.net/{url}"], archive, "Offline Chinese TTS")
    _extract(archive, MODELS)
    missing = [f for f in required if not os.path.isfile(os.path.join(target, f))]
    if missing:
        raise RuntimeError(f"incomplete offline TTS model: {missing}")
    os.remove(archive)
    return target


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Prepare the classroom-release models (ResNet18 / Depth / ASR / TTS).")
    p.add_argument("--all", action="store_true", help="prepare all models (default when no flag given)")
    p.add_argument("--resnet18", action="store_true", help="export ImageNet ResNet18 (512-d features)")
    p.add_argument("--depth", action="store_true", help="download Depth Anything V2 Small (zero-shot localization)")
    p.add_argument("--asr", action="store_true", help="download Paraformer INT8 + Silero VAD")
    p.add_argument("--tts", action="store_true", help="download offline Chinese TTS")
    args = p.parse_args(argv)

    do_all = args.all or not any([args.resnet18, args.depth, args.asr, args.tts])
    if do_all or args.resnet18:
        prepare_resnet18()
    if do_all or args.depth:
        prepare_depth()
    if do_all or args.asr:
        prepare_asr()
    if do_all or args.tts:
        prepare_tts()
    print("\nModel prep done. Offline TTS is available after downloading its model.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
