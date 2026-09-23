#!/usr/bin/env python
"""One-time model preparation (requires internet ONCE — runtime stays fully offline).

Downloads / converts:
  1. YOLOv8n ONNX  → OpenVINO IR (xml/bin) for OpenVINO backend
  2. DINOv2 ViT-S/14 ONNX (community export) for embedding
  3. sherpa-onnx Paraformer Chinese INT8 + Silero VAD
  4. Optional note for GPT-SoVITS (installed separately)

Usage:
    python tools/prepare_models.py --all
    python tools/prepare_models.py --yolo --dino --asr
    python tools/prepare_models.py --yolo-only-local path/to/yolov8n.onnx
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.request
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, "models")
os.makedirs(MODELS, exist_ok=True)

# --- YOLOv8n ----------------------------------------------------------------
# Prefer local ultralytics export (pip install ultralytics torch once).
YOLOV8N_PT_URL = "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt"
YOLOV8N_ONNX_DIRECT = [
    # community mirrors of ultralytics yolov8n.onnx (first success wins)
    "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.onnx",
]

# DINOv2 ViT-S/14 — export via torch.hub (one-time; runtime needs no torch)
DINOV2_HUB = ("facebookresearch/dinov2", "dinov2_vits14")

# sherpa-onnx Paraformer Chinese (int8 e2e) + silero VAD release bundles
SHERPA_PARAFORMER_INT8_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-paraformer-zh-2023-09-14.tar.bz2"
)
SILERO_VAD_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx"
)


def _download(url: str, dest: str, desc: str = "") -> str:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.isfile(dest) and os.path.getsize(dest) > 1024:
        print(f"[skip] {dest} exists")
        return dest
    print(f"[get ] {desc or url}\n       → {dest}")
    tmp = dest + ".part"
    urllib.request.urlretrieve(url, tmp)
    os.replace(tmp, dest)
    return dest


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


# --------------------------------------------------------------------- YOLO
def prepare_yolo(local_onnx: str | None = None) -> str:
    yolo_dir = os.path.join(MODELS, "yolov8n")
    os.makedirs(yolo_dir, exist_ok=True)
    onnx_path = os.path.join(yolo_dir, "yolov8n.onnx")
    ir_xml = os.path.join(yolo_dir, "yolov8n_openvino_model", "yolov8n.xml")

    if local_onnx:
        import shutil

        if not os.path.isfile(local_onnx):
            raise FileNotFoundError(local_onnx)
        shutil.copy2(local_onnx, onnx_path)
        print(f"[ok ] copied local ONNX → {onnx_path}")
    elif not os.path.isfile(onnx_path):
        got = False
        for url in YOLOV8N_ONNX_DIRECT:
            try:
                _download(url, onnx_path, "YOLOv8n ONNX")
                if os.path.getsize(onnx_path) > 1_000_000:
                    got = True
                    break
            except Exception as e:
                print(f"[warn] {url} failed: {e}")
                if os.path.isfile(onnx_path):
                    os.remove(onnx_path)
        if not got:
            # export from .pt via ultralytics (one-time pip install ultralytics)
            pt_path = os.path.join(yolo_dir, "yolov8n.pt")
            try:
                if not os.path.isfile(pt_path):
                    _download(YOLOV8N_PT_URL, pt_path, "YOLOv8n.pt")
                from ultralytics import YOLO

                model = YOLO(pt_path)
                model.export(format="onnx", imgsz=640, opset=13, simplify=True)
                exported = pt_path.replace(".pt", ".onnx")
                if os.path.isfile(exported):
                    if os.path.abspath(exported) != os.path.abspath(onnx_path):
                        os.replace(exported, onnx_path)
                    print(f"[ok ] exported → {onnx_path}")
                    # also export OpenVINO IR (FP16-friendly)
                    try:
                        model.export(format="openvino", imgsz=640)
                        print(f"[ok ] OpenVINO IR → {os.path.dirname(ir_xml)}")
                    except Exception as e:
                        print(f"[warn] openvino export: {e}")
            except Exception as e:
                print(f"[err] {e}")
                print("      Manual: pip install ultralytics && re-run, or place yolov8n.onnx at:")
                print(f"      {onnx_path}")

    # ONNX → OpenVINO IR (if not already exported by ultralytics)
    if os.path.isfile(onnx_path) and not os.path.isfile(ir_xml):
        try:
            import openvino as ov

            os.makedirs(os.path.dirname(ir_xml), exist_ok=True)
            print("[cvrt] ONNX → OpenVINO IR …")
            ov.convert_model(onnx_path, output_model=ir_xml)
            print(f"[ok ] {ir_xml}")
        except Exception as e:
            print(f"[warn] openvino convert failed: {e}")
            print("      YOLO can still run via opencv_dnn backend with the ONNX file.")
    if os.path.isfile(ir_xml):
        print(f"[done] YOLO IR ready: {ir_xml}")
    return onnx_path


# --------------------------------------------------------------------- DINO
def prepare_dino() -> str:
    dino_dir = os.path.join(MODELS, "dinov2_vits14")
    os.makedirs(dino_dir, exist_ok=True)
    onnx_path = os.path.join(dino_dir, "dinov2_vits14.onnx")
    if os.path.isfile(onnx_path) and os.path.getsize(onnx_path) > 10_000_000:
        print(f"[skip] {onnx_path}")
        return onnx_path

    print("[exp ] export DINOv2 ViT-S/14 via torch.hub (one-time; temp torch) …")
    try:
        import torch
        from torch import nn

        repo, name = DINOV2_HUB
        model = torch.hub.load(repo, name)
        model.eval()

        class _W(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, pixel_values):
                return self.m.forward(pixel_values, masks=None)

        w = _W(model)
        dummy = torch.randn(1, 3, 224, 224)
        torch.onnx.export(
            w,
            dummy,
            onnx_path,
            input_names=["pixel_values"],
            output_names=["features"],
            opset_version=17,
            dynamo=False,
            do_constant_folding=True,
        )
        print(f"[ok ] {onnx_path} ({os.path.getsize(onnx_path)} bytes)")
        return onnx_path
    except Exception as e:
        print(f"[err] torch export failed: {e}")
    print(
        f"[manual] place DINOv2 ViT-S/14 ONNX (input 1x3x224x224 → output 1x384) at:\n         {onnx_path}"
    )
    return ""


# ---------------------------------------------------------------------- ASR
def prepare_asr() -> str:
    asr_dir = os.path.join(MODELS, "sherpa-onnx-paraformer-zh-int8")
    os.makedirs(asr_dir, exist_ok=True)
    tokens = os.path.join(asr_dir, "tokens.txt")
    if os.path.isfile(tokens):
        print(f"[skip] {asr_dir}")
        return asr_dir

    archive = os.path.join(MODELS, "paraformer_zh.tar.bz2")
    try:
        _download(SHERPA_PARAFORMER_INT8_URL, archive, "sherpa-onnx Paraformer zh")
        tmp = os.path.join(MODELS, "_paraformer_extract")
        _extract(archive, tmp)
        # find extracted folder
        found_root = None
        for root, dirs, files in os.walk(tmp):
            if "tokens.txt" in files:
                found_root = root
                break
        if found_root:
            import shutil

            for fn in os.listdir(found_root):
                src = os.path.join(found_root, fn)
                dst = os.path.join(asr_dir, fn)
                if os.path.isfile(src):
                    if os.path.abspath(src) != os.path.abspath(dst):
                        shutil.copy2(src, dst)
            # prefer int8 encoder/decoder names for config
            _alias_int8(asr_dir)
            print(f"[ok ] ASR models → {asr_dir}")
        else:
            print("[err] tokens.txt not found in archive")
    except Exception as e:
        print(f"[err] ASR download failed: {e}")
        print(f"      Manual: put tokens.txt / model.onnx / decoder.onnx under {asr_dir}")
    _prepare_vad()
    return asr_dir


def _alias_int8(asr_dir: str) -> None:
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
    for root, _, fs in os.walk(os.path.join(MODELS, "_paraformer_extract")):
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
        _download(SILERO_VAD_URL, vad_path, "Silero VAD ONNX")
    except Exception as e:
        print(f"[err] VAD download failed: {e}")
    return vad_path


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--all", action="store_true")
    p.add_argument("--yolo", action="store_true")
    p.add_argument("--dino", action="store_true")
    p.add_argument("--asr", action="store_true")
    p.add_argument("--yolo-only-local", default=None, help="use a local yolov8n.onnx path")
    args = p.parse_args(argv)

    if not any([args.all, args.yolo, args.dino, args.asr, args.yolo_only_local]):
        p.print_help()
        print("\nNOTE: GPT-SoVITS is installed separately — see voice/tts_server.py")
        return 0

    if args.all or args.yolo or args.yolo_only_local:
        prepare_yolo(args.yolo_only_local)
    if args.all or args.dino:
        prepare_dino()
    if args.all or args.asr:
        prepare_asr()

    print("\nAll requested model prep done. Runtime = fully offline.")
    print("GPT-SoVITS: clone/install separately, then:")
    print(r"  python -m voice.tts_server --gpt-sovits-path D:\GPT-SoVITS --port 19880")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
