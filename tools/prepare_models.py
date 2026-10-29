#!/usr/bin/env python
"""One-time model preparation (requires internet ONCE — runtime stays fully offline).

Downloads / converts:
  1. ResNet18 (ImageNet pretrained) → ONNX  ← 选题一 core feature model
  2. YOLOv8n → ONNX (optional ROI helper)
  3. Depth Anything V2 Small → ONNX  ← zero-shot object localization
  4. MobileSAM (TinyViT encoder + mask decoder) → ONNX  ← pixel-precise mask refinement
  5. sherpa-onnx Paraformer Chinese INT8 + Silero VAD
  6. Optional note for GPT-SoVITS (installed separately)

Usage:
    python tools/prepare_models.py --all
    python tools/prepare_models.py --resnet18 --asr
    python tools/prepare_models.py --depth --mobilesam
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
            except Exception as e:
                print(f"[err] {e}")
                print("      Manual: pip install ultralytics && re-run, or place yolov8n.onnx at:")
                print(f"      {onnx_path}")

    if os.path.isfile(onnx_path):
        print(f"[done] YOLO ONNX ready: {onnx_path}")
    return onnx_path


# ----------------------------------------------------------------- ResNet18
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
# Depth Anything V2 Small ONNX — zero-shot object localization
# https://huggingface.co/onnx-community/depth-anything-v2-small
DEPTH_V2_SMALL_ONNX_URL = (
    "https://huggingface.co/onnx-community/depth-anything-v2-small/resolve/main/onnx/model.onnx"
)


def prepare_depth() -> str:
    """Download Depth Anything V2 Small ONNX for zero-shot localization."""
    depth_dir = os.path.join(MODELS, "depth_anything_v2")
    os.makedirs(depth_dir, exist_ok=True)
    onnx_path = os.path.join(depth_dir, "depth_anything_v2_vits.onnx")

    if os.path.isfile(onnx_path) and os.path.getsize(onnx_path) > 50_000_000:
        print(f"[skip] {onnx_path}")
        return onnx_path

    try:
        _download(DEPTH_V2_SMALL_ONNX_URL, onnx_path, "Depth Anything V2 Small ONNX")
        if os.path.getsize(onnx_path) < 50_000_000:
            print(f"[warn] file seems too small ({os.path.getsize(onnx_path)} bytes), download may be incomplete")
        else:
            print(f"[ok ] Depth Anything V2 Small → {onnx_path}")
    except Exception as e:
        print(f"[err] Depth model download failed: {e}")
        print("      Manual download:")
        print(f"      {DEPTH_V2_SMALL_ONNX_URL}")
        print(f"      Save as: {onnx_path}")
        print("      Alternative: export from HuggingFace transformers:")
        print("        pip install optimum[onnxruntime]")
        print("        optimum-cli export onnx --model onnx-community/depth-anything-v2-small --task depth-estimation <out_dir>")
        return ""
    return onnx_path


# ------------------------------------------------------------- MobileSAM
# MobileSAM (arXiv 2306.14289) — TinyViT encoder (5M params) + SAM mask decoder
# Encoder ONNX:  image (1,3,1024,1024) → embedding (1,256,64,64)
# Decoder ONNX:  embedding + point_coords + point_labels → mask (1,256,256)
#
# The checkpoint (mobile_sam.pt) is needed once to export ONNX via torch.
MOBILESAM_PT_URL = (
    "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt"
)


def prepare_mobilesam() -> str:
    """Export MobileSAM encoder + decoder to ONNX (needs torch once)."""
    sam_dir = os.path.join(MODELS, "mobile_sam")
    os.makedirs(sam_dir, exist_ok=True)
    enc_path = os.path.join(sam_dir, "mobilesam_encoder.onnx")
    dec_path = os.path.join(sam_dir, "mobilesam_decoder.onnx")

    if os.path.isfile(enc_path) and os.path.isfile(dec_path):
        print(f"[skip] MobileSAM ONNX already present in {sam_dir}")
        return sam_dir

    # 1) download checkpoint
    pt_path = os.path.join(sam_dir, "mobile_sam.pt")
    if not os.path.isfile(pt_path):
        try:
            _download(MOBILESAM_PT_URL, pt_path, "MobileSAM checkpoint (mobile_sam.pt)")
        except Exception as e:
            print(f"[err] MobileSAM checkpoint download failed: {e}")
            print(f"      Manual: {MOBILESAM_PT_URL}")
            print(f"      Save as: {pt_path}")
            return ""

    # 2) export encoder + decoder ONNX via MobileSAM's export script (needs torch)
    print("[exp ] exporting MobileSAM encoder + decoder → ONNX (requires torch) …")
    try:
        import sys
        import subprocess
        # try the standard MobileSAM export entrypoint if installed
        result = subprocess.run(
            [
                sys.executable, "-c",
                "from mobile_sam import sam_model_registry; "
                "import mobile_sam; print('mobile_sam OK')",
            ],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            print("[warn] mobile_sam package not installed. Install with:")
            print("       pip install git+https://github.com/ChaoningZhang/MobileSAM.git")
            print("       Then re-run: python tools/prepare_models.py --mobilesam")
            print("       Or export manually (see MobileSAM scripts/export_onnx_model.py):")
            print(f"         python scripts/export_onnx_model.py --checkpoint {pt_path} \\")
            print("           --model-type vit_t --output mobilesam_decoder.onnx \\")
            print("           --image-encoder-output mobilesam_encoder.onnx \\")
            print("           --fixed-image-size 1024,1024 --return-single-mask")
            return ""

        # inline export using mobile_sam API
        import torch
        from mobile_sam import sam_model_registry

        sam = sam_model_registry["vit_t"](checkpoint=pt_path)
        sam.eval()

        # --- encoder ONNX ---
        dummy_img = torch.randn(1, 3, 1024, 1024)
        torch.onnx.export(
            sam.image_encoder, dummy_img, enc_path,
            input_names=["input_image"], output_names=["image_embedding"],
            opset_version=17, dynamic_axes=None, dynamo=False,
        )
        print(f"[ok ] encoder → {enc_path}")

        # --- decoder ONNX (prompt encoder + mask decoder combined) ---
        dummy_emb = torch.randn(1, 256, 64, 64)
        dummy_coords = torch.tensor([[[512.0, 512.0]]])
        dummy_labels = torch.tensor([[1.0]])
        dummy_size = torch.tensor([[1080, 1920]], dtype=torch.int32)

        class DecoderWrapper(torch.nn.Module):
            def __init__(self, sam_model):
                super().__init__()
                self.prompt_encoder = sam_model.prompt_encoder
                self.mask_decoder = sam_model.mask_decoder

            def forward(self, image_embedding, point_coords, point_labels, orig_im_size):
                sparse, dense = self.prompt_encoder(
                    points=(point_coords, point_labels),
                    boxes=None, masks=None,
                )
                masks, iou_pred, _ = self.mask_decoder(
                    image_embedding=image_embedding,
                    image_pe=self.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse,
                    dense_prompt_embeddings=dense,
                    multimask_output=False,
                )
                return masks, iou_pred

        wrapper = DecoderWrapper(sam)
        torch.onnx.export(
            wrapper,
            (dummy_emb, dummy_coords, dummy_labels, dummy_size),
            dec_path,
            input_names=["image_embedding", "point_coords", "point_labels", "orig_im_size"],
            output_names=["masks", "scores"],
            opset_version=17,
            dynamic_axes={"point_coords": {1: "num_points"}, "point_labels": {1: "num_points"}},
            dynamo=False,
        )
        print(f"[ok ] decoder → {dec_path}")
    except Exception as e:
        print(f"[err] MobileSAM export failed: {e}")
        print("      Install: pip install git+https://github.com/ChaoningZhang/MobileSAM.git")
        print("      Then re-run: python tools/prepare_models.py --mobilesam")
        return ""

    return sam_dir


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
    p.add_argument("--resnet18", action="store_true", help="export ImageNet ResNet18 (选题一)")
    p.add_argument("--yolo", action="store_true", help="optional YOLOv8n ROI helper")
    p.add_argument("--depth", action="store_true", help="Depth Anything V2 Small for zero-shot localization")
    p.add_argument("--mobilesam", action="store_true", help="MobileSAM encoder+decoder ONNX for mask refinement")
    p.add_argument("--asr", action="store_true")
    p.add_argument("--yolo-only-local", default=None, help="use a local yolov8n.onnx path")
    args = p.parse_args(argv)

    if not any([args.all, args.resnet18, args.yolo, args.depth, args.mobilesam, args.asr, args.yolo_only_local]):
        p.print_help()
        print("\nNOTE: GPT-SoVITS is installed separately — see voice/tts_server.py")
        return 0

    if args.all or args.resnet18:
        prepare_resnet18()
    if args.all or args.yolo or args.yolo_only_local:
        prepare_yolo(args.yolo_only_local)
    if args.all or args.depth:
        prepare_depth()
    if args.all or args.mobilesam:
        prepare_mobilesam()
    if args.all or args.asr:
        prepare_asr()
    print("\nAll requested model prep done. Runtime = fully offline.")
    print("GPT-SoVITS: clone/install separately, then:")
    print(r"  python -m voice.tts_server --gpt-sovits-path D:\GPT-SoVITS --port 19880")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
