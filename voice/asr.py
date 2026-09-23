"""Offline Chinese ASR — Paraformer (CPU INT8) via ONNX Runtime.

Supports two model layouts:
  A) End-to-end FunASR Paraformer ONNX (model.int8.onnx):
       inputs : speech [B, T, 560], speech_lengths [B]
       outputs: logits [B, L, vocab], token_num, ...
     Feature pipeline: kaldi_native_fbank 80-dim + LFR7 + CMVN (am.mvn)
  B) sherpa-onnx OfflineParaformer (encoder.int8.onnx + decoder.int8.onnx)
     used automatically when those files exist.

VAD is handled separately (voice/vad.py). This module only decodes utterances.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Dict, List, Optional

import numpy as np

log = logging.getLogger(__name__)


def _load_cmvn_k2(path: str):
    """Parsing from k2-fsa sherpa-onnx paraformer test script (am.mvn)."""
    neg_mean = None
    inv_std = None
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.startswith("<LearnRateCoef>"):
                continue
            t = line.split()[3:-1]
            vals = [float(x) for x in t]
            if neg_mean is None:
                neg_mean = np.array(vals, dtype=np.float32)
            else:
                inv_std = np.array(vals, dtype=np.float32)
    if inv_std is None and neg_mean is not None:
        inv_std = np.ones_like(neg_mean)
    return neg_mean, inv_std


class ParaformerASR:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.sample_rate = int(cfg.get("sample_rate", 16000))
        self.num_threads = int(cfg.get("num_threads", 4))
        self.ready = False
        self.backend = "none"
        self.last_latency_ms = 0.0

        self._sess = None
        self._tokens: Dict[int, str] = {}
        self._neg_mean = None
        self._inv_std = None
        self._sherpa = None  # alternative path

    # ------------------------------------------------------------------ load
    def load(self) -> None:
        encoder = self.cfg.get("encoder", "")
        decoder = self.cfg.get("decoder", "")
        e2e = self.cfg.get("e2e_model") or encoder
        tokens_path = self.cfg.get("tokens", "")
        model_dir = self.cfg.get("model_dir", os.path.dirname(e2e or ""))

        # Path B: classic sherpa encoder+decoder pair
        if (
            encoder
            and decoder
            and os.path.isfile(encoder)
            and os.path.isfile(decoder)
            and os.path.abspath(encoder) != os.path.abspath(decoder)
            and os.path.basename(encoder) in ("encoder.int8.onnx", "encoder.onnx", "model_encoder.int8.onnx")
        ):
            self._load_sherpa(encoder, decoder, tokens_path)
            return

        # Path A: end-to-end model.int8.onnx (FunASR paraformer)
        candidates = [
            e2e,
            os.path.join(model_dir, "model.int8.onnx") if model_dir else "",
            os.path.join(model_dir, "model.onnx") if model_dir else "",
            encoder,
        ]
        path = next((c for c in candidates if c and os.path.isfile(c)), None)
        if not path:
            raise FileNotFoundError(f"Paraformer model not found under {model_dir}")

        # tokens
        if not tokens_path or not os.path.isfile(tokens_path):
            for cand in (
                os.path.join(model_dir, "tokens.txt") if model_dir else "",
                os.path.join(os.path.dirname(path), "tokens.txt"),
            ):
                if cand and os.path.isfile(cand):
                    tokens_path = cand
                    break
        if not tokens_path or not os.path.isfile(tokens_path):
            raise FileNotFoundError(f"tokens.txt not found for ASR ({tokens_path})")
        self._tokens = self._load_tokens(tokens_path)

        # cmvn
        cmvn = os.path.join(model_dir, "am.mvn") if model_dir else ""
        if not os.path.isfile(cmvn):
            alt = os.path.join(os.path.dirname(path), "am.mvn")
            cmvn = alt if os.path.isfile(alt) else ""
        if cmvn:
            self._neg_mean, self._inv_std = _load_cmvn_k2(cmvn)
        else:
            log.warning("am.mvn not found — CMVN disabled")
            self._neg_mean = self._inv_std = None

        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = self.num_threads
        so.inter_op_num_threads = max(1, self.num_threads // 2)
        so.log_severity_level = 3
        self._sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        self.backend = f"paraformer_e2e_cpu_int8:{os.path.basename(path)}"
        self.ready = True
        log.info("Paraformer ASR ready: %s tokens=%d", path, len(self._tokens))

    def _load_sherpa(self, encoder: str, decoder: str, tokens: str) -> None:
        import sherpa_onnx

        self._sherpa = sherpa_onnx.OfflineRecognizer.from_paraformer(
            tokens=tokens,
            encoder=encoder,
            decoder=decoder,
            num_threads=self.num_threads,
            sample_rate=self.sample_rate,
            feature_dim=80,
            decoding_method="greedy_search",
        )
        self.backend = "sherpa_paraformer"
        self.ready = True
        log.info("Paraformer ASR ready via sherpa-onnx")

    @staticmethod
    def _load_tokens(path: str) -> Dict[int, str]:
        ans: Dict[int, str] = {}
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                ans[i] = line.strip().split()[0]
        return ans

    # -------------------------------------------------------------- features
    def _features(self, samples: np.ndarray) -> np.ndarray:
        """float32 mono [-1,1] → LFR features (T', 560)."""
        import kaldi_native_fbank as knf

        # int16 scale for fbank accept_waveform list API
        if samples.dtype != np.float32:
            samples = samples.astype(np.float32)
        # clip
        pcm = np.clip(samples, -1.0, 1.0)
        opts = knf.FbankOptions()
        opts.frame_opts.dither = 0
        opts.frame_opts.snip_edges = False
        opts.frame_opts.samp_freq = self.sample_rate
        opts.mel_opts.num_bins = 80
        fbank = knf.OnlineFbank(opts)
        fbank.accept_waveform(self.sample_rate, (pcm * 32768.0).tolist())
        fbank.input_finished()
        n = fbank.num_frames_ready
        if n <= 0:
            return np.zeros((0, 80), dtype=np.float32)
        feats = np.stack([fbank.get_frame(i) for i in range(n)]).astype(np.float32)

        # LFR 7/6
        window_size = 7
        window_shift = 6
        if feats.shape[0] < window_size:
            # pad
            pad = np.repeat(feats[-1:], window_size - feats.shape[0], axis=0) if feats.shape[0] else np.zeros((window_size, 80), np.float32)
            feats = np.concatenate([feats, pad], axis=0)
        T = (feats.shape[0] - window_size) // window_shift + 1
        lfr = np.lib.stride_tricks.as_strided(
            feats,
            shape=(T, feats.shape[1] * window_size),
            strides=((window_shift * feats.shape[1]) * feats.itemsize, feats.itemsize),
        ).copy()
        if self._neg_mean is not None and self._inv_std is not None:
            if self._neg_mean.shape[0] == lfr.shape[1]:
                lfr = (lfr + self._neg_mean) * self._inv_std
        return lfr

    # ---------------------------------------------------------------- decode
    def transcribe(self, pcm: np.ndarray) -> str:
        if not self.ready:
            return ""
        samples = np.asarray(pcm)
        if samples.dtype == np.int16:
            samples = samples.astype(np.float32) / 32768.0
        else:
            samples = samples.astype(np.float32)
        if samples.size < self.sample_rate // 10:
            return ""
        # mono
        if samples.ndim > 1:
            samples = samples.mean(axis=1)

        t0 = time.perf_counter()
        try:
            if self._sherpa is not None:
                text = self._transcribe_sherpa(samples)
            else:
                text = self._transcribe_e2e(samples)
        except Exception as e:
            log.error("ASR failed: %s", e)
            return ""
        self.last_latency_ms = (time.perf_counter() - t0) * 1000.0
        return text

    def _transcribe_sherpa(self, samples: np.ndarray) -> str:
        stream = self._sherpa.create_stream()
        stream.accept_waveform(self.sample_rate, samples)
        self._sherpa.decode(stream)
        return stream.result.text.strip()

    def _transcribe_e2e(self, samples: np.ndarray) -> str:
        feats = self._features(samples)
        if feats.shape[0] == 0:
            return ""
        speech = np.expand_dims(feats, axis=0)  # (1, T, 560)
        speech_lengths = np.array([speech.shape[1]], dtype=np.int32)
        outputs = self._sess.run(
            ["logits", "token_num"],
            {"speech": speech, "speech_lengths": speech_lengths},
        )
        logits = outputs[0][0]  # (L, V)
        n = int(np.asarray(outputs[1]).reshape(-1)[0])
        if n <= 0 or logits.shape[0] == 0:
            return ""
        n = min(n, logits.shape[0])
        ids = logits[:n].argmax(axis=-1)
        parts: List[str] = []
        for i in ids:
            i = int(i)
            if i in (0, 2):  # blank / sos style
                continue
            tok = self._tokens.get(i)
            if not tok:
                continue
            # bilingual BPE: "ye@@ster@@day" → join without space
            if tok.endswith("@@"):
                parts.append(tok[:-2])
            else:
                parts.append(tok)
                if tok.isascii() and tok[-1].isalnum():
                    parts.append(" ")
        text = "".join(parts)
        import re

        text = re.sub(r"\s+", " ", text).strip()
        text = re.sub(r"\s+([，。！？、；：,.!?])", r"\1", text)
        text = re.sub(r"([，。！？、；：])\s+", r"\1", text)
        return text

    def transcribe_file(self, path: str) -> str:
        import soundfile as sf

        data, sr = sf.read(path, dtype="float32", always_2d=False)
        if data.ndim > 1:
            data = data.mean(axis=1)
        if sr != self.sample_rate:
            n_out = int(len(data) * self.sample_rate / sr)
            idx = np.linspace(0, len(data) - 1, n_out)
            data = np.interp(idx, np.arange(len(data)), data).astype(np.float32)
        return self.transcribe(data)
