"""Object feature library — SQLite metadata + NumPy embeddings + cosine retrieval.

选题一 core: store ResNet18 features per object, rank by cosine similarity,
declare "unknown" when the best score < match_threshold.

  remember : name + N image embeddings → feature bank
  identify : query embedding → ranked matches (score = cosine ∈ [-1, 1])
  unknown  : best score < threshold → 未知物体

MVP uses NumPy (course-project scale = tens of objects), no FAISS.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class MemoryMatch:
    object_id: int
    user_name: str
    generic_class: Optional[str]
    score: float
    rank: int
    raw_score: Optional[float] = None
    centered: bool = False


@dataclass
class MemoryObject:
    object_id: int
    user_name: str
    generic_class: Optional[str]
    created_at: str
    sample_count: int = 0


def _l2(x: np.ndarray) -> np.ndarray:
    a = np.asarray(x, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(a))
    return a / n if n > 1e-12 else a


class VisualMemory:
    def __init__(self, cfg: dict):
        self.db_path = cfg.get("db_path", "data/visual_memory.db")
        self.samples_dir = cfg.get("samples_dir", "data/samples")
        self.embeddings_path = cfg.get("embeddings_path", "data/resnet18_embeddings.npz")
        self._threshold = float(cfg.get("match_threshold", 0.65))
        # A single-class bank has no competing class for a margin check.
        self._single_threshold = float(cfg.get("single_match_threshold", self._threshold))
        # Common-component removal. ResNet18 avgpool features are ~87% "common
        # component" (lighting / texture / background statistics shared by every
        # photo), so two completely different objects still score ~0.6 cosine.
        # Subtracting the bank mean before the cosine drops cross-object scores to
        # ~-0.7 while keeping same-object scores intact.
        self.center_features = bool(cfg.get("center_features", True))
        # Scoring scale differs once features are centred (same-object ~0.5-0.9,
        # cross-object ~-0.7), so the threshold must be re-calibrated.
        self._threshold_centered = float(cfg.get("match_threshold_centered", 0.30))
        self._margin_threshold = float(cfg.get("margin_threshold", 0.05))
        self._margin_threshold_centered = float(cfg.get("margin_threshold_centered", 0.15))
        self.topk = int(cfg.get("topk", 3))
        self.max_samples = int(cfg.get("max_samples", 10))
        self.embed_dim = int(cfg.get("embed_dim", 512))

        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        os.makedirs(self.samples_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.embeddings_path) or ".", exist_ok=True)

        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_db()

        # in-memory cosine index: rows = L2-normalized embeddings
        self._matrix: np.ndarray = np.zeros((0, 0), dtype=np.float32)
        # common-component-removed view of the bank (rebuilt when the bank changes)
        self._centered_matrix: np.ndarray = np.zeros((0, 0), dtype=np.float32)
        self._mean_vec: np.ndarray = np.zeros((0,), dtype=np.float32)
        self._centered_row_ids: List[int] = []
        self._row_object_id: List[int] = []
        self._load_index()
        self._purge_featureless_objects()

    def _purge_featureless_objects(self) -> None:
        """Drop objects with zero usable embeddings (e.g. stale DINO-era rows)."""
        with self._lock:
            bank_ids = set(self._row_object_id)
            stale = []
            for o in self.list_objects():
                if o.object_id not in bank_ids:
                    stale.append(o)
            for o in stale:
                log.info("purge featureless object id=%s name=%r", o.object_id, o.user_name)
                self._conn.execute("DELETE FROM object_samples WHERE object_id = ?", (o.object_id,))
                self._conn.execute("DELETE FROM objects WHERE id = ?", (o.object_id,))
            if stale:
                self._conn.commit()

    # ------------------------------------------------------------------- db
    def _init_db(self) -> None:
        cur = self._conn.cursor()
        cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS objects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_name TEXT NOT NULL UNIQUE,
                generic_class TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS object_samples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                object_id INTEGER NOT NULL,
                image_path TEXT,
                embedding_row INTEGER,
                created_at TEXT NOT NULL,
                FOREIGN KEY(object_id) REFERENCES objects(id) ON DELETE CASCADE
            );
            """
        )
        self._conn.commit()

    # ---------------------------------------------------------------- index
    def _load_index(self) -> None:
        with self._lock:
            if os.path.isfile(self.embeddings_path):
                try:
                    data = np.load(self.embeddings_path, allow_pickle=False)
                    mat = data["matrix"].astype(np.float32)
                    ids = [int(x) for x in data["object_ids"]]
                    if mat.size and mat.shape[1] == self.embed_dim and len(ids) == mat.shape[0]:
                        self._matrix = mat
                        self._row_object_id = ids
                        self._refresh_centered()
                        return
                    log.info("Stale feature bank (dim mismatch), rebuilding …")
                except Exception as e:
                    log.warning("Failed to load feature bank: %s", e)
            self._rebuild_index()

    def _rebuild_index(self) -> None:
        with self._lock:
            rows: List[np.ndarray] = []
            ids: List[int] = []
            cur = self._conn.cursor()
            cur.execute(
                "SELECT s.object_id, s.image_path FROM object_samples s ORDER BY s.id"
            )
            for r in cur.fetchall():
                path = r["image_path"]
                oid = int(r["object_id"])
                if not path:
                    continue
                p = path + ".npy"
                if not os.path.isfile(p):
                    continue
                try:
                    e = np.load(p).astype(np.float32).reshape(-1)
                    if e.shape[0] != self.embed_dim:
                        continue  # skip stale DINO/old features
                    e = _l2(e)
                    if float(np.linalg.norm(e)) > 0:
                        rows.append(e)
                        ids.append(oid)
                except Exception as ex:
                    log.warning("skip sample %s: %s", p, ex)
            self._matrix = (
                np.stack(rows, axis=0).astype(np.float32)
                if rows
                else np.zeros((0, 0), np.float32)
            )
            self._row_object_id = ids
            self._save_index()
            self._refresh_centered()

    def _save_index(self) -> None:
        with self._lock:
            mat = self._matrix if self._matrix.size else np.zeros((0, 1), np.float32)
            np.savez_compressed(
                self.embeddings_path,
                matrix=mat,
                object_ids=np.array(self._row_object_id, dtype=np.int64),
            )

    # ------------------------------------------------- effective thresholds
    @property
    def centering_active(self) -> bool:
        """True when matching runs on the common-component-removed bank."""
        return bool(getattr(self, "_centered_matrix", np.zeros((0, 0))).size)

    @property
    def threshold(self) -> float:
        """Score threshold for the space currently in use."""
        if self.centering_active:
            return self._threshold_centered
        if len(set(self._row_object_id)) == 1:
            return self._single_threshold
        return self._threshold

    @property
    def margin_threshold(self) -> float:
        """top-1 must beat top-2 by this much to be called 'sure'."""
        if self.centering_active:
            return self._margin_threshold_centered
        return self._margin_threshold

    # ------------------------------------------------- common-component removal
    def _refresh_centered(self) -> None:
        """Recompute the centred bank used for matching.

        ResNet18 avgpool features are dominated by a shared "common component"
        (lighting, texture and background statistics). Measured on this project:
        87% of the 512-d vector is common to every photo, so two unrelated objects
        still score ~0.6 cosine. Subtracting the bank mean before the cosine
        separates them (cross-object drops to ~-0.7, same-object stays ~0.5-0.9).

        Guard: needs at least two distinct objects in the bank. With a single
        object the mean IS that object's class mean, which collapses intra-class
        similarity (measured 0.85 -> 0.03) and breaks recognition outright.
        """
        with self._lock:
            self._centered_matrix = np.zeros((0, 0), dtype=np.float32)
            self._mean_vec = np.zeros((0,), dtype=np.float32)
            if not self.center_features or self._matrix.size == 0:
                return
            distinct = len(set(self._row_object_id))
            if distinct < 2:
                return
            mean = self._matrix.mean(axis=0)
            # NOTE: _l2() flattens with reshape(-1) — it only works on 1-D
            # vectors. The bank must be normalised row-wise.
            centered = self._matrix - mean
            norms = np.linalg.norm(centered, axis=1, keepdims=True)
            centered = centered / np.maximum(norms, 1e-12)
            # Drop degenerate rows (a sample identical to the bank mean)
            keep = (norms[:, 0] > 1e-6)
            if not keep.any():
                return
            self._mean_vec = mean.astype(np.float32)
            self._centered_matrix = centered[keep].astype(np.float32)
            self._centered_row_ids = [oid for oid, k in zip(self._row_object_id, keep) if k]
            log.info("Centred feature bank: %d objects, %d samples, common component %.1f%%",
                     distinct, int(keep.sum()),
                     np.linalg.norm(mean) / max(np.linalg.norm(self._matrix, axis=1).mean(), 1e-8) * 100)

    def _match_space(self, embedding: Optional[np.ndarray]):
        """Return (matrix, row_ids, query_vector) in the space used for matching."""
        if self._centered_matrix.size:
            q = embedding - self._mean_vec
            return self._centered_matrix, self._centered_row_ids, _l2(q)
        return self._matrix, self._row_object_id, _l2(embedding)

    # ------------------------------------------------------------- remember
    def add_object(
        self,
        user_name: str,
        generic_class: Optional[str],
        embeddings: Sequence[np.ndarray],
        sample_images_paths: Sequence[str],
        replace_existing: bool = True,
    ) -> MemoryObject:
        """Create (or replace) object with N ResNet18 view embeddings."""
        if not embeddings:
            raise ValueError("add_object requires at least one embedding")
        with self._lock:
            now = time.strftime("%Y-%m-%d %H:%M:%S")
            cur = self._conn.cursor()
            cur.execute("SELECT id FROM objects WHERE user_name = ?", (user_name,))
            row = cur.fetchone()
            if row and not replace_existing:
                raise FileExistsError(f"object already exists: {user_name}")
            if row:
                oid = int(row["id"])
                cur.execute(
                    "UPDATE objects SET generic_class = ?, updated_at = ? WHERE id = ?",
                    (generic_class, now, oid),
                )
                cur.execute("DELETE FROM object_samples WHERE object_id = ?", (oid,))
                self._drop_object_rows(oid)
            else:
                cur.execute(
                    "INSERT INTO objects (user_name, generic_class, created_at, updated_at) VALUES (?,?,?,?)",
                    (user_name, generic_class, now, now),
                )
                oid = int(cur.lastrowid)

            new_rows: List[np.ndarray] = []
            for i, emb in enumerate(embeddings):
                e = _l2(emb)
                if e.shape[0] != self.embed_dim:
                    raise ValueError(
                        f"embedding dim {e.shape[0]} != expected {self.embed_dim}"
                    )
                img_path = sample_images_paths[i] if i < len(sample_images_paths) else None
                if img_path:
                    os.makedirs(os.path.dirname(img_path) or ".", exist_ok=True)
                    np.save(img_path + ".npy", e)
                row_i = int(self._matrix.shape[0]) + len(new_rows) if self._matrix.size else len(new_rows)
                new_rows.append(e)
                cur.execute(
                    "INSERT INTO object_samples (object_id, image_path, embedding_row, created_at) "
                    "VALUES (?,?,?,?)",
                    (oid, img_path, row_i, now),
                )

            mat = np.stack(new_rows, axis=0)
            if self._matrix.size == 0:
                self._matrix = mat
            elif self._matrix.shape[1] != mat.shape[1]:
                self._conn.commit()
                self._rebuild_index()
                return self.get_object(oid) or MemoryObject(oid, user_name, generic_class, now)
            else:
                self._matrix = np.concatenate([self._matrix, mat], axis=0)
            self._row_object_id.extend([oid] * len(new_rows))

            self._conn.commit()
            self._save_index()
            self._refresh_centered()
            log.info("Remembered '%s' with %d samples (id=%d)", user_name, len(embeddings), oid)
            return MemoryObject(oid, user_name, generic_class, now, sample_count=len(embeddings))

    def _drop_object_rows(self, oid: int) -> None:
        keep = [i for i, x in enumerate(self._row_object_id) if x != oid]
        if keep:
            self._matrix = self._matrix[keep]
            self._row_object_id = [self._row_object_id[i] for i in keep]
        else:
            dim = self._matrix.shape[1] if self._matrix.size else 0
            self._matrix = np.zeros((0, dim), np.float32)
            self._row_object_id = []

    # ------------------------------------------------------------- query
    def query(
        self,
        embedding: Optional[np.ndarray],
        topk: Optional[int] = None,
    ) -> List[MemoryMatch]:
        """Ranked cosine matches (best score first). Caller applies threshold for unknown."""
        with self._lock:
            if embedding is None or self._matrix.size == 0 or not self._row_object_id:
                return []
            mat, row_ids, qq = self._match_space(embedding)
            if qq.shape[0] != mat.shape[1]:
                log.warning("query dim %d != bank dim %d", qq.shape[0], mat.shape[1])
                return []
            scores = mat @ qq
            # Expose the same raw cosine scale regardless of whether ranking
            # uses a bank-centered space. Compute per object independently.
            raw_scores = self._matrix @ _l2(embedding)
            raw_best: dict[int, float] = {}
            for oid, score in zip(self._row_object_id, raw_scores):
                oid = int(oid)
                raw_best[oid] = max(raw_best.get(oid, -float('inf')), float(score))
            best: dict[int, float] = {}
            for i, s in enumerate(scores):
                oid = int(row_ids[i])
                fs = float(s)
                if oid not in best or fs > best[oid]:
                    best[oid] = fs
            ranked = sorted(best.items(), key=lambda kv: -kv[1])
            k = topk or self.topk
            ranked = ranked[: max(k, 1)]
            names = self._names_for_ids([oid for oid, _ in ranked])
            out: List[MemoryMatch] = []
            for rank, (oid, s) in enumerate(ranked):
                info = names.get(oid)
                out.append(
                    MemoryMatch(
                        object_id=oid,
                        user_name=info[0] if info else f"obj_{oid}",
                        generic_class=info[1] if info else None,
                        score=s,
                        rank=rank,
                        raw_score=raw_best[oid],
                        centered=self.centering_active,
                    )
                )
            return out

    def sample_images(self, user_name: str) -> List[str]:
        """Image paths of the enrolled samples for an object (may be empty on disk)."""
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                "SELECT s.image_path FROM object_samples s "
                "JOIN objects o ON o.id = s.object_id "
                "WHERE o.user_name = ? ORDER BY s.id",
                (user_name,),
            )
            out = []
            for r in cur.fetchall():
                p = r["image_path"]
                if p and os.path.isfile(p):
                    out.append(p)
            return out

    def best_match(self, embedding: Optional[np.ndarray] = None) -> Optional[MemoryMatch]:
        """Top-1 match regardless of threshold (caller decides known vs unknown)."""
        m = self.query(embedding, topk=1)
        return m[0] if m else None

    def top_k(self, embedding: Optional[np.ndarray], k: int = 3) -> List[MemoryMatch]:
        """Top-k matches for debugging / ranking display."""
        return self.query(embedding, topk=max(1, int(k))) or []

    def match_detections(
        self,
        embed_list: Sequence[Optional[np.ndarray]],
    ) -> Optional[tuple[int, MemoryMatch]]:
        """Return (index, match) for the globally best score (threshold applied by caller)."""
        best_i = -1
        best_m: Optional[MemoryMatch] = None
        for i, e in enumerate(embed_list):
            if e is None:
                continue
            m = self.best_match(e)
            if m and (best_m is None or m.score > best_m.score):
                best_m = m
                best_i = i
        if best_m is None:
            return None
        return best_i, best_m

    def _names_for_ids(self, ids: Sequence[int]) -> dict[int, tuple[str, Optional[str]]]:
        if not ids:
            return {}
        qmarks = ",".join("?" for _ in ids)
        cur = self._conn.execute(
            f"SELECT id, user_name, generic_class FROM objects WHERE id IN ({qmarks})",
            list(ids),
        )
        return {int(r["id"]): (r["user_name"], r["generic_class"]) for r in cur.fetchall()}

    # ------------------------------------------------------------ manage
    def get_object(self, oid: int) -> Optional[MemoryObject]:
        r = self._conn.execute(
            "SELECT o.id, o.user_name, o.generic_class, o.created_at, COUNT(s.id) as sc "
            "FROM objects o LEFT JOIN object_samples s ON s.object_id = o.id WHERE o.id = ? GROUP BY o.id",
            (oid,),
        ).fetchone()
        if not r:
            return None
        return MemoryObject(int(r["id"]), r["user_name"], r["generic_class"], r["created_at"], int(r["sc"]))

    def list_objects(self) -> List[MemoryObject]:
        rows = self._conn.execute(
            "SELECT o.id, o.user_name, o.generic_class, o.created_at, COUNT(s.id) as sc "
            "FROM objects o LEFT JOIN object_samples s ON s.object_id = o.id GROUP BY o.id ORDER BY o.id"
        ).fetchall()
        return [
            MemoryObject(int(r["id"]), r["user_name"], r["generic_class"], r["created_at"], int(r["sc"]))
            for r in rows
        ]

    def find_by_name(self, name: str) -> Optional[MemoryObject]:
        r = self._conn.execute("SELECT id FROM objects WHERE user_name = ?", (name,)).fetchone()
        if not r:
            r = self._conn.execute(
                "SELECT id, user_name FROM objects WHERE ? LIKE '%' || user_name || '%' "
                "OR user_name LIKE '%' || ? || '%'",
                (name, name),
            ).fetchone()
        if not r:
            return None
        return self.get_object(int(r["id"]))

    def rename_object(self, old_name: str, new_name: str) -> bool:
        """Rename an object. Returns False if old_name missing or new_name taken.

        Uses EXACT name matching — find_by_name() is fuzzy (LIKE), which would
        treat "牛奶瓶" as already taken when "这是牛奶瓶" exists.
        """
        new_name = (new_name or "").strip()
        if not new_name:
            return False
        with self._lock:
            row = self._conn.execute(
                "SELECT id FROM objects WHERE user_name = ?", (old_name,)
            ).fetchone()
            if not row:
                return False
            oid = int(row["id"])
            dup = self._conn.execute(
                "SELECT id FROM objects WHERE user_name = ? AND id != ?",
                (new_name, oid),
            ).fetchone()
            if dup:
                return False  # exact name already exists
            self._conn.execute(
                "UPDATE objects SET user_name = ?, updated_at = ? WHERE id = ?",
                (new_name, time.strftime("%Y-%m-%d %H:%M:%S"), oid),
            )
            self._conn.commit()
            log.info("Renamed object %r -> %r (id=%d)", old_name, new_name, oid)
            return True

    def delete_object(self, name_or_id) -> bool:
        with self._lock:
            if isinstance(name_or_id, str):
                obj = self.find_by_name(name_or_id)
                if not obj:
                    return False
                oid = obj.object_id
            else:
                oid = int(name_or_id)
            self._conn.execute("DELETE FROM object_samples WHERE object_id = ?", (oid,))
            self._conn.execute("DELETE FROM objects WHERE id = ?", (oid,))
            self._conn.commit()
            self._drop_object_rows(oid)
            self._save_index()
            self._refresh_centered()
            return True

    def stats(self) -> dict:
        n_obj = self._conn.execute("SELECT COUNT(*) c FROM objects").fetchone()["c"]
        n_smp = self._conn.execute("SELECT COUNT(*) c FROM object_samples").fetchone()["c"]
        return {
            "objects": int(n_obj),
            "samples": int(n_smp),
            "threshold": self.threshold,
            "matrix_shape": tuple(self._matrix.shape),
            "embed_dim": self.embed_dim,
        }

    def close(self) -> None:
        with self._lock:
            self._conn.close()
