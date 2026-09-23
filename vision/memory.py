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
        self.threshold = float(cfg.get("match_threshold", 0.65))
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

    def _save_index(self) -> None:
        with self._lock:
            mat = self._matrix if self._matrix.size else np.zeros((0, 1), np.float32)
            np.savez_compressed(
                self.embeddings_path,
                matrix=mat,
                object_ids=np.array(self._row_object_id, dtype=np.int64),
            )

    # ------------------------------------------------------------- remember
    def add_object(
        self,
        user_name: str,
        generic_class: Optional[str],
        embeddings: Sequence[np.ndarray],
        sample_images_paths: Sequence[str],
    ) -> MemoryObject:
        """Create (or replace) object with N ResNet18 view embeddings."""
        if not embeddings:
            raise ValueError("add_object requires at least one embedding")
        with self._lock:
            now = time.strftime("%Y-%m-%d %H:%M:%S")
            cur = self._conn.cursor()
            cur.execute("SELECT id FROM objects WHERE user_name = ?", (user_name,))
            row = cur.fetchone()
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
            qq = _l2(embedding)
            if qq.shape[0] != self._matrix.shape[1]:
                log.warning("query dim %d != bank dim %d", qq.shape[0], self._matrix.shape[1])
                return []
            scores = self._matrix @ qq
            best: dict[int, float] = {}
            for i, s in enumerate(scores):
                oid = int(self._row_object_id[i])
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
                    )
                )
            return out

    def best_match(self, embedding: Optional[np.ndarray] = None) -> Optional[MemoryMatch]:
        """Top-1 match regardless of threshold (caller decides known vs unknown)."""
        m = self.query(embedding, topk=1)
        return m[0] if m else None

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
