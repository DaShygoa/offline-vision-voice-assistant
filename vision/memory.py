"""Visual Memory — SQLite metadata + NumPy embeddings + cosine similarity.

MVP deliberately uses NumPy (spec §9): course-project scale = tens of objects.
Swap-in FAISS later behind the same interface if needed.

Teach Once → Recognize Later:
  remember: multiple view samples -> L2-normalized DINOv2 embeddings
  identify/search: query embedding vs all stored embeddings (max over samples)
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


class VisualMemory:
    def __init__(self, cfg: dict):
        self.db_path = cfg.get("db_path", "data/visual_memory.db")
        self.samples_dir = cfg.get("samples_dir", "data/samples")
        self.embeddings_path = cfg.get("embeddings_path", "data/embeddings.npz")
        self.threshold = float(cfg.get("match_threshold", 0.75))
        self.topk = int(cfg.get("topk", 3))
        self.max_samples = int(cfg.get("max_samples", 10))

        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        os.makedirs(self.samples_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.embeddings_path) or ".", exist_ok=True)

        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_db()

        # in-memory index: matrix (N, D) float32 L2-normalized, labels parallel array
        self._matrix: np.ndarray = np.zeros((0, 0), dtype=np.float32)
        self._row_object_id: List[int] = []
        self._load_index()

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
                    self._matrix = data["matrix"].astype(np.float32)
                    self._row_object_id = [int(x) for x in data["object_ids"]]
                    return
                except Exception as e:
                    log.warning("Failed to load embeddings index: %s", e)
            self._rebuild_index()

    def _rebuild_index(self) -> None:
        with self._lock:
            rows: List[np.ndarray] = []
            ids: List[int] = []
            cur = self._conn.cursor()
            cur.execute(
                "SELECT s.embedding_row, s.object_id, s.image_path FROM object_samples s ORDER BY s.id"
            )
            for r in cur.fetchall():
                path = r["image_path"]
                emb_path = path + ".npy" if path else None
                if emb_path and os.path.isfile(emb_path):
                    try:
                        e = np.load(emb_path).astype(np.float32).reshape(-1)
                        n = np.linalg.norm(e)
                        if n > 0:
                            e = e / n
                        rows.append(e)
                        ids.append(int(r["object_id"]))
                    except Exception as ex:
                        log.warning("skip sample %s: %s", emb_path, ex)
            if rows:
                self._matrix = np.stack(rows, axis=0).astype(np.float32)
                self._row_object_id = ids
            else:
                # unknown dim — will expand on first add
                self._matrix = np.zeros((0, 0), dtype=np.float32)
                self._row_object_id = []
            self._save_index()

    def _save_index(self) -> None:
        with self._lock:
            np.savez_compressed(
                self.embeddings_path,
                matrix=self._matrix if self._matrix.size else np.zeros((0, 1), np.float32),
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
        """Create (or replace) object with N view embeddings."""
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
                # remove old rows from matrix
                keep = [i for i, x in enumerate(self._row_object_id) if x != oid]
                if keep:
                    self._matrix = self._matrix[keep]
                    self._row_object_id = [self._row_object_id[i] for i in keep]
                else:
                    self._matrix = np.zeros((0, embeddings[0].reshape(-1).shape[0]), np.float32)
                    self._row_object_id = []
            else:
                cur.execute(
                    "INSERT INTO objects (user_name, generic_class, created_at, updated_at) VALUES (?,?,?,?)",
                    (user_name, generic_class, now, now),
                )
                oid = int(cur.lastrowid)

            dim = embeddings[0].reshape(-1).shape[0]
            if self._matrix.size == 0:
                self._matrix = np.zeros((0, dim), dtype=np.float32)

            start_row = self._matrix.shape[0]
            new_vecs = []
            for i, emb in enumerate(embeddings):
                e = np.asarray(emb, dtype=np.float32).reshape(-1)
                n = np.linalg.norm(e)
                if n > 0:
                    e = e / n
                new_vecs.append(e)
                img_path = sample_images_paths[i] if i < len(sample_images_paths) else None
                # persist embedding next to image
                if img_path:
                    os.makedirs(os.path.dirname(img_path) or ".", exist_ok=True)
                    np.save(img_path + ".npy", e)
                cur.execute(
                    "INSERT INTO object_samples (object_id, image_path, embedding_row, created_at) VALUES (?,?,?,?)",
                    (oid, img_path, start_row + i, now),
                )
            if new_vecs:
                mat = np.stack(new_vecs, axis=0)
                if self._matrix.shape[1] != mat.shape[1]:
                    # dimension mismatch — rebuild from disk
                    self._conn.commit()
                    self._rebuild_index()
                    return self.get_object(oid) or MemoryObject(oid, user_name, generic_class, now)
                self._matrix = np.concatenate([self._matrix, mat], axis=0) if self._matrix.size else mat
                self._row_object_id.extend([oid] * len(new_vecs))
            self._conn.commit()
            self._save_index()
            log.info("Remembered '%s' with %d samples (id=%d)", user_name, len(embeddings), oid)
            return MemoryObject(oid, user_name, generic_class, now, sample_count=len(embeddings))

    # ------------------------------------------------------------- query
    def query(self, embedding: np.ndarray, topk: Optional[int] = None) -> List[MemoryMatch]:
        """Return matches with score >= threshold (or topk if provided)."""
        with self._lock:
            if self._matrix.size == 0 or embedding is None:
                return []
            q = np.asarray(embedding, dtype=np.float32).reshape(-1)
            n = np.linalg.norm(q)
            if n > 0:
                q = q / n
            if q.shape[0] != self._matrix.shape[1]:
                return []
            scores = self._matrix @ q  # cosine since L2-normalized
            k = topk or self.topk
            if scores.shape[0] <= k:
                idx = np.argsort(-scores)
            else:
                idx = np.argpartition(-scores, k)[:k]
                idx = idx[np.argsort(-scores[idx])]

            # aggregate by object_id: max score per object
            best: dict[int, float] = {}
            for i in idx:
                oid = int(self._row_object_id[int(i)])
                s = float(scores[int(i)])
                if oid not in best or s > best[oid]:
                    best[oid] = s
            ranked = sorted(best.items(), key=lambda kv: -kv[1])
            out: List[MemoryMatch] = []
            names = self._names_for_ids([oid for oid, _ in ranked])
            for rank, (oid, s) in enumerate(ranked):
                if s < self.threshold:
                    continue
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

    def best_match(self, embedding: np.ndarray) -> Optional[MemoryMatch]:
        m = self.query(embedding, topk=1)
        return m[0] if m else None

    def match_detections(
        self, embed_list: Sequence[Optional[np.ndarray]]
    ) -> Optional[tuple[int, MemoryMatch]]:
        """Return (index in embed_list, match) for the globally best score above threshold."""
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
        return [MemoryObject(int(r["id"]), r["user_name"], r["generic_class"], r["created_at"], int(r["sc"])) for r in rows]

    def find_by_name(self, name: str) -> Optional[MemoryObject]:
        r = self._conn.execute("SELECT id FROM objects WHERE user_name = ?", (name,)).fetchone()
        if not r:
            # fuzzy contains
            r = self._conn.execute(
                "SELECT id, user_name FROM objects WHERE ? LIKE '%' || user_name || '%' OR user_name LIKE '%' || ? || '%'",
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
            keep = [i for i, x in enumerate(self._row_object_id) if x != oid]
            if self._matrix.size:
                if keep:
                    self._matrix = self._matrix[keep]
                else:
                    self._matrix = np.zeros((0, self._matrix.shape[1]), np.float32)
            self._row_object_id = [self._row_object_id[i] for i in keep]
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
        }

    def close(self) -> None:
        with self._lock:
            self._conn.close()
