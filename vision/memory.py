"""Visual Memory — SQLite metadata + NumPy embeddings + cosine similarity.

MVP deliberately uses NumPy (spec §9): course-project scale = tens of objects.

Feature channels (side-by-side matrices, never concatenated into one dim):
  * dino  : DINOv2 384-d appearance (primary)
  * shape : OpenCV silhouette / Hu / Fourier (optional)
  * depth : monocular depth signature (optional)

Match score = weighted average of available channels (renormalized if a channel
is missing on query or store) → same threshold still usable after calibration.

Teach Once → Recognize Later:
  remember: multiple view samples -> L2-normalized multi-channel vectors
  identify/search: query vectors vs all stored (max over samples per object)
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
    # optional channel breakdown for debugging / UI
    score_dino: Optional[float] = None
    score_shape: Optional[float] = None
    score_depth: Optional[float] = None


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
        self.embeddings_path = cfg.get("embeddings_path", "data/embeddings.npz")
        self.threshold = float(cfg.get("match_threshold", 0.75))
        self.topk = int(cfg.get("topk", 3))
        self.max_samples = int(cfg.get("max_samples", 10))
        # channel fusion weights (renormalized over available channels)
        self.use_shape = bool(cfg.get("use_shape", True))
        self.use_depth = bool(cfg.get("use_depth", True))
        self.w_dino = float(cfg.get("weight_dino", 0.55))
        self.w_shape = float(cfg.get("weight_shape", 0.30))
        self.w_depth = float(cfg.get("weight_depth", 0.15))

        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        os.makedirs(self.samples_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.embeddings_path) or ".", exist_ok=True)

        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_db()

        # in-memory indices — parallel row order with DINO when sample has all channels
        # Each side matrix has its own object_ids (rows may be missing for some samples)
        self._matrix: np.ndarray = np.zeros((0, 0), dtype=np.float32)  # dino
        self._row_object_id: List[int] = []
        self._shape_matrix: np.ndarray = np.zeros((0, 0), dtype=np.float32)
        self._shape_row_object_id: List[int] = []
        self._depth_matrix: np.ndarray = np.zeros((0, 0), dtype=np.float32)
        self._depth_row_object_id: List[int] = []
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
                shape_row INTEGER,
                depth_row INTEGER,
                created_at TEXT NOT NULL,
                FOREIGN KEY(object_id) REFERENCES objects(id) ON DELETE CASCADE
            );
            """
        )
        # migrate older DBs missing shape/depth columns
        cols = {r["name"] for r in cur.execute("PRAGMA table_info(object_samples)").fetchall()}
        for name in ("shape_row", "depth_row"):
            if name not in cols:
                try:
                    cur.execute(f"ALTER TABLE object_samples ADD COLUMN {name} INTEGER")
                except sqlite3.OperationalError:
                    pass
        self._conn.commit()

    # ---------------------------------------------------------------- index
    def _load_index(self) -> None:
        with self._lock:
            if os.path.isfile(self.embeddings_path):
                try:
                    data = np.load(self.embeddings_path, allow_pickle=False)
                    self._matrix = data["matrix"].astype(np.float32)
                    self._row_object_id = [int(x) for x in data["object_ids"]]
                    if "shape_matrix" in data.files:
                        self._shape_matrix = data["shape_matrix"].astype(np.float32)
                        self._shape_row_object_id = [int(x) for x in data["shape_object_ids"]]
                    if "depth_matrix" in data.files:
                        self._depth_matrix = data["depth_matrix"].astype(np.float32)
                        self._depth_row_object_id = [int(x) for x in data["depth_object_ids"]]
                    return
                except Exception as e:
                    log.warning("Failed to load embeddings index: %s", e)
            self._rebuild_index()

    def _rebuild_index(self) -> None:
        with self._lock:
            dino_rows: List[np.ndarray] = []
            dino_ids: List[int] = []
            shape_rows: List[np.ndarray] = []
            shape_ids: List[int] = []
            depth_rows: List[np.ndarray] = []
            depth_ids: List[int] = []
            cur = self._conn.cursor()
            cur.execute(
                "SELECT s.object_id, s.image_path FROM object_samples s ORDER BY s.id"
            )
            for r in cur.fetchall():
                path = r["image_path"]
                oid = int(r["object_id"])
                if not path:
                    continue
                for side, rows, ids in (
                    (".npy", dino_rows, dino_ids),
                    (".shape.npy", shape_rows, shape_ids),
                    (".depth.npy", depth_rows, depth_ids),
                ):
                    p = path + side
                    if os.path.isfile(p):
                        try:
                            e = np.load(p).astype(np.float32).reshape(-1)
                            e = _l2(e)
                            if float(np.linalg.norm(e)) > 0:
                                rows.append(e)
                                ids.append(oid)
                        except Exception as ex:
                            log.warning("skip sample %s: %s", p, ex)
            self._matrix = np.stack(dino_rows, axis=0).astype(np.float32) if dino_rows else np.zeros((0, 0), np.float32)
            self._row_object_id = dino_ids
            self._shape_matrix = np.stack(shape_rows, axis=0).astype(np.float32) if shape_rows else np.zeros((0, 0), np.float32)
            self._shape_row_object_id = shape_ids
            self._depth_matrix = np.stack(depth_rows, axis=0).astype(np.float32) if depth_rows else np.zeros((0, 0), np.float32)
            self._depth_row_object_id = depth_ids
            self._save_index()

    def _save_index(self) -> None:
        with self._lock:
            def _m(m: np.ndarray) -> np.ndarray:
                return m if m.size else np.zeros((0, 1), np.float32)

            np.savez_compressed(
                self.embeddings_path,
                matrix=_m(self._matrix),
                object_ids=np.array(self._row_object_id, dtype=np.int64),
                shape_matrix=_m(self._shape_matrix),
                shape_object_ids=np.array(self._shape_row_object_id, dtype=np.int64),
                depth_matrix=_m(self._depth_matrix),
                depth_object_ids=np.array(self._depth_row_object_id, dtype=np.int64),
            )

    # ------------------------------------------------------------- remember
    def add_object(
        self,
        user_name: str,
        generic_class: Optional[str],
        embeddings: Sequence[np.ndarray],
        sample_images_paths: Sequence[str],
        shape_embeddings: Optional[Sequence[Optional[np.ndarray]]] = None,
        depth_embeddings: Optional[Sequence[Optional[np.ndarray]]] = None,
    ) -> MemoryObject:
        """Create (or replace) object with N view embeddings (+ optional shape/depth)."""
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

            dim = int(np.asarray(embeddings[0], dtype=np.float32).reshape(-1).shape[0])
            if self._matrix.size == 0:
                self._matrix = np.zeros((0, dim), dtype=np.float32)

            shape_list = list(shape_embeddings) if shape_embeddings is not None else [None] * len(embeddings)
            depth_list = list(depth_embeddings) if depth_embeddings is not None else [None] * len(embeddings)

            new_dino: List[np.ndarray] = []
            new_shape: List[np.ndarray] = []
            new_depth: List[np.ndarray] = []
            shape_oids: List[int] = []
            depth_oids: List[int] = []

            for i, emb in enumerate(embeddings):
                e = _l2(emb)
                img_path = sample_images_paths[i] if i < len(sample_images_paths) else None
                shape_row = None
                depth_row = None

                if i < len(shape_list) and shape_list[i] is not None:
                    sv = _l2(shape_list[i])
                    if float(np.linalg.norm(sv)) > 0:
                        if self._shape_matrix.size == 0:
                            self._shape_matrix = np.zeros((0, sv.shape[0]), np.float32)
                        if self._shape_matrix.shape[1] != sv.shape[0]:
                            self._shape_matrix = np.zeros((0, sv.shape[0]), np.float32)
                            self._shape_row_object_id = []
                        shape_row = int(self._shape_matrix.shape[0]) + len(new_shape)
                        new_shape.append(sv)
                        shape_oids.append(oid)
                        if img_path:
                            os.makedirs(os.path.dirname(img_path) or ".", exist_ok=True)
                            np.save(img_path + ".shape.npy", sv)

                if i < len(depth_list) and depth_list[i] is not None:
                    dv = _l2(depth_list[i])
                    if float(np.linalg.norm(dv)) > 0:
                        if self._depth_matrix.size == 0:
                            self._depth_matrix = np.zeros((0, dv.shape[0]), np.float32)
                        if self._depth_matrix.shape[1] != dv.shape[0]:
                            self._depth_matrix = np.zeros((0, dv.shape[0]), np.float32)
                            self._depth_row_object_id = []
                        depth_row = int(self._depth_matrix.shape[0]) + len(new_depth)
                        new_depth.append(dv)
                        depth_oids.append(oid)
                        if img_path:
                            os.makedirs(os.path.dirname(img_path) or ".", exist_ok=True)
                            np.save(img_path + ".depth.npy", dv)

                if img_path:
                    os.makedirs(os.path.dirname(img_path) or ".", exist_ok=True)
                    np.save(img_path + ".npy", e)

                dino_row = int(self._matrix.shape[0]) + len(new_dino)
                new_dino.append(e)
                cur.execute(
                    "INSERT INTO object_samples (object_id, image_path, embedding_row, shape_row, depth_row, created_at) "
                    "VALUES (?,?,?,?,?,?)",
                    (oid, img_path, dino_row, shape_row, depth_row, now),
                )

            if new_dino:
                mat = np.stack(new_dino, axis=0)
                if self._matrix.shape[1] != mat.shape[1]:
                    self._conn.commit()
                    self._rebuild_index()
                    return self.get_object(oid) or MemoryObject(oid, user_name, generic_class, now)
                self._matrix = (
                    np.concatenate([self._matrix, mat], axis=0) if self._matrix.size else mat
                )
                self._row_object_id.extend([oid] * len(new_dino))
            if new_shape:
                sm = np.stack(new_shape, axis=0)
                self._shape_matrix = (
                    np.concatenate([self._shape_matrix, sm], axis=0) if self._shape_matrix.size else sm
                )
                self._shape_row_object_id.extend(shape_oids)
            if new_depth:
                dm = np.stack(new_depth, axis=0)
                self._depth_matrix = (
                    np.concatenate([self._depth_matrix, dm], axis=0) if self._depth_matrix.size else dm
                )
                self._depth_row_object_id.extend(depth_oids)

            self._conn.commit()
            self._save_index()
            n_s = sum(1 for s in shape_list if s is not None)
            n_d = sum(1 for d in depth_list if d is not None)
            log.info(
                "Remembered '%s' with %d samples (id=%d) shape=%d depth=%d",
                user_name,
                len(embeddings),
                oid,
                n_s,
                n_d,
            )
            return MemoryObject(oid, user_name, generic_class, now, sample_count=len(embeddings))

    def _drop_object_rows(self, oid: int) -> None:
        keep_d = [i for i, x in enumerate(self._row_object_id) if x != oid]
        keep_s = [i for i, x in enumerate(self._shape_row_object_id) if x != oid]
        keep_p = [i for i, x in enumerate(self._depth_row_object_id) if x != oid]
        if keep_d:
            self._matrix = self._matrix[keep_d]
            self._row_object_id = [self._row_object_id[i] for i in keep_d]
        else:
            self._matrix = np.zeros((0, self._matrix.shape[1] if self._matrix.size else 0), np.float32)
            self._row_object_id = []
        if keep_s:
            self._shape_matrix = self._shape_matrix[keep_s]
            self._shape_row_object_id = [self._shape_row_object_id[i] for i in keep_s]
        else:
            self._shape_matrix = np.zeros((0, self._shape_matrix.shape[1] if self._shape_matrix.size else 0), np.float32)
            self._shape_row_object_id = []
        if keep_p:
            self._depth_matrix = self._depth_matrix[keep_p]
            self._depth_row_object_id = [self._depth_row_object_id[i] for i in keep_p]
        else:
            self._depth_matrix = np.zeros((0, self._depth_matrix.shape[1] if self._depth_matrix.size else 0), np.float32)
            self._depth_row_object_id = []

    # ------------------------------------------------------------- query
    @staticmethod
    def _channel_scores(
        matrix: np.ndarray,
        row_oids: Sequence[int],
        q: Optional[np.ndarray],
    ) -> dict[int, float]:
        """max cosine per object_id for one channel."""
        if q is None or matrix.size == 0 or not row_oids:
            return {}
        qq = _l2(q)
        if qq.shape[0] != matrix.shape[1]:
            return {}
        scores = matrix @ qq
        best: dict[int, float] = {}
        for i, s in enumerate(scores):
            oid = int(row_oids[i])
            fs = float(s)
            if oid not in best or fs > best[oid]:
                best[oid] = fs
        return best

    def query(
        self,
        embedding: Optional[np.ndarray],
        topk: Optional[int] = None,
        shape_embedding: Optional[np.ndarray] = None,
        depth_embedding: Optional[np.ndarray] = None,
    ) -> List[MemoryMatch]:
        """Fused match: weighted avg of available channels, renormalized."""
        with self._lock:
            if self._matrix.size == 0 and self._shape_matrix.size == 0 and self._depth_matrix.size == 0:
                return []
            if embedding is None and shape_embedding is None and depth_embedding is None:
                return []

            s_dino = self._channel_scores(self._matrix, self._row_object_id, embedding)
            s_shape = (
                self._channel_scores(self._shape_matrix, self._shape_row_object_id, shape_embedding)
                if self.use_shape
                else {}
            )
            s_depth = (
                self._channel_scores(self._depth_matrix, self._depth_row_object_id, depth_embedding)
                if self.use_depth
                else {}
            )

            # collect candidate object ids: prefer dino candidates when present
            cand: set[int] = set()
            if embedding is not None:
                cand |= set(s_dino.keys())
            if shape_embedding is not None:
                cand |= set(s_shape.keys())
            if depth_embedding is not None:
                cand |= set(s_depth.keys())
            if not cand:
                return []

            fused: dict[int, float] = {}
            parts: dict[int, tuple[Optional[float], Optional[float], Optional[float]]] = {}
            for oid in cand:
                num = 0.0
                den = 0.0
                pd = s_dino.get(oid)
                ps = s_shape.get(oid) if self.use_shape else None
                pp = s_depth.get(oid) if self.use_depth else None
                # only count a channel if BOTH query and store provide it
                if pd is not None and embedding is not None:
                    num += self.w_dino * pd
                    den += self.w_dino
                if ps is not None and shape_embedding is not None:
                    num += self.w_shape * ps
                    den += self.w_shape
                if pp is not None and depth_embedding is not None:
                    num += self.w_depth * pp
                    den += self.w_depth
                if den <= 0:
                    continue
                fused[oid] = num / den
                parts[oid] = (pd if embedding is not None else None, ps, pp)

            ranked = sorted(fused.items(), key=lambda kv: -kv[1])
            k = topk or self.topk
            ranked = ranked[: max(k, 1)]

            out: List[MemoryMatch] = []
            names = self._names_for_ids([oid for oid, _ in ranked])
            for rank, (oid, s) in enumerate(ranked):
                if s < self.threshold:
                    continue
                info = names.get(oid)
                pd, ps, pp = parts.get(oid, (None, None, None))
                out.append(
                    MemoryMatch(
                        object_id=oid,
                        user_name=info[0] if info else f"obj_{oid}",
                        generic_class=info[1] if info else None,
                        score=s,
                        rank=rank,
                        score_dino=pd,
                        score_shape=ps,
                        score_depth=pp,
                    )
                )
            return out

    def best_match(
        self,
        embedding: Optional[np.ndarray] = None,
        shape_embedding: Optional[np.ndarray] = None,
        depth_embedding: Optional[np.ndarray] = None,
    ) -> Optional[MemoryMatch]:
        m = self.query(embedding, topk=1, shape_embedding=shape_embedding, depth_embedding=depth_embedding)
        return m[0] if m else None

    def match_detections(
        self,
        embed_list: Sequence[Optional[np.ndarray]],
        shape_list: Optional[Sequence[Optional[np.ndarray]]] = None,
        depth_list: Optional[Sequence[Optional[np.ndarray]]] = None,
    ) -> Optional[tuple[int, MemoryMatch]]:
        """Return (index, match) for the globally best fused score above threshold."""
        best_i = -1
        best_m: Optional[MemoryMatch] = None
        for i, e in enumerate(embed_list):
            if e is None and (shape_list is None or i >= len(shape_list) or shape_list[i] is None):
                continue
            sh = shape_list[i] if shape_list is not None and i < len(shape_list) else None
            dp = depth_list[i] if depth_list is not None and i < len(depth_list) else None
            m = self.best_match(e, shape_embedding=sh, depth_embedding=dp)
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
            "shape_shape": tuple(self._shape_matrix.shape),
            "depth_shape": tuple(self._depth_matrix.shape),
            "weights": [self.w_dino, self.w_shape, self.w_depth],
        }

    def close(self) -> None:
        with self._lock:
            self._conn.close()
