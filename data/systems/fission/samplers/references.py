"""Exact allclose collision checks accelerated by a tree plus an insertion buffer."""
import numpy as np
from scipy.spatial import cKDTree


class FissionReferenceIndex:
    """Index every reference immediately, including rows not yet added to a tree.

    An infinity-norm query conservatively covers the per-coordinate allclose
    tolerance. Final decisions always use the original rtol/atol predicate.
    """
    def __init__(self, vectors, rtol=1e-5, atol=1e-9, rebuild_every=2048, recent_every=32):
        vectors = [np.array(v, dtype=float, copy=True) for v in vectors]
        if not vectors:
            raise ValueError('Fission deduplication requires real-mutant references')
        shape = vectors[0].shape
        if len(shape) != 1 or not shape[0] or any(v.shape != shape or not np.isfinite(v).all() for v in vectors):
            raise ValueError('Reference vectors must be finite and share one nonempty shape')
        if (not np.isfinite([rtol, atol]).all() or min(rtol, atol) < 0
                or rebuild_every < 1 or recent_every < 1):
            raise ValueError('Invalid collision tolerances or rebuild intervals')
        self.rtol, self.atol = rtol, atol
        self.rebuild_every = rebuild_every
        self.recent_every = recent_every
        self.since_rebuild = 0
        self.raw = vectors
        arr = np.asarray(self.raw)
        self.scale = np.maximum(np.abs(arr).max(axis=0), 1e-3)
        self.norm = [v / self.scale for v in self.raw]
        self._reference_bound = np.abs(np.asarray(self.norm)).max(axis=0)
        self._update_radius()
        self.tree = cKDTree(np.asarray(self.norm))
        self.recent = []
        self.recent_tree = None
        self._recent_indexed = 0

    def _update_radius(self):
        self.radius = float(np.max(self.atol / self.scale + self.rtol * self._reference_bound))

    def _vector(self, vector):
        vector = np.asarray(vector, dtype=float)
        if vector.shape != self.scale.shape or not np.isfinite(vector).all():
            raise ValueError('Collision vector has invalid dimensions or nonfinite values')
        return vector

    def collides(self, vector):
        vector = self._vector(vector)
        point = vector / self.scale
        # Bound rounding in normalised coordinates before applying exact allclose.
        padding = 8 * np.finfo(float).eps * (1 + max(np.abs(point).max(), self._reference_bound.max()))
        radius = self.radius + padding
        for tree, references in ((self.recent_tree, self.recent), (self.tree, self.raw)):
            if tree is not None:
                for index in tree.query_ball_point(point, radius, p=np.inf):
                    if np.allclose(vector, references[index], rtol=self.rtol, atol=self.atol):
                        return True
        pending = self.recent[self._recent_indexed:]
        if pending and np.isclose(vector[None], np.asarray(pending), rtol=self.rtol, atol=self.atol).all(axis=1).any():
            return True
        return False

    def add(self, vector):
        vector = self._vector(vector).copy()
        point = vector / self.scale
        self.raw.append(vector)
        self.norm.append(point)
        self.recent.append(vector)
        self._reference_bound = np.maximum(self._reference_bound, np.abs(point))
        self._update_radius()
        self.since_rebuild += 1
        if self.since_rebuild >= self.rebuild_every:
            self.tree = cKDTree(np.asarray(self.norm))
            self.recent = []
            self.recent_tree = None
            self._recent_indexed = 0
            self.since_rebuild = 0
        elif len(self.recent) - self._recent_indexed >= self.recent_every:
            self.recent_tree = cKDTree(np.asarray(self.norm[-len(self.recent):]))
            self._recent_indexed = len(self.recent)
