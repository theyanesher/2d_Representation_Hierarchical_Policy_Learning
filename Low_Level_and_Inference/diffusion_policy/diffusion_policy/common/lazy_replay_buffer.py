"""
Lazy HDF5-backed replay buffer.

Reads only metadata (shapes, dtypes, episode lengths) at construction time.
Actual array data is loaded from disk on demand via LazyH5Array.

Provides the subset of the ReplayBuffer interface required by SequenceSampler:
  - .keys()
  - .episode_ends          (numpy array)
  - .__getitem__(key)      -> LazyH5Array with .shape, .dtype, and [start:stop]
"""

import os
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Union

import h5py
import numpy as np
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Per-thread (per-worker) LRU file handle cache
# ---------------------------------------------------------------------------
# threading.local() stores data per-thread, but on Linux DataLoader workers
# are forked processes and inherit the parent thread's local storage —
# including any h5py handles opened in the main process (e.g. during
# get_normalizer). A PID check detects this and discards inherited handles
# so each worker opens its own fresh file descriptors.
#
# The cache is bounded to _H5_CACHE_SIZE handles per worker to avoid hitting
# the OS open-file-descriptor limit (EMFILE / errno 24) when the dataset
# contains many HDF5 files.

_H5_CACHE_SIZE = 8  # max open handles per DataLoader worker / thread

_tl = threading.local()


def _get_h5_handle(path: str) -> h5py.File:
    """Return a per-thread, per-process LRU-cached read-only h5py file handle."""
    pid = os.getpid()
    if not hasattr(_tl, 'pid') or _tl.pid != pid:
        _tl.cache = OrderedDict()
        _tl.pid = pid

    cache: OrderedDict = _tl.cache
    if path in cache:
        cache.move_to_end(path)  # mark as most-recently used
        return cache[path]

    # Evict least-recently-used handle if at capacity
    while len(cache) >= _H5_CACHE_SIZE:
        _, evicted = cache.popitem(last=False)
        try:
            evicted.close()
        except Exception:
            pass

    handle = h5py.File(path, 'r')
    cache[path] = handle
    return handle


# ---------------------------------------------------------------------------
# LazyH5Array
# ---------------------------------------------------------------------------

class LazyH5Array:
    """
    Array-like view over one HDF5 dataset key spread across multiple files.

    Each file holds exactly one episode. Global flat indices are mapped to
    (file_idx, local_index) at access time.

    SequenceSampler guarantees that every [start:stop] slice lives within a
    single episode, so single-file reads are the common fast path. Full-array
    reads (e.g. [:] for normalizer stats) are also supported via multi-file
    concatenation.

    Attributes
    ----------
    shape : tuple
        Total shape across all episodes, first dim is total timesteps.
    dtype : np.dtype
        Data type of the underlying HDF5 dataset.
    """

    def __init__(
        self,
        h5_paths: List[str],
        h5_key: str,
        episode_starts: np.ndarray,
        episode_ends: np.ndarray,
        shape: tuple,
        dtype: np.dtype,
    ):
        self._h5_paths = h5_paths
        self._h5_key = h5_key
        self._episode_starts = episode_starts
        self._episode_ends = episode_ends
        self._shape = shape
        self._dtype = dtype

    @property
    def shape(self) -> tuple:
        return self._shape

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    def _ep_idx(self, global_idx: int) -> int:
        """Episode index that contains global_idx."""
        return int(np.searchsorted(self._episode_ends, global_idx, side='right'))

    def __getitem__(self, s):
        if isinstance(s, (int, np.integer)):
            idx = int(s)
            if idx < 0:
                idx += self._shape[0]
            ep = self._ep_idx(idx)
            local = idx - int(self._episode_starts[ep])
            f = _get_h5_handle(self._h5_paths[ep])
            return f[self._h5_key][local]

        elif isinstance(s, slice):
            start, stop, step = s.indices(self._shape[0])
            if step != 1:
                raise IndexError("LazyH5Array only supports step=1 slices")
            if start >= stop:
                return np.zeros((0,) + self._shape[1:], dtype=self._dtype)

            ep0 = self._ep_idx(start)
            ep1 = self._ep_idx(stop - 1)

            if ep0 == ep1:
                # fast path: slice within one episode
                ep_start = int(self._episode_starts[ep0])
                f = _get_h5_handle(self._h5_paths[ep0])
                return f[self._h5_key][start - ep_start : stop - ep_start]
            else:
                # slice spans multiple episodes (e.g. full [:] reads)
                chunks = []
                for ep in range(ep0, ep1 + 1):
                    ep_start = int(self._episode_starts[ep])
                    ep_end = int(self._episode_ends[ep])
                    chunk_start = max(start, ep_start) - ep_start
                    chunk_stop = min(stop, ep_end) - ep_start
                    f = _get_h5_handle(self._h5_paths[ep])
                    chunks.append(f[self._h5_key][chunk_start:chunk_stop])
                return np.concatenate(chunks, axis=0)

        else:
            raise IndexError(f"LazyH5Array does not support indexing with {type(s)}")


# ---------------------------------------------------------------------------
# LazyH5Buffer
# ---------------------------------------------------------------------------

class LazyH5Buffer:
    """
    Drop-in replacement for ReplayBuffer backed by HDF5 files on disk.

    Parameters
    ----------
    h5_paths : list of Path or str
        One HDF5 file per episode, in order.
    key_to_h5path : dict
        Maps buffer key (e.g. 'action', 'rgb') to the HDF5 dataset path
        within each file (e.g. 'action', 'obs/rgb').
        Keys absent from a file are silently skipped.

    Example
    -------
    >>> buf = LazyH5Buffer(
    ...     h5_paths=sorted(data_dir.glob("*.h5")),
    ...     key_to_h5path={
    ...         'action': 'action',
    ...         'rgb':    'obs/rgb',
    ...         'state':  'obs/state',
    ...     }
    ... )
    >>> sampler = SequenceSampler(replay_buffer=buf, sequence_length=16, ...)
    """

    def __init__(
        self,
        h5_paths: List[Union[str, Path]],
        key_to_h5path: Dict[str, str],
    ):
        self._h5_paths = [str(p) for p in h5_paths]
        self._key_to_h5path = key_to_h5path

        episode_lengths: List[int] = []
        key_shapes: Dict[str, tuple] = {}
        key_dtypes: Dict[str, np.dtype] = {}

        for path in tqdm(self._h5_paths, desc="Loading metadata from HDF5 files"):
            with h5py.File(path, 'r') as f:
                # Determine episode length from the first available key.
                ep_len = None
                for h5_key in key_to_h5path.values():
                    if h5_key in f:
                        ep_len = f[h5_key].shape[0]
                        break
                if ep_len is None:
                    raise ValueError(
                        f"None of the expected HDF5 keys found in {path}. "
                        f"Expected one of: {list(key_to_h5path.values())}"
                    )
                episode_lengths.append(ep_len)

                # Collect shape/dtype from first file that has each key.
                for buf_key, h5_key in key_to_h5path.items():
                    if h5_key in f and buf_key not in key_shapes:
                        ds = f[h5_key]
                        key_shapes[buf_key] = ds.shape[1:]
                        key_dtypes[buf_key] = ds.dtype

        lengths = np.array(episode_lengths, dtype=np.int64)
        self._episode_ends = np.cumsum(lengths)
        self._episode_starts = np.concatenate([[0], self._episode_ends[:-1]])

        total_steps = int(self._episode_ends[-1]) if len(self._episode_ends) > 0 else 0

        # Build (and cache) one LazyH5Array per key.
        self._arrays: Dict[str, LazyH5Array] = {}
        for buf_key in key_to_h5path:
            if buf_key not in key_shapes:
                continue  # key missing from all files
            self._arrays[buf_key] = LazyH5Array(
                h5_paths=self._h5_paths,
                h5_key=key_to_h5path[buf_key],
                episode_starts=self._episode_starts,
                episode_ends=self._episode_ends,
                shape=(total_steps,) + key_shapes[buf_key],
                dtype=key_dtypes[buf_key],
            )

    # ---- ReplayBuffer-compatible interface --------------------------------

    @property
    def episode_ends(self) -> np.ndarray:
        return self._episode_ends

    @property
    def n_episodes(self) -> int:
        return len(self._episode_ends)

    @property
    def n_steps(self) -> int:
        return int(self._episode_ends[-1]) if len(self._episode_ends) > 0 else 0

    @property
    def episode_lengths(self) -> np.ndarray:
        return np.diff(np.concatenate([[0], self._episode_ends]))

    def keys(self):
        return self._arrays.keys()

    def __contains__(self, key: str) -> bool:
        return key in self._arrays

    def __getitem__(self, key: str) -> LazyH5Array:
        return self._arrays[key]
