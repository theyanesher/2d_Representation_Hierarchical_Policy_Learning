from __future__ import annotations
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np

class HDF5DataRecorder:
    """
    Append nested observations with Virtual Dataset optimization for static data.
    Static data is stored once physically but presented as a full-length time series.
    """

    def __init__(
        self, 
        path: Path | str, 
        mode: str = "w", 
        compression: str | None = "gzip",
        compression_opts: int | None = 4, 
        shuffle: bool = True, 
        scaleoffset: int | None = None,
        virtual_keys: Iterable[str] | None = None,
        verify_static: bool = False
    ):
        self.path = Path(path)
        self.file = h5py.File(self.path, mode)
        self.compression = compression
        self.compression_opts = compression_opts
        self.shuffle = shuffle
        self.scaleoffset = scaleoffset
        
        self.virtual_keys = set(virtual_keys) if virtual_keys else set()
        self.verify_static = verify_static
        self._finalized = False
        # Track where virtual datasets need to be created: {name: (group_path, sample_shape, dtype)}
        self._vds_registry: dict[str, tuple[str, tuple[int, ...], np.dtype]] = {}

    def append_observation(self, observation: dict[str, Any]) -> None:
        if self._finalized:
            raise RuntimeError("Cannot append to a finalized recorder.")
        self._append_recursive(self.file, observation)

    def _append_recursive(self, group: h5py.Group, node: dict[str, Any]) -> None:
        for key, value in node.items():
            if isinstance(value, dict):
                subgroup = group.require_group(key)
                self._append_recursive(subgroup, value)
            else:
                self._append_value(group, key, value)

    def _append_value(self, group: h5py.Group, name: str, value: Any) -> None:
        data = np.asarray(value)
        if data.dtype == object:
            raise ValueError(f"Cannot store object dtype for '{name}'.")

        # Handle Virtual Keys: Store physically only once in a hidden group
        if name in self.virtual_keys:
            phys_group = self.file.require_group("_physical")
            if name not in phys_group:
                # Store the single frame
                phys_group.create_dataset(
                    name, 
                    data=data, 
                    compression=self.compression,
                    compression_opts=self.compression_opts,
                    shuffle=self.shuffle,
                    scaleoffset=self.scaleoffset
                )
                # Register the intent to create a VDS in the original group later
                self._vds_registry[name] = (group.name, data.shape, data.dtype)
            elif self.verify_static:
                if not np.array_equal(phys_group[name][()], data):
                    raise ValueError(f"Static data check failed for '{name}': data changed at a new timestep.")
            return

        # Normal Time-Series Append Logic
        sample_shape = data.shape
        if name not in group:
            group.create_dataset(
                name,
                data=data.reshape((1,) + sample_shape),
                maxshape=(None,) + sample_shape,
                chunks=(1,) + sample_shape if sample_shape else (1,),
                compression=self.compression,
                compression_opts=self.compression_opts,
                shuffle=self.shuffle,
                scaleoffset=self.scaleoffset,
            )
        else:
            dataset = group[name]
            dataset.resize(dataset.shape[0] + 1, axis=0)
            dataset[-1] = data

    def finalize(self) -> None:
        """
        Converts staged virtual keys into full-length Virtual Datasets.
        This determines the required length based on existing non-virtual datasets.
        """
        if self._finalized or self.file is None:
            return

        # 1. Determine the total timesteps recorded from the first available non-virtual dataset
        total_steps = 0
        def _find_max_steps(group):
            nonlocal total_steps
            for item in group.values():
                if total_steps > 0: return
                if isinstance(item, h5py.Dataset) and item.name.split('/')[-1] not in self.virtual_keys:
                    total_steps = item.shape[0]
                elif isinstance(item, h5py.Group):
                    _find_max_steps(item)
        
        _find_max_steps(self.file)

        if total_steps == 0 and self._vds_registry:
            # If ONLY virtual keys were recorded, we assume 1 timestep or user didn't record data
            total_steps = 1

        # 2. Create Virtual Datasets in the intended groups
        for name, (group_path, sample_shape, dtype) in self._vds_registry.items():
            target_group = self.file[group_path]
            backing_ds = self.file["_physical"][name]
            
            # Create the layout (T, ...)
            layout = h5py.VirtualLayout(shape=(total_steps, *sample_shape), dtype=dtype)
            
            # Map the single physical slice to every index in the virtual time series
            v_source = h5py.VirtualSource(self.file.filename, backing_ds.name, shape=sample_shape)
            for i in range(total_steps):
                layout[i] = v_source
            
            target_group.create_virtual_dataset(name, layout)

        self._finalized = True

    def save_static_dict(self, data: dict[str, Any], group_name: str) -> None:
        if group_name in self.file:
            raise ValueError(f"Group '{group_name}' already exists.")
        root_grp = self.file.create_group(group_name)
        self._save_static_recursive(root_grp, data)

    def _save_static_recursive(self, group: h5py.Group, node: dict[str, Any]) -> None:
        for key, value in node.items():
            if isinstance(value, dict):
                subgroup = group.create_group(key)
                self._save_static_recursive(subgroup, value)
            else:
                self._save_static_value(group, key, value)

    def _save_static_value(self, group: h5py.Group, name: str, value: Any) -> None:
        data = np.asarray(value)
        if data.dtype.kind in {'U', 'S'}:
             data = data.astype(h5py.string_dtype(encoding='utf-8'))
        elif data.dtype == object:
             try:
                 data = data.astype(str).astype(h5py.string_dtype(encoding='utf-8'))
             except ValueError:
                 raise ValueError(f"Cannot save object type for '{name}'.")
        
        kwargs = {}
        if data.ndim > 0:
            kwargs = {"compression": self.compression, "compression_opts": self.compression_opts}
        
        group.create_dataset(name, data=data, **kwargs)

    def close(self) -> None:
        if self.file is not None:
            try:
                self.finalize()
            finally:
                self.file.close()
                self.file = None

    def __enter__(self) -> "HDF5DataRecorder":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.close()
        elif self.file:
            self.file.close()
            self.file = None

    @staticmethod
    def load_static_dict(group: h5py.Group) -> dict[str, Any]:
        out = {}
        for k, v in group.items():
            if isinstance(v, h5py.Group):
                out[k] = HDF5DataRecorder.load_static_dict(v)
            else:
                val = v[()]
                if isinstance(val, bytes):
                    val = val.decode('utf-8')
                elif isinstance(val, np.ndarray) and val.dtype.kind == 'S':
                    val = val.astype(str)
                out[k] = val
        return out