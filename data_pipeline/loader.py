import glob
import os
import re
from collections import defaultdict

import h5py
import numpy as np

VTK_EXTENSIONS = {'.vtk', '.vtp', '.vti'}


class DataLoader:
    """Handles loading HDF5 snapshot data from Illustris simulations"""

    _last_files = []
    _last_header = {}

    @classmethod
    def get_last_files(cls):
        return list(cls._last_files)

    @classmethod
    def get_last_header(cls):
        return dict(cls._last_header)

    @staticmethod
    def load_snapshot(snapshot_path):
        files = DataLoader._resolve_snapshot_files(snapshot_path)
        if not files:
            return {}

        DataLoader._last_files = files
        DataLoader._last_header = DataLoader._read_header_metadata(files[0])

        print(f"[DataLoader] Loading {len(files)} file(s)...")
        for f in files[:3]:
            print(f"  - {os.path.basename(f)}")
        if len(files) > 3:
            print(f"  ... and {len(files) - 3} more")

        data = defaultdict(lambda: defaultdict(list))
        for filename in files:
            try:
                with h5py.File(filename, "r") as handle:
                    for ptype in handle.keys():
                        if ptype.startswith("PartType"):
                            for key in handle[ptype].keys():
                                data[ptype][key].append(handle[ptype][key][()])
            except Exception as exc:
                print(f"[DataLoader] Error reading {filename}: {exc}")
                continue

        if not data:
            return {}
        return dict(data)

    @staticmethod
    def _resolve_snapshot_files(snapshot_path):
        if os.path.isdir(snapshot_path):
            files = glob.glob(os.path.join(snapshot_path, "*.hdf5"))
        elif os.path.isfile(snapshot_path):
            files = DataLoader._expand_snapshot_pattern(snapshot_path)
        else:
            files = []
        return DataLoader._sort_files(files)

    @staticmethod
    def _expand_snapshot_pattern(file_path):
        base_dir = os.path.dirname(file_path)
        base_name = os.path.basename(file_path)
        match = re.search(r'([._])(\d+)\.hdf5$', base_name)
        if match:
            prefix = base_name[:match.start(2)]
            pattern = os.path.join(base_dir, f"{prefix}*.hdf5")
            files = glob.glob(pattern)
            if files:
                return files
        return [file_path]

    @staticmethod
    def _sort_files(files):
        def natural_key(path):
            parts = re.split(r'(\d+)', os.path.basename(path))
            return [int(p) if p.isdigit() else p.lower() for p in parts]

        return sorted(files, key=natural_key)

    @staticmethod
    def _read_header_metadata(file_path):
        metadata = {}
        try:
            with h5py.File(file_path, "r") as handle:
                header = handle.get('Header')
                if header is not None:
                    for key, value in header.attrs.items():
                        metadata[key] = DataLoader._to_python(value)
        except Exception as exc:
            print(f"[DataLoader] Warning: Unable to read header metadata from {file_path}: {exc}")
        return metadata

    @staticmethod
    def _to_python(value):
        if isinstance(value, bytes):
            try:
                return value.decode('utf-8')
            except Exception:
                return value
        if isinstance(value, np.ndarray):
            if value.shape == ():
                return value.item()
            if value.size == 1:
                return value.reshape(-1)[0].item()
            return value.tolist()
        try:
            return value.item()
        except AttributeError:
            return value

    @staticmethod
    def create_polydata(particle_data):
        if "Coordinates" not in particle_data:
            return None

        result = {}
        for field_name, values_list in particle_data.items():
            try:
                if field_name == "Coordinates":
                    result[field_name] = np.vstack(values_list)
                else:
                    values = np.concatenate([
                        v.reshape(v.shape[0], -1) for v in values_list
                    ])
                    result[field_name] = values
            except (ValueError, TypeError) as exc:
                print(f"[DataLoader] Warning: Could not process '{field_name}': {exc}")

        return result


def load_multi_file_hdf5(base_path, particle_type, coord_field, scalar_field):
    snapshot_data = DataLoader.load_snapshot(base_path)
    if not snapshot_data or particle_type not in snapshot_data:
        raise ValueError(f"No data found for {particle_type}")

    polydata = DataLoader.create_polydata(snapshot_data[particle_type])
    if not polydata:
        raise ValueError("Failed to create polydata")

    if coord_field not in polydata:
        available = list(polydata.keys())
        raise ValueError(f"Coordinate field '{coord_field}' not found. Available: {available}")

    if scalar_field not in polydata:
        available = list(polydata.keys())
        raise ValueError(f"Scalar field '{scalar_field}' not found. Available: {available}")

    coords = polydata[coord_field]
    scalars = polydata[scalar_field]

    if coords.ndim == 1:
        coords = coords.reshape(-1, 3)

    if scalars.ndim > 1:
        if scalars.shape[1] == 1:
            scalars = scalars.flatten()
        else:
            scalars = np.linalg.norm(scalars, axis=1)
            print(f"[DataLoader] {scalar_field} is vector, using magnitude")

    last_files = DataLoader.get_last_files()
    if last_files:
        num_files = len(last_files)
    else:
        num_files = len(DataLoader._resolve_snapshot_files(base_path)) or 1

    print(f"[DataLoader] ✓ Loaded {len(coords):,} particles from {num_files} file(s)")
    return coords, scalars, num_files


def load_vtk_polydata(path):
    """Load VTK/VTK-derived datasets via pyvista and return polydata dict."""
    try:
        import pyvista as pv  # Lazy import so dependency stays optional
    except ImportError as exc:  # pragma: no cover - dependency optional in CI
        raise ImportError(
            "pyvista is required to load VTK/VT* files. Install it via pip install pyvista"
        ) from exc

    dataset = pv.read(path)
    if dataset is None or dataset.n_points == 0:
        raise ValueError(f"VTK file '{path}' did not contain any points")

    polydata = {
        'Coordinates': np.asarray(dataset.points, dtype=np.float32)
    }

    point_data = dataset.point_data
    if point_data:
        for key in point_data.keys():
            name = key or 'Scalars'
            polydata[name] = np.asarray(point_data[key])
    elif dataset.active_scalars is not None:
        polydata['Scalars'] = np.asarray(dataset.active_scalars)

    field_metadata = {
        'format': 'VTK',
        'dataset_type': dataset.__class__.__name__,
        'n_points': int(dataset.n_points),
        'n_cells': int(getattr(dataset, 'n_cells', 0)),
    }

    return polydata, field_metadata


__all__ = [
    "DataLoader",
    "load_multi_file_hdf5",
    "load_vtk_polydata",
    "VTK_EXTENSIONS",
]
