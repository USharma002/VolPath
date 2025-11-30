import time
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal
from scipy.interpolate import LinearNDInterpolator, RBFInterpolator, griddata
from scipy.spatial import cKDTree

from config.optional import (
    HAS_CUPY,
    HAS_NUMBA,
    HAS_NATURAL_NEIGHBOR,
    RBFInterpolator_GPU,
    cp,
    njit,
)


if HAS_NUMBA:

    @njit(parallel=True, fastmath=True)
    def numba_nearest_interp(data, scalars, query_points):
        """Fast nearest neighbor using Numba."""
        n_queries = query_points.shape[0]
        n_data = data.shape[0]
        result = np.zeros(n_queries, dtype=np.float32)
        for i in range(n_queries):
            min_dist = 1e30
            best_val = 0.0
            qx, qy, qz = query_points[i]
            for j in range(n_data):
                dx = qx - data[j, 0]
                dy = qy - data[j, 1]
                dz = qz - data[j, 2]
                dist = dx * dx + dy * dy + dz * dz
                if dist < min_dist:
                    min_dist = dist
                    best_val = scalars[j]
            result[i] = best_val
        return result


else:  # pragma: no cover

    def numba_nearest_interp(data, scalars, query_points):
        raise RuntimeError("Numba not available")


class InterpolationWorker(QThread):
    """Worker thread for interpolation with GPU/Numba acceleration."""

    progress = pyqtSignal(int)  # -1 for indeterminate
    finished = pyqtSignal(np.ndarray, float, dict)
    status = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, data, scalars, resolution, method, epsilon=1.0, bounds=None):
        super().__init__()
        self.data = data
        self.scalars = scalars
        self.resolution = resolution
        self.method = method
        self.epsilon = epsilon
        self.bounds = bounds

    def run(self):
        try:
            start_time = time.time()
            self.status.emit(f"Interpolating with {self.method} at {self.resolution}³...")

            required_mb = (self.resolution ** 3) * 4 / (1024 ** 2)
            self.status.emit(f"Required memory: {required_mb:.1f} MB")

            accel_info = []
            if HAS_CUPY:
                accel_info.append("CuPy GPU")
            if HAS_NUMBA:
                accel_info.append("Numba JIT")
            if accel_info:
                self.status.emit(f"✓ Acceleration available: {', '.join(accel_info)}")

            if required_mb > 4000:
                self.status.emit("⚠ WARNING: Very large volume, may fail!")

            epsilon_kernels = {
                'gaussian',
                'multiquadric',
                'inverse_multiquadric',
                'inverse_quadratic',
            }

            self.progress.emit(-1)

            use_gpu = HAS_CUPY and self.method.startswith('rbf_')
            interp = None

            if use_gpu:
                self.status.emit("🚀 Using GPU acceleration (CuPy)")
                kernel = self.method.replace('rbf_', '')
                neighbors = self._estimate_neighbors()
                self.status.emit(
                    f"⏳ Building GPU RBF system: {kernel}, {neighbors} neighbors (10-30s)..."
                )
                build_start = time.time()
                try:
                    data_gpu = cp.asarray(self.data)
                    scalars_gpu = cp.asarray(self.scalars)
                    kwargs = {
                        'neighbors': neighbors,
                        'kernel': kernel,
                    }
                    if kernel in epsilon_kernels:
                        kwargs['epsilon'] = self.epsilon
                    interp = RBFInterpolator_GPU(data_gpu, scalars_gpu, **kwargs)
                    self.status.emit(f"✓ GPU RBF system built ({time.time() - build_start:.1f}s)")
                except Exception as gpu_error:  # pragma: no cover - GPU fallback
                    self.status.emit(f"⚠ GPU failed: {gpu_error}")
                    self.status.emit("Falling back to CPU RBF...")
                    use_gpu = False
                    interp = self._build_cpu_rbf(kernel, neighbors, epsilon_kernels)
                    self.status.emit(f"✓ CPU RBF system built ({time.time() - build_start:.1f}s)")

            elif self.method == 'linear':
                self.status.emit("⏳ Building Delaunay triangulation (5-30s for large data)...")
                build_start = time.time()
                interp = LinearNDInterpolator(self.data, self.scalars, fill_value=1e-8)
                self.status.emit(f"✓ Triangulation complete ({time.time() - build_start:.1f}s)")

            elif self.method == 'nearest':
                if HAS_NUMBA and len(self.data) < 100000:
                    self.status.emit("🚀 Using Numba-accelerated nearest neighbor")
                    interp = 'numba_nearest'
                else:
                    self.status.emit("⏳ Building KD-tree (2-10s)...")
                    build_start = time.time()
                    self.kdtree = cKDTree(self.data)
                    interp = 'kdtree'
                    self.status.emit(f"✓ KD-tree complete ({time.time() - build_start:.1f}s)")

            elif self.method.startswith('rbf_'):
                kernel = self.method.replace('rbf_', '')
                neighbors = self._estimate_neighbors()
                self.status.emit(
                    f"⏳ Building RBF system: {kernel}, {neighbors} neighbors (20-60s)..."
                )
                interp = self._build_cpu_rbf(kernel, neighbors, epsilon_kernels)
                self.status.emit(f"✓ RBF system built")

            elif self.method == 'natural_neighbor':
                if not HAS_NATURAL_NEIGHBOR:
                    raise RuntimeError("Natural neighbor interpolation requires nnpycgal")
                interp = 'nn'
            else:
                raise ValueError(f"Unknown method: {self.method}")

            self.progress.emit(0)
            self.status.emit("Starting grid interpolation...")

            mins, maxs = self._resolve_bounds()
            axis_x = np.linspace(mins[0], maxs[0], self.resolution, dtype=np.float32)
            axis_y = np.linspace(mins[1], maxs[1], self.resolution, dtype=np.float32)
            axis_z = np.linspace(mins[2], maxs[2], self.resolution, dtype=np.float32)
            grid_values = np.empty((self.resolution, self.resolution, self.resolution), dtype=np.float32)

            Y, Z = np.meshgrid(axis_y, axis_z, indexing='ij')
            slice_points = np.empty((Y.size, 3), dtype=np.float32)
            slice_points[:, 1] = Y.ravel()
            slice_points[:, 2] = Z.ravel()

            slice_points_gpu = cp.asarray(slice_points) if use_gpu else None

            for i, x_val in enumerate(axis_x):
                slice_points[:, 0] = x_val
                if use_gpu:
                    slice_points_gpu[...] = slice_points
                    slice_values = cp.asnumpy(interp(slice_points_gpu))
                elif interp == 'numba_nearest':
                    slice_values = numba_nearest_interp(self.data, self.scalars, slice_points)
                elif interp == 'kdtree':
                    distances, indices = self.kdtree.query(slice_points, k=1)
                    slice_values = self.scalars[indices]
                elif interp == 'nn':
                    slice_values = griddata(
                        self.data,
                        self.scalars,
                        slice_points,
                        method='linear',
                        fill_value=1e-8,
                    )
                else:
                    slice_values = interp(slice_points)

                grid_values[i, :, :] = slice_values.reshape((self.resolution, self.resolution))
                progress_pct = int(100 * (i + 1) / self.resolution)
                self.progress.emit(progress_pct)

            stats, normalized = self._normalize_volume(grid_values)
            elapsed = time.time() - start_time
            stats['elapsed'] = elapsed
            stats['acceleration'] = 'GPU' if use_gpu else (
                'Numba' if interp == 'numba_nearest' else 'CPU'
            )
            self.finished.emit(normalized.astype(np.float32), elapsed, stats)
        except MemoryError:  # pragma: no cover
            self.error.emit(f"MEMORY ERROR: Not enough RAM for {self.resolution}³ grid!")
        except Exception as exc:  # pragma: no cover - safety net
            self.error.emit(f"ERROR: {exc}")

    def _estimate_neighbors(self):
        if len(self.data) > 10000:
            return min(20, len(self.data) // 50)
        return min(30, max(10, len(self.data) // 10))

    def _build_cpu_rbf(self, kernel, neighbors, epsilon_kernels):
        kwargs = {
            'kernel': kernel,
            'neighbors': neighbors,
        }
        if kernel in epsilon_kernels:
            kwargs['epsilon'] = self.epsilon
        return RBFInterpolator(self.data, self.scalars, **kwargs)

    def _resolve_bounds(self):
        if self.bounds is not None:
            mins, maxs = self.bounds
            mins = np.asarray(mins, dtype=np.float32)
            maxs = np.asarray(maxs, dtype=np.float32)
        else:
            mins = np.zeros(3, dtype=np.float32)
            maxs = np.ones(3, dtype=np.float32)
        return mins, maxs

    def _normalize_volume(self, grid_values):
        stats = {
            'raw_min': float(np.nanmin(grid_values)),
            'raw_max': float(np.nanmax(grid_values)),
            'raw_mean': float(np.nanmean(grid_values)),
        }
        positive_mask = grid_values > 0
        stats['positive_count'] = int(positive_mask.sum())
        stats['positive_pct'] = 100.0 * stats['positive_count'] / grid_values.size

        grid_values_log = grid_values
        grid_values_log[~positive_mask] = np.nan
        with np.errstate(divide='ignore'):
            np.log10(grid_values_log, where=positive_mask, out=grid_values_log)

        valid_mask = np.isfinite(grid_values_log, out=positive_mask)
        stats['valid_count'] = int(valid_mask.sum())
        stats['valid_pct'] = 100.0 * stats['valid_count'] / grid_values.size
        stats['nan_count'] = int(np.isnan(grid_values_log).sum())
        stats['inf_count'] = int(np.isinf(grid_values_log).sum())

        if stats['valid_count'] == 0:
            raise RuntimeError("No valid values after log transform")

        if stats['valid_pct'] < 10.0:
            self.status.emit(f"⚠ WARNING: Only {stats['valid_pct']:.1f}% valid values!")

        valid_values = grid_values_log[valid_mask]
        grid_min = valid_values.min()
        grid_max = valid_values.max()
        stats['log_min'] = float(grid_min)
        stats['log_max'] = float(grid_max)
        stats['log_range'] = float(grid_max - grid_min)
        if stats['log_range'] < 1e-10:
            raise RuntimeError("All values are identical after log transform")

        grid_values_log[~valid_mask] = grid_min
        normalized = (grid_values_log - grid_min) / (grid_max - grid_min)
        normalized = np.clip(normalized, 0, 1)
        return stats, normalized


class InterpolatorFactory:
    """Factory for creating interpolation workers."""

    def __init__(self, worker_cls=None):
        self.worker_cls = worker_cls or InterpolationWorker

    def create_worker(self, data, scalars, resolution, method, epsilon, bounds=None):
        return self.worker_cls(data, scalars, resolution, method, epsilon, bounds)


__all__ = [
    "InterpolationWorker",
    "InterpolatorFactory",
    "numba_nearest_interp",
]
