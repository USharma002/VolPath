import time
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal, QCoreApplication
from scipy.interpolate import LinearNDInterpolator, RBFInterpolator
from scipy.spatial import cKDTree
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp

from config.optional import (
    HAS_CUPY,
    HAS_NUMBA,
    HAS_NATURAL_NEIGHBOR,
    RBFInterpolator_GPU,
    cp,
    njit,
    prange,
    nninterpol,
)


# =============================================================================
# Numba-accelerated interpolation kernels
# =============================================================================

if HAS_NUMBA:

    @njit(parallel=True, fastmath=True, cache=True)
    def numba_nearest_interp(data, scalars, query_points):
        """Fast nearest neighbor using Numba with parallel execution."""
        n_queries = query_points.shape[0]
        n_data = data.shape[0]
        result = np.empty(n_queries, dtype=np.float32)
        
        for i in prange(n_queries):
            min_dist = np.float32(1e30)
            best_val = np.float32(0.0)
            qx = query_points[i, 0]
            qy = query_points[i, 1]
            qz = query_points[i, 2]
            
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

    @njit(parallel=True, fastmath=True, cache=True)
    def numba_idw_interp(data, scalars, query_points, power=2.0):
        """Inverse Distance Weighting interpolation with Numba."""
        n_queries = query_points.shape[0]
        n_data = data.shape[0]
        result = np.empty(n_queries, dtype=np.float32)
        
        for i in prange(n_queries):
            qx = query_points[i, 0]
            qy = query_points[i, 1]
            qz = query_points[i, 2]
            
            weight_sum = np.float32(0.0)
            value_sum = np.float32(0.0)
            exact_match = False
            exact_val = np.float32(0.0)
            
            for j in range(n_data):
                dx = qx - data[j, 0]
                dy = qy - data[j, 1]
                dz = qz - data[j, 2]
                dist_sq = dx * dx + dy * dy + dz * dz
                
                if dist_sq < 1e-12:
                    exact_match = True
                    exact_val = scalars[j]
                    break
                
                # weight = 1 / dist^power
                weight = 1.0 / (dist_sq ** (power / 2.0))
                weight_sum += weight
                value_sum += weight * scalars[j]
            
            if exact_match:
                result[i] = exact_val
            else:
                result[i] = value_sum / weight_sum
        
        return result

else:  # pragma: no cover

    def numba_nearest_interp(data, scalars, query_points):
        raise RuntimeError("Numba not available")
    
    def numba_idw_interp(data, scalars, query_points, power=2.0):
        raise RuntimeError("Numba not available")


class InterpolationWorker(QThread):
    """Worker thread for interpolation with GPU/Numba acceleration."""

    progress = pyqtSignal(int)  # -1 for indeterminate
    finished = pyqtSignal(np.ndarray, float, dict)
    status = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, data, scalars, resolution, method, epsilon=1.0, bounds=None):
        super().__init__()
        # Store data as contiguous float32 for optimal performance
        self.data = np.ascontiguousarray(data, dtype=np.float32)
        self.scalars = np.ascontiguousarray(scalars, dtype=np.float32)
        self.resolution = resolution
        self.method = method
        self.epsilon = epsilon
        self.bounds = bounds
        self._n_workers = min(mp.cpu_count(), 8)

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

            self.progress.emit(-1)

            # Resolve bounds once
            mins, maxs = self._resolve_bounds()
            
            # Route to optimized interpolation method
            if self.method == 'linear':
                grid_values = self._interpolate_linear_optimized(mins, maxs)
            elif self.method == 'nearest':
                grid_values = self._interpolate_nearest_optimized(mins, maxs)
            elif self.method.startswith('rbf_'):
                grid_values = self._interpolate_rbf_optimized(mins, maxs)
            elif self.method == 'natural_neighbor':
                if not HAS_NATURAL_NEIGHBOR:
                    raise RuntimeError("Natural neighbor interpolation requires nnpycgal")
                axis_x = np.linspace(mins[0], maxs[0], self.resolution, dtype=np.float32)
                axis_y = np.linspace(mins[1], maxs[1], self.resolution, dtype=np.float32)
                axis_z = np.linspace(mins[2], maxs[2], self.resolution, dtype=np.float32)
                grid_values = self._interpolate_natural_neighbor_3d(axis_x, axis_y, axis_z, mins, maxs)
            else:
                raise ValueError(f"Unknown method: {self.method}")

            stats, normalized = self._normalize_volume(grid_values)
            elapsed = time.time() - start_time
            stats['elapsed'] = elapsed
            stats['method'] = self.method
            self.finished.emit(normalized.astype(np.float32), elapsed, stats)
            
        except MemoryError:  # pragma: no cover
            self.error.emit(f"MEMORY ERROR: Not enough RAM for {self.resolution}³ grid!")
        except Exception as exc:  # pragma: no cover - safety net
            import traceback
            traceback.print_exc()
            self.error.emit(f"ERROR: {exc}")

    # =========================================================================
    # Optimized Linear Interpolation
    # =========================================================================
    def _interpolate_linear_optimized(self, mins, maxs):
        """Optimized linear interpolation with parallel slice processing."""
        resolution = self.resolution
        
        self.status.emit("⏳ Building Delaunay triangulation...")
        build_start = time.time()
        
        # Build interpolator once
        # Use np.nan for fill_value so points outside convex hull are properly
        # excluded during log normalization (1e-8 caused log10=-8, compressing dynamic range)
        interp = LinearNDInterpolator(self.data, self.scalars, fill_value=np.nan)
        self.status.emit(f"✓ Triangulation complete ({time.time() - build_start:.1f}s)")
        
        # Pre-compute grid axes
        axis_x = np.linspace(mins[0], maxs[0], resolution, dtype=np.float32)
        axis_y = np.linspace(mins[1], maxs[1], resolution, dtype=np.float32)
        axis_z = np.linspace(mins[2], maxs[2], resolution, dtype=np.float32)
        
        # Pre-compute YZ meshgrid (reused for each X slice)
        Y, Z = np.meshgrid(axis_y, axis_z, indexing='ij')
        yz_flat = np.column_stack([Y.ravel(), Z.ravel()]).astype(np.float32)
        n_yz = yz_flat.shape[0]
        
        grid_values = np.empty((resolution, resolution, resolution), dtype=np.float32)
        
        self.status.emit(f"⏳ Interpolating {resolution} slices...")
        self.progress.emit(0)
        
        # Process slices in chunks for better cache utilization
        # But cap chunk size so temporary `chunk_points` arrays don't exceed
        # a reasonable memory budget (prevents UI freezing on large volumes).
        # Target ~150 MB per chunk for the temporary points buffer.
        max_chunk_mem_bytes = 150 * 1024 * 1024
        bytes_per_float = 4
        max_floats = max(1, max_chunk_mem_bytes // bytes_per_float)
        # n_yz = resolution**2 (precomputed above)
        safe_chunk = max(1, int(max_floats / (n_yz * 3)))
        chunk_size = min(max(1, resolution // 20), safe_chunk)
        
        for i in range(0, resolution, chunk_size):
            chunk_end = min(i + chunk_size, resolution)
            chunk_len = chunk_end - i
            
            # Build query points for entire chunk
            chunk_points = np.empty((chunk_len * n_yz, 3), dtype=np.float32)
            for j, xi in enumerate(range(i, chunk_end)):
                start_idx = j * n_yz
                end_idx = start_idx + n_yz
                chunk_points[start_idx:end_idx, 0] = axis_x[xi]
                chunk_points[start_idx:end_idx, 1:] = yz_flat
            
            # Interpolate entire chunk at once
            chunk_values = interp(chunk_points)
            
            # Reshape and store
            for j, xi in enumerate(range(i, chunk_end)):
                start_idx = j * n_yz
                end_idx = start_idx + n_yz
                grid_values[xi, :, :] = chunk_values[start_idx:end_idx].reshape((resolution, resolution))
            
            progress_pct = int(100 * chunk_end / resolution)
            self.progress.emit(progress_pct)
            QCoreApplication.processEvents()
        
        return grid_values

    # =========================================================================
    # Optimized Nearest Neighbor Interpolation
    # =========================================================================
    def _interpolate_nearest_optimized(self, mins, maxs):
        """Optimized nearest neighbor with KD-tree or Numba."""
        resolution = self.resolution
        n_points = len(self.data)
        
        # Pre-compute grid
        axis_x = np.linspace(mins[0], maxs[0], resolution, dtype=np.float32)
        axis_y = np.linspace(mins[1], maxs[1], resolution, dtype=np.float32)
        axis_z = np.linspace(mins[2], maxs[2], resolution, dtype=np.float32)
        
        # Create full 3D query grid at once
        X, Y, Z = np.meshgrid(axis_x, axis_y, axis_z, indexing='ij')
        query_points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()]).astype(np.float32)
        total_queries = query_points.shape[0]
        
        grid_values = np.empty(total_queries, dtype=np.float32)
        
        # Choose optimal method based on data size
        use_numba = HAS_NUMBA and n_points < 50000 and total_queries < 500000
        
        if use_numba:
            self.status.emit(f"🚀 Using Numba-accelerated nearest neighbor ({n_points} pts → {total_queries} queries)")
            self.progress.emit(-1)
            
            # Process in chunks to show progress
            chunk_size = total_queries // 10
            for i in range(0, total_queries, chunk_size):
                chunk_end = min(i + chunk_size, total_queries)
                grid_values[i:chunk_end] = numba_nearest_interp(
                    self.data, self.scalars, query_points[i:chunk_end]
                )
                self.progress.emit(int(100 * chunk_end / total_queries))
                QCoreApplication.processEvents()
        else:
            self.status.emit("⏳ Building KD-tree...")
            build_start = time.time()
            kdtree = cKDTree(self.data, leafsize=32, balanced_tree=True)
            self.status.emit(f"✓ KD-tree built ({time.time() - build_start:.1f}s)")
            
            # Query in chunks
            self.status.emit(f"⏳ Querying {total_queries:,} points...")
            chunk_size = min(500000, total_queries // 10)
            
            for i in range(0, total_queries, chunk_size):
                chunk_end = min(i + chunk_size, total_queries)
                _, indices = kdtree.query(query_points[i:chunk_end], k=1, workers=-1)
                grid_values[i:chunk_end] = self.scalars[indices]
                
                self.progress.emit(int(100 * chunk_end / total_queries))
                QCoreApplication.processEvents()
        
        return grid_values.reshape((resolution, resolution, resolution))

    # =========================================================================
    # Optimized RBF Interpolation
    # =========================================================================
    def _interpolate_rbf_optimized(self, mins, maxs):
        """Optimized RBF interpolation with GPU support and chunking."""
        resolution = self.resolution
        kernel = self.method.replace('rbf_', '')
        
        epsilon_kernels = {'gaussian', 'multiquadric', 'inverse_multiquadric', 'inverse_quadratic'}
        neighbors = self._estimate_neighbors()
        
        # Try GPU first
        use_gpu = HAS_CUPY
        interp = None
        
        if use_gpu:
            self.status.emit(f"🚀 Building GPU RBF: {kernel}, {neighbors} neighbors...")
            build_start = time.time()
            try:
                data_gpu = cp.asarray(self.data)
                scalars_gpu = cp.asarray(self.scalars)
                kwargs = {'neighbors': neighbors, 'kernel': kernel}
                if kernel in epsilon_kernels:
                    kwargs['epsilon'] = self.epsilon
                interp = RBFInterpolator_GPU(data_gpu, scalars_gpu, **kwargs)
                self.status.emit(f"✓ GPU RBF built ({time.time() - build_start:.1f}s)")
            except Exception as e:
                self.status.emit(f"⚠ GPU failed: {e}, falling back to CPU")
                use_gpu = False
        
        if not use_gpu:
            self.status.emit(f"⏳ Building CPU RBF: {kernel}, {neighbors} neighbors...")
            build_start = time.time()
            kwargs = {'kernel': kernel, 'neighbors': neighbors}
            if kernel in epsilon_kernels:
                kwargs['epsilon'] = self.epsilon
            interp = RBFInterpolator(self.data, self.scalars, **kwargs)
            self.status.emit(f"✓ CPU RBF built ({time.time() - build_start:.1f}s)")
        
        # Pre-compute grid
        axis_x = np.linspace(mins[0], maxs[0], resolution, dtype=np.float32)
        axis_y = np.linspace(mins[1], maxs[1], resolution, dtype=np.float32)
        axis_z = np.linspace(mins[2], maxs[2], resolution, dtype=np.float32)
        
        Y, Z = np.meshgrid(axis_y, axis_z, indexing='ij')
        yz_flat = np.column_stack([Y.ravel(), Z.ravel()]).astype(np.float32)
        n_yz = yz_flat.shape[0]
        
        grid_values = np.empty((resolution, resolution, resolution), dtype=np.float32)
        
        # Determine chunk size based on path
        chunk_size = 4 if use_gpu else max(4, resolution // 16)
        
        self.status.emit(f"⏳ Interpolating {resolution} slices (chunk_size={chunk_size})...")
        self.progress.emit(0)
        
        if use_gpu:
            # GPU path: batch multiple slices
            slice_points = np.empty((n_yz, 3), dtype=np.float32)
            slice_points[:, 1:] = yz_flat
            slice_points_gpu = cp.asarray(slice_points)
            
            for i, x_val in enumerate(axis_x):
                slice_points_gpu[:, 0] = x_val
                grid_values[i, :, :] = cp.asnumpy(interp(slice_points_gpu)).reshape((resolution, resolution))
                
                if i % 4 == 0:
                    self.progress.emit(int(100 * (i + 1) / resolution))
                    QCoreApplication.processEvents()
        else:
            # CPU path: process in larger chunks
            slice_points = np.empty((n_yz, 3), dtype=np.float32)
            slice_points[:, 1:] = yz_flat
            
            for i in range(0, resolution, chunk_size):
                chunk_end = min(i + chunk_size, resolution)
                
                # Build chunk query points
                chunk_len = chunk_end - i
                chunk_points = np.empty((chunk_len * n_yz, 3), dtype=np.float32)
                
                for j, xi in enumerate(range(i, chunk_end)):
                    start_idx = j * n_yz
                    end_idx = start_idx + n_yz
                    chunk_points[start_idx:end_idx, 0] = axis_x[xi]
                    chunk_points[start_idx:end_idx, 1:] = yz_flat
                
                # Interpolate chunk
                chunk_values = interp(chunk_points)
                
                # Store results
                for j, xi in enumerate(range(i, chunk_end)):
                    start_idx = j * n_yz
                    end_idx = start_idx + n_yz
                    grid_values[xi, :, :] = chunk_values[start_idx:end_idx].reshape((resolution, resolution))
                
                self.progress.emit(int(100 * chunk_end / resolution))
                QCoreApplication.processEvents()
        
        return grid_values

    def _estimate_neighbors(self):
        n = len(self.data)
        if n > 50000:
            return min(15, n // 100)
        elif n > 10000:
            return min(20, n // 50)
        return min(30, max(10, n // 10))

    def _interpolate_natural_neighbor_3d(self, axis_x, axis_y, axis_z, mins, maxs):
        """
        3D natural neighbor interpolation using 2D nninterpol slice-by-slice.
        
        This is the BASELINE implementation - no fallbacks, errors are raised.
        
        Algorithm:
        1. Pre-sort points by Z for fast slice extraction via binary search
        2. For each Z slice, extract nearby points within adaptive thickness
        3. Convert float coordinates to integer grid indices (nninterpol requirement)
        4. Handle duplicate grid cells by averaging values
        5. Call nninterpol for true Sibson natural neighbor interpolation
        """
        import multiprocessing as mp
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        resolution = self.resolution
        n_points = len(self.data)
        
        if n_points < 4:
            raise ValueError(f"Natural neighbor requires at least 4 points, got {n_points}")
        
        # Pre-compute coordinate scaling to grid space [0, resolution-1]
        x_range = maxs[0] - mins[0]
        y_range = maxs[1] - mins[1]
        z_range = maxs[2] - mins[2]
        
        if x_range < 1e-8 or y_range < 1e-8 or z_range < 1e-8:
            raise ValueError(f"Data range too small: x={x_range}, y={y_range}, z={z_range}")
        
        x_scale = (resolution - 1) / x_range
        y_scale = (resolution - 1) / y_range
        
        # Scale X,Y to grid space, keep Z in world space for slicing
        scaled_x = ((self.data[:, 0] - mins[0]) * x_scale).astype(np.float64)
        scaled_y = ((self.data[:, 1] - mins[1]) * y_scale).astype(np.float64)
        data_z = self.data[:, 2].astype(np.float64)
        scalars = self.scalars.astype(np.float64)
        
        # Sort by Z for efficient slice extraction
        z_order = np.argsort(data_z)
        sorted_x = scaled_x[z_order]
        sorted_y = scaled_y[z_order]
        sorted_z = data_z[z_order]
        sorted_scalars = scalars[z_order]
        
        # Adaptive slice thickness
        base_thickness = z_range / resolution
        
        # Pre-compute slice bounds using binary search
        slice_bounds = []
        for k in range(resolution):
            z_val = axis_z[k]
            thickness = base_thickness * 1.5
            z_lo, z_hi = z_val - thickness, z_val + thickness
            
            i_lo = np.searchsorted(sorted_z, z_lo, side='left')
            i_hi = np.searchsorted(sorted_z, z_hi, side='right')
            
            # Expand thickness if too few points
            while i_hi - i_lo < 10 and (i_lo > 0 or i_hi < n_points):
                thickness *= 1.5
                z_lo, z_hi = z_val - thickness, z_val + thickness
                i_lo = np.searchsorted(sorted_z, z_lo, side='left')
                i_hi = np.searchsorted(sorted_z, z_hi, side='right')
            
            slice_bounds.append((k, i_lo, i_hi))
        
        self.status.emit(f"⏳ Natural neighbor: {resolution}³ grid, {n_points} points...")
        
        # Output array
        grid_values = np.empty((resolution, resolution, resolution), dtype=np.float32)
        
        # Track errors for reporting
        errors = []
        
        def process_slice(args):
            """Process a single Z slice with natural neighbor interpolation."""
            k, i_lo, i_hi = args
            
            # Extract points for this slice
            if i_hi > i_lo:
                pts_x = sorted_x[i_lo:i_hi]
                pts_y = sorted_y[i_lo:i_hi]
                pts_s = sorted_scalars[i_lo:i_hi]
            else:
                # Use all points if slice is empty
                pts_x = sorted_x
                pts_y = sorted_y
                pts_s = sorted_scalars
            
            n_pts = len(pts_x)
            
            # Convert float coordinates to integer grid indices
            # nninterpol REQUIRES List[int] for coordinates
            pts_x_int = np.clip(np.round(pts_x).astype(np.int32), 0, resolution - 1)
            pts_y_int = np.clip(np.round(pts_y).astype(np.int32), 0, resolution - 1)
            
            # Aggregate values at duplicate grid positions (vectorized)
            coords = pts_x_int * resolution + pts_y_int  # Unique key per cell
            unique_coords, inverse = np.unique(coords, return_inverse=True)
            
            # Sum values and counts per unique cell
            n_unique = len(unique_coords)
            if n_unique < 3:
                # Not enough unique points for triangulation
                return k, None, f"Slice {k}: only {n_unique} unique grid positions"
            
            sums = np.bincount(inverse, weights=pts_s, minlength=n_unique)
            counts = np.bincount(inverse, minlength=n_unique)
            avg_values = sums / counts
            
            # Decode back to x, y coordinates
            unique_x = (unique_coords // resolution).tolist()
            unique_y = (unique_coords % resolution).tolist()
            unique_s = avg_values.tolist()
            
            # Call nninterpol - NOTE: nninterpol uses (row, col) = (y, x) convention
            # We pass (y, x) so output[i,j] corresponds to our (x=i, y=j)
            # This means: nninterpol(y_coords, x_coords, ...) -> result[x, y]
            result = np.array(
                nninterpol(unique_y, unique_x, unique_s, resolution, resolution),
                dtype=np.float32
            )
            
            # Replace NaN/inf with small positive value (outside convex hull)
            np.nan_to_num(result, copy=False, nan=1e-10, posinf=1e-10, neginf=1e-10)
            
            return k, result, None
        
        # Process slices in parallel - nninterpol releases GIL
        n_workers = min(mp.cpu_count(), 8, resolution)
        completed = [0]
        
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(process_slice, sb): sb[0] for sb in slice_bounds}
            
            for future in as_completed(futures):
                k, result, error = future.result()
                
                if error:
                    errors.append(error)
                    # Fill with small value on error
                    grid_values[:, :, k] = 1e-10
                else:
                    grid_values[:, :, k] = result
                
                completed[0] += 1
                if completed[0] % max(1, resolution // 20) == 0:
                    pct = int(100 * completed[0] / resolution)
                    self.progress.emit(pct)
                    QCoreApplication.processEvents()
        
        self.progress.emit(100)
        
        if errors:
            n_errors = len(errors)
            self.status.emit(f"⚠ Natural neighbor: {n_errors}/{resolution} slices had issues")
            # Log first few errors
            for err in errors[:5]:
                print(f"  {err}")
        else:
            self.status.emit(f"✓ Natural neighbor complete")
        
        return grid_values

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
        """Normalize volume data with efficient in-place operations."""
        stats = {
            'raw_min': float(np.nanmin(grid_values)),
            'raw_max': float(np.nanmax(grid_values)),
            'raw_mean': float(np.nanmean(grid_values)),
        }
        positive_mask = grid_values > 0
        stats['positive_count'] = int(positive_mask.sum())
        stats['positive_pct'] = 100.0 * stats['positive_count'] / grid_values.size

        # Work in-place to save memory
        grid_values[~positive_mask] = np.nan
        with np.errstate(divide='ignore', invalid='ignore'):
            np.log10(grid_values, where=positive_mask, out=grid_values)

        valid_mask = np.isfinite(grid_values)
        stats['valid_count'] = int(valid_mask.sum())
        stats['valid_pct'] = 100.0 * stats['valid_count'] / grid_values.size
        stats['nan_count'] = grid_values.size - stats['valid_count'] - int(np.isinf(grid_values).sum())
        stats['inf_count'] = int(np.isinf(grid_values).sum())

        if stats['valid_count'] == 0:
            raise RuntimeError("No valid values after log transform")

        if stats['valid_pct'] < 10.0:
            self.status.emit(f"⚠ WARNING: Only {stats['valid_pct']:.1f}% valid values!")

        # Compute stats from valid values only
        valid_values = grid_values[valid_mask]
        grid_min = valid_values.min()
        grid_max = valid_values.max()
        stats['log_min'] = float(grid_min)
        stats['log_max'] = float(grid_max)
        stats['log_range'] = float(grid_max - grid_min)
        
        if stats['log_range'] < 1e-10:
            raise RuntimeError("All values are identical after log transform")

        # In-place normalization
        grid_values[~valid_mask] = grid_min
        grid_values -= grid_min
        grid_values /= (grid_max - grid_min)
        np.clip(grid_values, 0, 1, out=grid_values)
        
        return stats, grid_values.astype(np.float32, copy=False)


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
