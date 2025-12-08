"""Plenoxel grid builder and renderer utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pyvista as pv

from renderers.base import RendererBase
from scipy.spatial import cKDTree
from config.optional import HAS_NUMBA, njit


USE_NUMBA_GRID = bool(HAS_NUMBA and njit is not None)

_AXIS_COORD_CACHE: Dict[int, np.ndarray] = {}


@dataclass
class PlenoxelNode:
    """Represents a single leaf cell in the plenoxel grid."""

    mins: np.ndarray
    maxs: np.ndarray
    population: int
    mean_scalar: float
    depth: int
    corner_values: Optional[np.ndarray] = None


def build_plenoxel_grid(
    points: np.ndarray,
    scalars: Optional[np.ndarray],
    *,
    min_points: int = 500,
    min_depth: int = 1,
    max_depth: int = 6,
    max_cells: int = 50000,
    corner_neighbors: int = 12,
) -> Tuple[List[PlenoxelNode], Dict[str, float]]:
    """Construct a sparse plenoxel grid by recursively subdividing space."""

    nodes: List[PlenoxelNode]
    stats: Dict[str, float]

    if points is None or len(points) == 0:
        return [], {
            'node_count': 0,
            'max_depth_reached': 0,
            'min_population': None,
            'max_population': None,
        }

    pts = np.ascontiguousarray(points, dtype=np.float32)
    scalars_arr = None
    if scalars is not None:
        scalars_arr = np.ascontiguousarray(scalars, dtype=np.float32)

    if USE_NUMBA_GRID:
        try:
            nodes, stats = _build_plenoxel_grid_numba(
                pts,
                scalars_arr,
                min_points,
                min_depth,
                max_depth,
                max_cells,
            )
        except Exception:  # pragma: no cover - fall back to python path
            nodes, stats = _build_plenoxel_grid_python(
                pts,
                scalars_arr,
                min_points,
                min_depth,
                max_depth,
                max_cells,
            )
    else:
        nodes, stats = _build_plenoxel_grid_python(
            pts,
            scalars_arr,
            min_points,
            min_depth,
            max_depth,
            max_cells,
        )

    if nodes and scalars_arr is not None and corner_neighbors > 0:
        _assign_corner_values(nodes, pts, scalars_arr, corner_neighbors)

    return nodes, stats


def _build_plenoxel_grid_python(
    pts: np.ndarray,
    scalars_arr: Optional[np.ndarray],
    min_points: int,
    min_depth: int,
    max_depth: int,
    max_cells: int,
) -> Tuple[List[PlenoxelNode], Dict[str, float]]:
    nodes: List[PlenoxelNode] = []
    stats = {
        'node_count': 0,
        'max_depth_reached': 0,
        'min_population': None,
        'max_population': None,
    }

    total_indices = np.arange(pts.shape[0])

    def _make_node(indices: np.ndarray, mins: np.ndarray, maxs: np.ndarray, depth: int) -> None:
        population = int(indices.size)
        if population == 0:
            return
        stats['max_depth_reached'] = max(stats['max_depth_reached'], depth)
        if stats['min_population'] is None:
            stats['min_population'] = population
            stats['max_population'] = population
        else:
            stats['min_population'] = min(stats['min_population'], population)
            stats['max_population'] = max(stats['max_population'], population)
        mean_scalar = 0.0
        if scalars_arr is not None and population > 0:
            mean_scalar = float(np.mean(scalars_arr[indices]))
        nodes.append(
            PlenoxelNode(
                mins=mins.astype(np.float32, copy=True),
                maxs=maxs.astype(np.float32, copy=True),
                population=population,
                mean_scalar=mean_scalar,
                depth=depth,
            )
        )
        stats['node_count'] = len(nodes)

    def recurse(indices: np.ndarray, mins: np.ndarray, maxs: np.ndarray, depth: int) -> None:
        if indices.size == 0 or len(nodes) >= max_cells:
            return
        if depth >= max_depth:
            _make_node(indices, mins, maxs, depth)
            return
        if depth >= min_depth and indices.size <= min_points:
            _make_node(indices, mins, maxs, depth)
            return

        mid = (mins + maxs) * 0.5
        pts_slice = pts[indices]
        x_ge = pts_slice[:, 0] >= mid[0]
        y_ge = pts_slice[:, 1] >= mid[1]
        z_ge = pts_slice[:, 2] >= mid[2]

        produced_child = False
        for octant in range(8):
            mask = (
                (x_ge if (octant & 1) else ~x_ge)
                & (y_ge if (octant & 2) else ~y_ge)
                & (z_ge if (octant & 4) else ~z_ge)
            )
            child_indices = indices[mask]
            if child_indices.size == 0:
                continue
            produced_child = True
            child_mins = np.array([
                mid[0] if (octant & 1) else mins[0],
                mid[1] if (octant & 2) else mins[1],
                mid[2] if (octant & 4) else mins[2],
            ], dtype=np.float32)
            child_maxs = np.array([
                maxs[0] if (octant & 1) else mid[0],
                maxs[1] if (octant & 2) else mid[1],
                maxs[2] if (octant & 4) else mid[2],
            ], dtype=np.float32)
            recurse(child_indices, child_mins, child_maxs, depth + 1)
        if not produced_child:
            _make_node(indices, mins, maxs, depth)

    recurse(total_indices, np.zeros(3, dtype=np.float32), np.ones(3, dtype=np.float32), 0)
    return nodes, stats


if USE_NUMBA_GRID:

    @njit(cache=True)
    def _numba_octree_build(
        points,
        scalars,
        has_scalars,
        min_points,
        min_depth,
        max_depth,
        max_cells,
    ):
        n_points = points.shape[0]
        max_cells = max(1, max_cells)

        node_mins = np.zeros((max_cells, 3), dtype=np.float32)
        node_maxs = np.zeros((max_cells, 3), dtype=np.float32)
        node_pop = np.zeros(max_cells, dtype=np.int32)
        node_mean = np.zeros(max_cells, dtype=np.float32)
        node_depth = np.zeros(max_cells, dtype=np.int16)

        indices = np.arange(n_points, dtype=np.int32)
        scratch = np.empty(n_points, dtype=np.int32)
        codes = np.empty(n_points, dtype=np.uint8)

        stack_cap = max(max_cells * 2 + 16, 64)
        stack_start = np.empty(stack_cap, dtype=np.int32)
        stack_end = np.empty(stack_cap, dtype=np.int32)
        stack_depth = np.empty(stack_cap, dtype=np.int16)
        stack_mins = np.empty((stack_cap, 3), dtype=np.float32)
        stack_maxs = np.empty((stack_cap, 3), dtype=np.float32)

        stack_ptr = 0
        stack_start[0] = 0
        stack_end[0] = n_points
        stack_depth[0] = 0
        stack_mins[0, 0] = 0.0
        stack_mins[0, 1] = 0.0
        stack_mins[0, 2] = 0.0
        stack_maxs[0, 0] = 1.0
        stack_maxs[0, 1] = 1.0
        stack_maxs[0, 2] = 1.0

        child_counts = np.zeros(8, dtype=np.int32)
        child_starts = np.zeros(8, dtype=np.int32)
        child_next = np.zeros(8, dtype=np.int32)

        node_count = 0
        max_depth_reached = 0
        min_population = 0
        max_population = 0

        while stack_ptr >= 0 and node_count < max_cells:
            start = stack_start[stack_ptr]
            end = stack_end[stack_ptr]
            depth = int(stack_depth[stack_ptr])
            min0 = stack_mins[stack_ptr, 0]
            min1 = stack_mins[stack_ptr, 1]
            min2 = stack_mins[stack_ptr, 2]
            max0 = stack_maxs[stack_ptr, 0]
            max1 = stack_maxs[stack_ptr, 1]
            max2 = stack_maxs[stack_ptr, 2]
            stack_ptr -= 1

            count = end - start
            if count <= 0:
                continue

            if depth >= max_depth or (depth >= min_depth and count <= min_points):
                total = 0.0
                if has_scalars and count > 0:
                    for idx_ptr in range(start, end):
                        total += scalars[indices[idx_ptr]]
                mean_val = total / count if (has_scalars and count > 0) else 0.0
                node_mins[node_count, 0] = min0
                node_mins[node_count, 1] = min1
                node_mins[node_count, 2] = min2
                node_maxs[node_count, 0] = max0
                node_maxs[node_count, 1] = max1
                node_maxs[node_count, 2] = max2
                node_pop[node_count] = count
                node_mean[node_count] = mean_val
                node_depth[node_count] = depth
                node_count += 1
                if depth > max_depth_reached:
                    max_depth_reached = depth
                if min_population == 0 or count < min_population:
                    min_population = count
                if count > max_population:
                    max_population = count
                continue

            mid0 = (min0 + max0) * 0.5
            mid1 = (min1 + max1) * 0.5
            mid2 = (min2 + max2) * 0.5

            produced_children = 0
            for i in range(8):
                child_counts[i] = 0

            for offset in range(count):
                idx_val = indices[start + offset]
                code = 0
                if points[idx_val, 0] >= mid0:
                    code |= 1
                if points[idx_val, 1] >= mid1:
                    code |= 2
                if points[idx_val, 2] >= mid2:
                    code |= 4
                codes[start + offset] = code
                child_counts[code] += 1

            for i in range(8):
                if child_counts[i] > 0:
                    produced_children += 1

            if produced_children == 0 or stack_ptr + produced_children >= stack_cap:
                total = 0.0
                if has_scalars and count > 0:
                    for idx_ptr in range(start, end):
                        total += scalars[indices[idx_ptr]]
                mean_val = total / count if (has_scalars and count > 0) else 0.0
                node_mins[node_count, 0] = min0
                node_mins[node_count, 1] = min1
                node_mins[node_count, 2] = min2
                node_maxs[node_count, 0] = max0
                node_maxs[node_count, 1] = max1
                node_maxs[node_count, 2] = max2
                node_pop[node_count] = count
                node_mean[node_count] = mean_val
                node_depth[node_count] = depth
                node_count += 1
                if depth > max_depth_reached:
                    max_depth_reached = depth
                if min_population == 0 or count < min_population:
                    min_population = count
                if count > max_population:
                    max_population = count
                continue

            cursor = 0
            for i in range(8):
                child_starts[i] = cursor
                child_next[i] = cursor
                cursor += child_counts[i]

            for offset in range(count):
                code = codes[start + offset]
                dest = start + child_next[code]
                scratch[dest] = indices[start + offset]
                child_next[code] += 1

            for offset in range(count):
                indices[start + offset] = scratch[start + offset]

            for octant in range(8):
                child_count = child_counts[octant]
                if child_count == 0:
                    continue
                child_start = start + child_starts[octant]
                child_end = child_start + child_count
                stack_ptr += 1
                stack_start[stack_ptr] = child_start
                stack_end[stack_ptr] = child_end
                stack_depth[stack_ptr] = depth + 1
                stack_mins[stack_ptr, 0] = mid0 if (octant & 1) else min0
                stack_mins[stack_ptr, 1] = mid1 if (octant & 2) else min1
                stack_mins[stack_ptr, 2] = mid2 if (octant & 4) else min2
                stack_maxs[stack_ptr, 0] = max0 if (octant & 1) else mid0
                stack_maxs[stack_ptr, 1] = max1 if (octant & 2) else mid1
                stack_maxs[stack_ptr, 2] = max2 if (octant & 4) else mid2

        if node_count == 0:
            min_population = 0
            max_population = 0

        return (
            node_count,
            node_mins,
            node_maxs,
            node_pop,
            node_mean,
            node_depth,
            max_depth_reached,
            min_population,
            max_population,
        )


    def _build_plenoxel_grid_numba(
        pts: np.ndarray,
        scalars_arr: Optional[np.ndarray],
        min_points: int,
        min_depth: int,
        max_depth: int,
        max_cells: int,
    ) -> Tuple[List[PlenoxelNode], Dict[str, float]]:
        has_scalars = scalars_arr is not None and scalars_arr.size == pts.shape[0]
        scalars_input = scalars_arr if has_scalars else np.zeros(1, dtype=np.float32)
        result = _numba_octree_build(
            pts,
            scalars_input,
            has_scalars,
            min_points,
            min_depth,
            max_depth,
            max_cells,
        )
        node_count = int(result[0])
        if node_count == 0:
            return [], {
                'node_count': 0,
                'max_depth_reached': 0,
                'min_population': None,
                'max_population': None,
            }
        mins_arr = result[1][:node_count].copy()
        maxs_arr = result[2][:node_count].copy()
        pop_arr = result[3][:node_count].astype(np.int32, copy=False)
        mean_arr = result[4][:node_count].astype(np.float32, copy=False)
        depth_arr = result[5][:node_count].astype(np.int32, copy=False)
        stats = {
            'node_count': node_count,
            'max_depth_reached': int(result[6]),
            'min_population': int(result[7]) if result[7] > 0 else None,
            'max_population': int(result[8]) if result[8] > 0 else None,
        }
        nodes: List[PlenoxelNode] = []
        for idx in range(node_count):
            nodes.append(
                PlenoxelNode(
                    mins=mins_arr[idx].copy(),
                    maxs=maxs_arr[idx].copy(),
                    population=int(pop_arr[idx]),
                    mean_scalar=float(mean_arr[idx]),
                    depth=int(depth_arr[idx]),
                )
            )
        return nodes, stats

    @njit(cache=True)
    def _numba_plenoxel_rasterize(
        mins_arr,
        maxs_arr,
        corner_arr,
        resolution,
    ):
        node_count = mins_arr.shape[0]
        volume = np.zeros((resolution, resolution, resolution), dtype=np.float32)
        raw_min = 0.0
        raw_max = 0.0
        has_value = False
        for idx in range(node_count):
            min0 = mins_arr[idx, 0]
            min1 = mins_arr[idx, 1]
            min2 = mins_arr[idx, 2]
            max0 = maxs_arr[idx, 0]
            max1 = maxs_arr[idx, 1]
            max2 = maxs_arr[idx, 2]

            x0 = int(min0 * resolution)
            if x0 < 0:
                x0 = 0
            elif x0 > resolution - 1:
                x0 = resolution - 1
            x1 = int(max0 * resolution)
            if x1 < 1:
                x1 = 1
            elif x1 > resolution:
                x1 = resolution

            y0 = int(min1 * resolution)
            if y0 < 0:
                y0 = 0
            elif y0 > resolution - 1:
                y0 = resolution - 1
            y1 = int(max1 * resolution)
            if y1 < 1:
                y1 = 1
            elif y1 > resolution:
                y1 = resolution

            z0 = int(min2 * resolution)
            if z0 < 0:
                z0 = 0
            elif z0 > resolution - 1:
                z0 = resolution - 1
            z1 = int(max2 * resolution)
            if z1 < 1:
                z1 = 1
            elif z1 > resolution:
                z1 = resolution

            sx = x1 - x0
            sy = y1 - y0
            sz = z1 - z0
            if sx <= 0 or sy <= 0 or sz <= 0:
                continue

            inv_sx = 1.0 / sx
            inv_sy = 1.0 / sy
            inv_sz = 1.0 / sz

            corners = corner_arr[idx]
            c000 = corners[0]
            c100 = corners[1]
            c010 = corners[2]
            c110 = corners[3]
            c001 = corners[4]
            c101 = corners[5]
            c011 = corners[6]
            c111 = corners[7]

            for ix in range(sx):
                u = (ix + 0.5) * inv_sx
                omu = 1.0 - u
                xi = x0 + ix
                for iy in range(sy):
                    v = (iy + 0.5) * inv_sy
                    omv = 1.0 - v
                    yi = y0 + iy
                    for iz in range(sz):
                        w = (iz + 0.5) * inv_sz
                        omw = 1.0 - w
                        zi = z0 + iz
                        value = (
                            c000 * omu * omv * omw +
                            c100 * u * omv * omw +
                            c010 * omu * v * omw +
                            c110 * u * v * omw +
                            c001 * omu * omv * w +
                            c101 * u * omv * w +
                            c011 * omu * v * w +
                            c111 * u * v * w
                        )
                        volume[xi, yi, zi] = value
                        if not has_value:
                            raw_min = value
                            raw_max = value
                            has_value = True
                        else:
                            if value < raw_min:
                                raw_min = value
                            if value > raw_max:
                                raw_max = value

        if not has_value:
            raw_min = 0.0
            raw_max = 0.0
        return volume, raw_min, raw_max


    def _plenoxel_volume_numba(
        nodes: Sequence[PlenoxelNode],
        resolution: int,
    ) -> Tuple[np.ndarray, float, float]:
        node_count = len(nodes)
        mins_arr = np.empty((node_count, 3), dtype=np.float32)
        maxs_arr = np.empty((node_count, 3), dtype=np.float32)
        corner_arr = np.empty((node_count, 8), dtype=np.float32)
        for idx, node in enumerate(nodes):
            mins_arr[idx] = node.mins
            maxs_arr[idx] = node.maxs
            if node.corner_values is not None and len(node.corner_values) == 8:
                corner_arr[idx] = node.corner_values
            else:
                corner_arr[idx].fill(node.mean_scalar)
        return _numba_plenoxel_rasterize(mins_arr, maxs_arr, corner_arr, resolution)

else:  # pragma: no cover - numba not available

    def _build_plenoxel_grid_numba(
        pts: np.ndarray,
        scalars_arr: Optional[np.ndarray],
        min_points: int,
        min_depth: int,
        max_depth: int,
        max_cells: int,
    ) -> Tuple[List[PlenoxelNode], Dict[str, float]]:
        raise RuntimeError("Numba not available for plenoxel acceleration")

    def _plenoxel_volume_numba(
        nodes: Sequence[PlenoxelNode],
        resolution: int,
    ) -> Tuple[np.ndarray, float, float]:
        raise RuntimeError("Numba not available for plenoxel acceleration")


class PlenoxelRenderer(RendererBase):
    """Renders plenoxel leaf cells as wireframe boxes using PyVista.
    
    Memory-optimized: builds all wireframe geometry in a single pass using
    numpy arrays instead of creating individual pv.Box objects.
    """

    # Edge indices for a box (8 vertices, 12 edges)
    EDGE_TEMPLATE = np.array([
        [0, 1], [1, 3], [3, 2], [2, 0],  # bottom face
        [4, 5], [5, 7], [7, 6], [6, 4],  # top face
        [0, 4], [1, 5], [2, 6], [3, 7]   # vertical edges
    ], dtype=np.uint32)
    
    def __init__(self, plotter, extent):
        super().__init__(plotter, extent)
        self._mesh_cache = None  # Cache last mesh for memory tracking

    def clear(self, render: bool = False):
        """Clear renderer and release mesh cache."""
        self._mesh_cache = None
        super().clear(render=render)

    def _build_box_vertices(self, mins: np.ndarray, maxs: np.ndarray) -> np.ndarray:
        """Build 8 corner vertices for a box. Memory-efficient inline."""
        return np.array([
            [mins[0], mins[1], mins[2]],  # 0
            [maxs[0], mins[1], mins[2]],  # 1
            [mins[0], maxs[1], mins[2]],  # 2
            [maxs[0], maxs[1], mins[2]],  # 3
            [mins[0], mins[1], maxs[2]],  # 4
            [maxs[0], mins[1], maxs[2]],  # 5
            [mins[0], maxs[1], maxs[2]],  # 6
            [maxs[0], maxs[1], maxs[2]],  # 7
        ], dtype=np.float32)

    def render(
        self,
        nodes: Sequence[PlenoxelNode],
        stats: Optional[Dict[str, float]] = None,
        *,
        visible: bool = True,
    ):
        if not nodes:
            self.clear()
            return None

        n_nodes = len(nodes)
        
        # Pre-allocate arrays for all geometry at once (memory efficient)
        # Each box has 8 vertices and 12 edges (24 line segment endpoints)
        all_vertices = np.empty((n_nodes * 8, 3), dtype=np.float32)
        all_lines = np.empty((n_nodes * 12, 3), dtype=np.int64)  # [2, idx1, idx2] format
        
        # Population-based coloring
        min_pop = stats.get('min_population') if stats else None
        max_pop = stats.get('max_population') if stats else None
        pop_range = max(max_pop - min_pop, 1) if (min_pop is not None and max_pop is not None) else None
        
        # Build all geometry in a single pass
        for i, node in enumerate(nodes):
            mins = node.mins * self.extent
            maxs = node.maxs * self.extent
            
            # Vertex offset for this box
            v_offset = i * 8
            e_offset = i * 12
            
            # Build vertices inline (avoid function call overhead)
            all_vertices[v_offset:v_offset + 8] = [
                [mins[0], mins[1], mins[2]],
                [maxs[0], mins[1], mins[2]],
                [mins[0], maxs[1], mins[2]],
                [maxs[0], maxs[1], mins[2]],
                [mins[0], mins[1], maxs[2]],
                [maxs[0], mins[1], maxs[2]],
                [mins[0], maxs[1], maxs[2]],
                [maxs[0], maxs[1], maxs[2]],
            ]
            
            # Build edges with offset indices
            for j, (e0, e1) in enumerate(self.EDGE_TEMPLATE):
                all_lines[e_offset + j] = [2, v_offset + e0, v_offset + e1]

        # Create single PolyData with all lines
        cells = all_lines.ravel()
        mesh = pv.PolyData(all_vertices, lines=cells)
        self._mesh_cache = mesh  # Keep reference for memory management
        
        # Compute average color from population
        if pop_range is not None and min_pop is not None:
            t_avg = np.mean([(node.population - min_pop) / pop_range for node in nodes])
        else:
            t_avg = 0.5
        avg_color = (0.2 + 0.7 * t_avg, 0.6 * (1.0 - t_avg) + 0.2, 1.0 - 0.5 * t_avg)

        self.clear()
        
        if not visible:
            return None

        try:
            self.actor = self.plotter.add_mesh(
                mesh,
                color=avg_color,
                opacity=0.85,
                line_width=1,
                render_lines_as_tubes=False,
                show_scalar_bar=False,
            )
            
            self.set_active(visible)
            
        except Exception as e:
            print(f"Plenoxel render error: {e}")
            return None
        
        return self.actor


def _assign_corner_values(
    nodes: Sequence[PlenoxelNode],
    points: np.ndarray,
    scalars: np.ndarray,
    neighbors: int,
) -> None:
    """Assign interpolated corner values to each node using k-NN weighting."""

    nodes = list(nodes)
    if not nodes:
        return
    neighbors = max(1, int(neighbors))
    neighbors = min(neighbors, points.shape[0])
    try:
        tree = cKDTree(points)
    except Exception:
        # Fallback: skip interpolation if KD-tree fails
        return

    eps = 1e-12
    total_corners = len(nodes) * 8
    all_points = np.empty((total_corners, 3), dtype=np.float32)
    cursor = 0
    for node in nodes:
        all_points[cursor:cursor + 8] = _node_corners(node.mins, node.maxs)
        cursor += 8
    try:
        dists, idxs = tree.query(all_points, k=neighbors, workers=-1)
    except TypeError:  # Older SciPy without workers arg
        dists, idxs = tree.query(all_points, k=neighbors)
    except Exception:
        return
    if neighbors == 1:
        dists = dists[:, None]
        idxs = idxs[:, None]
    weights = 1.0 / (np.maximum(dists, eps) ** 2)
    values = np.sum(weights * scalars[idxs], axis=1) / np.sum(weights, axis=1)
    cursor = 0
    for node in nodes:
        corner_vals = values[cursor:cursor + 8]
        if corner_vals.size == 8:
            node.corner_values = corner_vals.astype(np.float32, copy=False)
        cursor += 8


def _node_corners(mins: np.ndarray, maxs: np.ndarray) -> np.ndarray:
    corners = np.empty((8, 3), dtype=np.float32)
    corners[0] = (mins[0], mins[1], mins[2])
    corners[1] = (maxs[0], mins[1], mins[2])
    corners[2] = (mins[0], maxs[1], mins[2])
    corners[3] = (maxs[0], maxs[1], mins[2])
    corners[4] = (mins[0], mins[1], maxs[2])
    corners[5] = (maxs[0], mins[1], maxs[2])
    corners[6] = (mins[0], maxs[1], maxs[2])
    corners[7] = (maxs[0], maxs[1], maxs[2])
    return corners


def _get_axis_coords(length: int) -> np.ndarray:
    length = max(1, int(length))
    cached = _AXIS_COORD_CACHE.get(length)
    if cached is None:
        arr = (np.arange(length, dtype=np.float32) + 0.5) / float(length)
        _AXIS_COORD_CACHE[length] = arr
        cached = arr
    return cached


def plenoxel_volume_from_nodes(
    nodes: Sequence[PlenoxelNode],
    *,
    target_depth: Optional[int] = None,
) -> Tuple[Optional[np.ndarray], Dict[str, float]]:
    """Convert plenoxel nodes with corner values into a dense volume array."""

    nodes = list(nodes)
    stats = {
        'raw_min': None,
        'raw_max': None,
        'resolution': 0,
        'node_count': len(nodes),
    }
    if not nodes:
        return None, stats

    if target_depth is None:
        target_depth = max(node.depth for node in nodes)
    target_depth = max(target_depth, 1)
    resolution = 2 ** target_depth
    stats['resolution'] = resolution

    volume = None
    raw_min = 0.0
    raw_max = 0.0

    if USE_NUMBA_GRID:
        try:
            volume, raw_min, raw_max = _plenoxel_volume_numba(nodes, resolution)
        except Exception:
            volume = None

    if volume is None:
        volume, raw_min, raw_max = _plenoxel_volume_python(nodes, resolution)

    stats['raw_min'] = float(raw_min)
    stats['raw_max'] = float(raw_max)

    span = raw_max - raw_min
    if span > 0:
        volume = (volume - raw_min) / span
    else:
        volume.fill(0.0)
    return volume, stats


def _node_block(node: PlenoxelNode, cell_size: np.ndarray, out: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    """Build interpolated block for a plenoxel node.
    
    Memory-optimized: reuses output buffer if provided, uses in-place operations.
    
    Args:
        node: PlenoxelNode with corner values
        cell_size: (sx, sy, sz) voxel dimensions for this block
        out: Optional pre-allocated output buffer of shape (sx, sy, sz)
    
    Returns:
        Interpolated block array of shape (sx, sy, sz)
    """
    corner_vals = node.corner_values
    if corner_vals is None or corner_vals.size != 8:
        corner_vals = np.full(8, node.mean_scalar, dtype=np.float32)

    sx, sy, sz = map(int, cell_size)
    if min(sx, sy, sz) <= 0:
        return None

    # Get cached axis coordinates
    u = _get_axis_coords(sx)[:, None, None]
    v = _get_axis_coords(sy)[None, :, None]
    w = _get_axis_coords(sz)[None, None, :]
    
    # Pre-compute complement terms to reduce allocations
    u1 = 1.0 - u
    v1 = 1.0 - v
    w1 = 1.0 - w

    c000, c100, c010, c110, c001, c101, c011, c111 = corner_vals
    
    # Allocate or reuse output buffer
    if out is None or out.shape != (sx, sy, sz):
        out = np.empty((sx, sy, sz), dtype=np.float32)
    
    # Trilinear interpolation with minimal intermediate allocations
    # Bottom face (w=0)
    np.multiply(u1, v1, out=out)
    out *= w1
    out *= c000
    
    # Add remaining terms
    out += c100 * (u * v1 * w1)
    out += c010 * (u1 * v * w1)
    out += c110 * (u * v * w1)
    out += c001 * (u1 * v1 * w)
    out += c101 * (u * v1 * w)
    out += c011 * (u1 * v * w)
    out += c111 * (u * v * w)
    
    return out


def _plenoxel_volume_python(
    nodes: Sequence[PlenoxelNode],
    resolution: int,
) -> Tuple[np.ndarray, float, float]:
    """Convert plenoxel nodes to dense volume array.
    
    Memory-optimized: reuses block buffer across nodes when possible.
    """
    volume = np.zeros((resolution, resolution, resolution), dtype=np.float32)
    raw_min = None
    raw_max = None
    
    # Track last block size to reuse buffer when dimensions match
    last_block_size = None
    block_buffer = None
    
    for node in nodes:
        mins_idx = np.clip((node.mins * resolution).astype(int), 0, resolution - 1)
        maxs_idx = np.clip((node.maxs * resolution).astype(int), 1, resolution)
        cell_size = maxs_idx - mins_idx
        if np.any(cell_size <= 0):
            continue
        
        # Reuse block buffer if size matches
        cell_size_tuple = tuple(cell_size)
        if cell_size_tuple == last_block_size and block_buffer is not None:
            block = _node_block(node, cell_size, out=block_buffer)
        else:
            block = _node_block(node, cell_size, out=None)
            if block is not None:
                block_buffer = block
                last_block_size = cell_size_tuple
        
        if block is None:
            continue
        x0, y0, z0 = mins_idx
        x1, y1, z1 = mins_idx + cell_size
        volume[x0:x1, y0:y1, z0:z1] = block
        block_min = float(block.min())
        block_max = float(block.max())
        if raw_min is None or block_min < raw_min:
            raw_min = block_min
        if raw_max is None or block_max > raw_max:
            raw_max = block_max
    if raw_min is None:
        raw_min = 0.0
    if raw_max is None:
        raw_max = raw_min
    return volume, raw_min, raw_max


__all__ = [
    "PlenoxelNode",
    "build_plenoxel_grid",
    "PlenoxelRenderer",
    "plenoxel_volume_from_nodes",
]
