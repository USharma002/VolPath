# VolPath

This is a volume renderer with scalar field reconstruction.
The omtivation for this came from the unstructured point clod data and how to resample it on a grid for Volume Rendering

While many interpolators are there this repository provides the folloing ways:
- Linear Interpolation
- Nearest Neighbor
- Gaussian RBF
- Natural Neigbhor
- Neural Representation (Training, Saving and Checkpoint in UI)
- Octree
- Point Cloud Visualization

The code is meant to work with scientific data and currently supports ```hdf5``` and ```vtk``` file formats.
The original idea was to make it work with the IllustrisTNG Dataset and have visualized a Subhalo from TNG100-1.

