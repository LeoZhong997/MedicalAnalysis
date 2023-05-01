"""
For the sake of speed, we’ll use a 2D dataset here,
although needless to say the workflow would be identical for 3D data.

We’ll generate the data by taking Decathlon’s 3D brain tumor dataset,
taking the 2D slice containing the most voxels > 0 (the most label),
and then saving the new dataset to disk.

After that, we’ll do normal 2D training with a few augmentations,
which means that we’ll be able to benefit from the inverse transformations.
"""



