from pytomography.metadata.SPECT import SPECTObjectMeta, SPECTProjMeta
from pytomography.projectors.SPECT import SPECTSystemMatrix
from pytomography.likelihoods import PoissonLogLikelihood
from pytomography.algorithms import OSEM
import numpy as np
import torch
import sys

# Load data
raw = np.fromfile(sys.argv[1], dtype=np.float32)
projections = torch.tensor(raw.reshape(1, 120, 128, 128))

# Metadata (from sim)
object_meta = SPECTObjectMeta(dr=(4.42, 4.42, 4.42), shape=(128, 128, 64))

angles = np.concatenate([
    np.linspace(0,   87,  30, endpoint=True),
    np.linspace(90,  177, 30, endpoint=True),
    np.linspace(180, 267, 30, endpoint=True),
    np.linspace(270, 357, 30, endpoint=True),
])
proj_meta = SPECTProjMeta(angles=angles, radii=400.0, dr=(4.42, 4.42))

# System matrix built from metadata
system_matrix = SPECTSystemMatrix(
    obj2obj_transforms=[],          # e.g. attenuation/PSF operators go here
    proj2proj_transforms=[],
    object_meta=object_meta,
    proj_meta=proj_meta,
)

# OSEM
likelihood = PoissonLogLikelihood(system_matrix, projections)
algorithm = OSEM(likelihood)
recon = algorithm(n_iters=4, n_subsets=8)  # shape: (1, 128, 128, 64)