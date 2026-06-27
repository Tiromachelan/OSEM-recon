"""
OSEM reconstruction of a SPECT sinogram produced by the GATE simulation in
"../Simulation Pipeline".

Geometry (taken directly from gateSimQuadHead.py):
  * 4 detector heads, 30 views each -> 120 projections total.
      head 1: 0,3,...,87   head 2: 90,...,177
      head 3: 180,...,267  head 4: 270,...,357   (3 deg step)
    combineRawFiles.py concatenates the 4 heads in order, so the combined
    sinogram angle order is simply 0,3,6,...,357.
  * Orbit radius        : 40 cm  (= 400 mm)
  * Detector            : 128 x 128 pixels
  * Pixel spacing       : 2.21 mm * 2 = 4.42 mm   (add_digitizer spacing)

The simulation writes the projections (ITK/MHD .raw, x varies fastest), so a
flat reshape gives axes (angle, detector_y, detector_x). The gantry orbits
about world-Z, and each head's initial rotation maps the detector y-axis onto
world-Z. Hence detector_y is the *axial* direction and detector_x is the
*transaxial* direction. pytomography expects projections ordered
(angle, r=transaxial, z=axial), so we swap the two detector axes.

Usage:
    python main.py [sinogram.raw] [n_iters] [n_subsets]
"""

from pytomography.metadata.SPECT import SPECTObjectMeta, SPECTProjMeta
from pytomography.projectors.SPECT import SPECTSystemMatrix
from pytomography.likelihoods import PoissonLogLikelihood
from pytomography.algorithms import OSEM
import numpy as np
import torch
import sys

# ---------------------------------------------------------------------------
# Acquisition / detector parameters (must match the simulation)
# ---------------------------------------------------------------------------
N_PROJECTIONS = 120          # 4 heads x 30 views
DETECTOR_SHAPE = (128, 128)  # (detector pixels per side)
PIXEL_SPACING = 4.42         # mm, = 2.21 mm * 2
ORBIT_RADIUS = 400.0         # mm, = 40 cm
VOXEL_SIZE = PIXEL_SPACING   # mm; reconstruct on the detector-pixel grid

# ---------------------------------------------------------------------------
# Orientation conventions
# ---------------------------------------------------------------------------
# pytomography and other reconstruction programs (e.g. an external MLEM tool)
# do not share a single coordinate convention, so the reconstructed volume can
# come out rotated and/or mirrored relative to a reference. There are exactly
# two *physical* degrees of freedom to reconcile; set them to match your
# reference. Everything else (np.rot90/np.flip on the final image) is cosmetic
# display orientation and does not change the physics.
#
#   TRANSPOSE_DETECTOR_AXES
#       Which detector axis is axial vs transaxial. The GATE .raw is written
#       x-fastest, giving (angle, detector_y, detector_x); the gantry orbits
#       world-Z and each head's initial rotation maps detector_y -> world-Z,
#       so detector_y is axial. pytomography wants (angle, transaxial, axial),
#       hence we swap the two detector axes. Toggling this rotates the coronal
#       view by 90 degrees.
#
#   REVERSE_GANTRY_SENSE
#       The direction the gantry rotates. pytomography assumes one handedness;
#       if the acquisition (or the other program) uses the opposite sense the
#       volume is MIRRORED (patient left/right swapped). A mirror cannot be
#       removed by any rotation, so this must be corrected here, at the source,
#       not by flipping the displayed image. Implemented by flipping the
#       transaxial detector axis (equivalent to negating the projection angles).
TRANSPOSE_DETECTOR_AXES = True
REVERSE_GANTRY_SENSE = False

# ---------------------------------------------------------------------------
# Command-line arguments
# ---------------------------------------------------------------------------
sinogram_path = sys.argv[1] if len(sys.argv) > 1 else "example_sinogram.raw"
n_iters = int(sys.argv[2]) if len(sys.argv) > 2 else 4
n_subsets = int(sys.argv[3]) if len(sys.argv) > 3 else 8

# ---------------------------------------------------------------------------
# Load the sinogram
# ---------------------------------------------------------------------------
raw = np.fromfile(sinogram_path, dtype=np.float32)
expected = N_PROJECTIONS * DETECTOR_SHAPE[0] * DETECTOR_SHAPE[1]
if raw.size != expected:
    raise ValueError(
        f"{sinogram_path} has {raw.size} float32 values but the configured "
        f"geometry expects {expected} "
        f"({N_PROJECTIONS} x {DETECTOR_SHAPE[0]} x {DETECTOR_SHAPE[1]})."
    )

# Raw layout is (angle, detector_y=axial, detector_x=transaxial).
sinogram = raw.reshape(N_PROJECTIONS, DETECTOR_SHAPE[0], DETECTOR_SHAPE[1])
if TRANSPOSE_DETECTOR_AXES:
    # -> (angle, transaxial, axial), the order pytomography expects.
    sinogram = np.transpose(sinogram, (0, 2, 1))
if REVERSE_GANTRY_SENSE:
    # Flip the transaxial axis to invert rotational handedness (un-mirror).
    sinogram = np.flip(sinogram, axis=2)
projections = torch.tensor(np.ascontiguousarray(sinogram))

# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------
# Object is reconstructed on a cubic grid. The transaxial dimensions (x, y)
# must be equal (the object is rotated about its z-axis during projection) and
# match the transaxial detector size; the z-axis matches the axial detector
# size.
object_meta = SPECTObjectMeta(
    dr=(VOXEL_SIZE, VOXEL_SIZE, VOXEL_SIZE),
    shape=(DETECTOR_SHAPE[1], DETECTOR_SHAPE[1], DETECTOR_SHAPE[0]),  # (x, y, z)
)

# Detector angle for every projection: 0, 3, ..., 357 degrees.
angles = np.arange(N_PROJECTIONS) * (360.0 / N_PROJECTIONS)

proj_meta = SPECTProjMeta(
    projection_shape=DETECTOR_SHAPE,          # (transaxial, axial)
    dr=(PIXEL_SPACING, PIXEL_SPACING),
    angles=angles,
    radii=np.full(N_PROJECTIONS, ORBIT_RADIUS),
)

# ---------------------------------------------------------------------------
# System matrix (no attenuation / PSF modelling yet -- add transforms here)
# ---------------------------------------------------------------------------
system_matrix = SPECTSystemMatrix(
    obj2obj_transforms=[],   # e.g. attenuation / PSF operators go here
    proj2proj_transforms=[],
    object_meta=object_meta,
    proj_meta=proj_meta,
)

# ---------------------------------------------------------------------------
# OSEM reconstruction
# ---------------------------------------------------------------------------
likelihood = PoissonLogLikelihood(system_matrix, projections)
algorithm = OSEM(likelihood)
recon = algorithm(n_iters=n_iters, n_subsets=n_subsets)  # shape: (x, y, z)

recon_np = recon.cpu().numpy().astype(np.float32)
print(
    f"Reconstruction complete: shape={tuple(recon_np.shape)} "
    f"min={recon_np.min():.4g} max={recon_np.max():.4g} sum={recon_np.sum():.4g}"
)

# Save result next to the input sinogram.
output_path = "reconstruction.raw"
recon_np.tofile(output_path)
print(f"Saved reconstruction to {output_path}")
