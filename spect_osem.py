#!/usr/bin/env python3
"""
Reconstruct SPECT data using OSEM (OSMAPOSLReconstructor).

This script loads acquisition, attenuation, and (optionally) initial image data,
sets up an acquisition model, and runs the reconstruction.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from sirf.STIR import (
    AcquisitionData, ImageData, SPECTUBMatrix, AcquisitionModelUsingMatrix,
    OSMAPOSLReconstructor, make_Poisson_loglikelihood,
    SeparableGaussianImageFilter,
)




def create_spect_uniform_image(sinogram, offset=0):
    """
    Create a uniform image for SPECT data from a sinogram.

    Args:
        sinogram (AcquisitionData): The SPECT sinogram.
        offset (float, optional): Offset for the image origin in the z-direction.

    Returns:
        ImageData: A uniform SPECT image.
    """
    image = sinogram.create_uniform_image(1)
    voxel_size = list(image.voxel_sizes())
    voxel_size[0] *= 2  # Adjust z-direction voxel size.
    dims = list(image.dimensions())
    dims[0] = dims[0] // 2 + dims[0] % 2
    dims[1] -= dims[1] % 2
    dims[2] = dims[1]
    origin = (offset, 0, 0)
    new_image = ImageData()
    new_image.initialise(tuple(dims), tuple(voxel_size), tuple(origin))
    return new_image


def get_spect_data(data_dir):
    """
    Load SPECT acquisition, attenuation, and initial image data from the given directory.

    Args:
        data_dir (str): Path to the data directory.

    Returns:
        dict: A dictionary with keys 'acquisition_data', 'attenuation', and 'initial_image'.
    """
    spect_data = {}
    acq_path = os.path.join(data_dir, "peak.hs")
    spect_data["acquisition_data"] = AcquisitionData(acq_path)

    # Try to load attenuation data.
    attn_path = os.path.join(data_dir, "umap_zoomed.hv")
    try:
        spect_data["attenuation"] = ImageData(attn_path)
    except Exception:
        print("No attenuation data found.")

    # Try to load an initial image; if unavailable, create a uniform image.
    init_img_path = os.path.join(data_dir, "initial_image.hv")
    try:
        init_image = ImageData(init_img_path)
        spect_data["initial_image"] = init_image.maximum(0)
    except Exception:
        print("No initial image found. Creating a uniform image from the acquisition data.")
        spect_data["initial_image"] = create_spect_uniform_image(
            spect_data["acquisition_data"]
        )

    return spect_data


def get_spect_acquisition_model(spect_data, keep_all_views=True):
    """
    Create an acquisition model for SPECT reconstruction using a SPECTUBMatrix.

    Args:
        spect_data (dict): Dictionary containing SPECT data.
        keep_all_views (bool, optional): Whether to keep all views in cache.

    Returns:
        AcquisitionModelUsingMatrix: The SPECT acquisition model.
    """
    spect_mat = SPECTUBMatrix()
    try:
        spect_mat.set_attenuation_image(spect_data["attenuation"])
    except Exception:
        print("No attenuation data available for the acquisition model.")

    spect_mat.set_keep_all_views_in_cache(keep_all_views)
    # Set the resolution model parameters (example values).
    spect_mat.set_resolution_model(0.9323, 0.03, False)

    spect_am = AcquisitionModelUsingMatrix(spect_mat)
    try:
        spect_am.set_additive_term(spect_data["additive"])
    except Exception:
        print("No additive data found for the acquisition model.")

    return spect_am


def get_reconstructor(acquisition_data, acq_model, initial_image, num_subsets, num_epochs):
    """
    Create and set up an OSMAPOSLReconstructor.

    Args:
        acquisition_data (AcquisitionData): The SPECT acquisition data.
        acq_model (AcquisitionModelUsingMatrix): The SPECT acquisition model.
        initial_image (ImageData): The initial image.
        num_subsets (int): Number of subsets.
        num_epochs (int): Number of epochs.

    Returns:
        OSMAPOSLReconstructor: The configured reconstructor.
    """
    reconstructor = OSMAPOSLReconstructor()
    poisson_ll = make_Poisson_loglikelihood(acq_data=acquisition_data, acq_model=acq_model)
    reconstructor.set_objective_function(poisson_ll)
    reconstructor.set_num_subsets(num_subsets)
    reconstructor.set_num_subiterations(num_subsets * num_epochs)
    reconstructor.set_up(initial_image)
    return reconstructor


def main(args):
    """
    Main processing function for SPECT reconstruction.

    Args:
        data_path (str): Path to the SPECT data directory.

    Returns:
        ImageData: The reconstructed image.
    """
    spect_data = get_spect_data(args.data_path)
    spect_am = get_spect_acquisition_model(spect_data, keep_all_views=True)
    initial_image = spect_data["initial_image"]
    acquisition_data = spect_data["acquisition_data"]

    reconstructor = get_reconstructor(acquisition_data, spect_am, initial_image, num_subsets=args.num_subsets, num_epochs=args.num_epochs)
    reconstructor.reconstruct(initial_image)

    recon_image = reconstructor.get_current_estimate()
    return recon_image


if __name__ == "__main__":

    # create arguments
    parser = argparse.ArgumentParser(description='Reconstruct with OSEM')
    parser.add_argument(
        '--data_path', type=str, 
        default='/home/storage/copied_data/data/anthropomorphic_Y90/SPECTY90phantoms/phantom_140', 
        help='Path to the SPECT data directory'
    )
    parser.add_argument('--num_subsets', type=int, default=12, help='Number of subsets')
    parser.add_argument('--num_epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--smoothing_kernel', type=tuple, default=None, help='Smoothing kernel FWHM in mm')

    args, unknown = parser.parse_known_args()

    args = parser.parse_args()

    reconstruction = main(args)

    if args.smoothing_kernel:
        gauss = SeparableGaussianImageFilter()
        gauss.set_fwhms(args.smoothing_kernel)
        gauss.apply(reconstruction)
        suffix = f"_gauss{args.smoothing_kernel[0]}"
    else:
        suffix = ""

    output_filename = os.path.join(args.data_path, f"spect_recon{suffix}.hv")
    reconstruction.write(output_filename)
