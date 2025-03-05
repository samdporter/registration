import argparse
import glob
import os
import subprocess
import sys

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np

from cil.framework import BlockDataContainer
from sirf.Reg import NiftyAladinSym
from sirf.STIR import AcquisitionData, ImageData

def create_spect_uniform_image(sinogram, origin=None):
    """
    Create a uniform image for SPECT data based on the sinogram dimensions.

    Adjusts the z-direction voxel size and image dimensions to create a template
    image.

    Args:
        sinogram (AcquisitionData): The SPECT sinogram.
        origin (tuple, optional): The origin of the image. Defaults to (0, 0, 0)
            if not provided.

    Returns:
        ImageData: A uniform SPECT image initialized with the computed dimensions
            and voxel sizes.
    """
    # Create a uniform image from the sinogram and adjust z-voxel size.
    image = sinogram.create_uniform_image(1)
    voxel_size = list(image.voxel_sizes())
    voxel_size[0] *= 2  # Adjust z-direction voxel size.

    # Compute new dimensions based on the uniform image.
    dims = list(image.dimensions())
    dims[0] = dims[0] // 2 + dims[0] % 2  # Halve the first dimension (with rounding)
    dims[1] -= dims[1] % 2                # Ensure even number for second dimension
    dims[2] = dims[1]                     # Set third dimension equal to second dimension

    if origin is None:
        origin = (0, 0, 0)

    # Initialize a new image with computed dimensions, voxel sizes, and origin.
    new_image = ImageData()
    new_image.initialise(tuple(dims), tuple(voxel_size), tuple(origin))
    return new_image

def plot_image_info(image, save_path, title=None):
    """
    Plot mid-slices of a 3D image and overlay geometry information.

    The function plots:
      - The central slice in each spatial dimension (x, y, z).
      - The direction matrix using a custom color map (-1 as red, 0 as white, and 1 as blue)
        with the numerical values overlaid.

    Args:
        image: An image object with methods `get_geometrical_info()` and `as_array()`,
            and an attribute `shape` (e.g., a PET/SPECT ImageData).
        save_path (str): File path (including filename) where the plot will be saved.
        title (str, optional): A main title for the figure.
    """
    # Retrieve geometry information from the image.
    geom_info = image.get_geometrical_info()
    info_titles = [
        f"Offset: {geom_info.get_offset()}",
        f"Spacing: {geom_info.get_spacing()}",
        f"Size: {image.shape}",
        "Direction matrix"
    ]

    # Create a figure with 4 subplots.
    fig, axes = plt.subplots(1, 4, figsize=(15, 5))
    arr = image.as_array()

    axims = []
    # Plot central slices along each dimension.
    axims.append(axes[0].imshow(arr[arr.shape[0] // 2], cmap="gray"))
    axims.append(axes[1].imshow(arr[:, arr.shape[1] // 2], cmap="gray"))
    axims.append(axes[2].imshow(arr[:, :, arr.shape[2] // 2], cmap="gray"))

    # add colorbars
    for ax, im in zip(axes, axims):
        fig.colorbar(im, ax=ax, shrink=0.7)

    # Get the direction matrix and plot it.
    direction_matrix = geom_info.get_direction_matrix()

    # Define a custom colormap: red for -1, white for 0, and blue for 1.
    cmap = ListedColormap(["red", "white", "blue"])
    norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], cmap.N)
    axes[3].imshow(direction_matrix, cmap=cmap, norm=norm)

    # Overlay the numerical values on the direction matrix plot.
    for (i, j), val in np.ndenumerate(direction_matrix):
        axes[3].text(
            j, i, f"{val:.0f}", ha="center", va="center",
            color="black", fontsize=12
        )

    # Set titles for each subplot and remove axes for clarity.
    for ax, sub_title in zip(axes, info_titles):
        ax.set_title(sub_title)
        ax.axis("off")

    # Set a main title for the entire figure if provided.
    if title is not None:
        fig.suptitle(title)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def main():
    """
    Main function to process and register PET and SPECT images.

    The workflow is as follows:
      1. Parse command-line arguments.
      2. Insert custom source paths and import local utility operators.
      3. Process PET data:
         - Load sinograms, create uniform images, apply couch shifts, and combine images.
         - Process CTAC data and convert to an attenuation map (u-map).
         - Zoom u-maps to match template images and optionally save plot information.
      4. Process SPECT data:
         - Load and threshold CTAC data, convert to u-map using modality-specific parameters.
         - Resample SPECT u-map to PET resolution.
         - Create a uniform SPECT image template, adjust offsets, and align images.
         - Optionally save plots of the processed images.
      5. Perform rigid registration of SPECT u-map to PET u-map.
      6. Save transformation matrices and registered images.
      7. Optionally plot and save the difference between the PET and registered SPECT images.
    """
    # Parse command-line arguments.
    parser = argparse.ArgumentParser(description="Register PET and SPECT images")
    parser.add_argument(
        "--data_path",
        type=str,
        default="/home/storage/prepared_data/phantom_data/",
        help="Path to the input data.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="/home/storage/prepared_data/phantom_data/",
        help="Path where output files will be saved.",
    )
    parser.add_argument(
        "--patient",
        type=str,
        default="nema_phantom_data",
        help="Patient name for data processing.",
    )
    parser.add_argument(
        "--spect_sub_phantom",
        type=str,
        default="",
        help="Sub-phantom name for data processing.",
    )
    parser.add_argument(
        "--pet_sub_phantom",
        type=str,
        default="",
        help="Sub-phantom name for data processing.",
    )
    # be careful with this. Not very cleverly implemented
    parser.add_argument(
        "--source_path",
        type=str,
        default="/home/sam/working/BSREM_PSMR_MIC_2024/src",
        help="Path to the source code for additional utilities.",
    )
    parser.add_argument(
        "--working_path",
        type=str,
        default="/home/storage/prepared_data/tmp",
        help="Path to the temporary working directory.",
    )
    parser.add_argument(
        "--save_plots",
        action="store_true",
        help="Flag to save diagnostic plots.",
    )
    parser.add_argument(
        "--levels",
        type=int,
        default=4,
        help="Number of levels for reconstruction.",
    )
    parser.add_argument(
        "--levels_to_perform",
        type=int,
        default=4,
        help="Number of levels to actually perform.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=50,
        help="Number of iterations for reconstruction.",
    )
    parser.add_argument(
        "--xy_image_size",
        type=int,
        default=167,
        help="XY dimension size for the images.",
    )
    parser.add_argument(
        "--zooms",
        type=lambda s: list(map(float, s.split(","))),
        default=[1, 2, 2],
        help="Zoom factors in z,y,x (comma-separated).",
    )
    args, _ = parser.parse_known_args()
    
    # Insert the source path for custom utilities and import them.
    sys.path.insert(0, args.source_path)
    from utilities.shifts import (
        CouchShiftOperator,
        ImageCombineOperator,
        get_couch_shift_from_sinogram,
    )

    pet_switch=True

    try:
        pet_sino = AcquisitionData(os.path.join
                                    (
                                    args.data_path,
                                    args.patient,
                                    "PET",
                                    args.pet_sub_phantom,
                                    f"prompts.hs",
                                    )
                                )
    except:
        print("No PET sinogram found")
        pet_switch=False

    if pet_switch: 

        # Create uniform images for each sinogram.
        pet_image = pet_sino.create_uniform_image(1, args.xy_image_size)

        # Write template images for each bed position.
        pet_image.write(
            os.path.join(
                args.output_path, 
                args.patient, 
                "PET", 
                args.pet_sub_phantom,
                "template_image.hv"
            )
        )

        print(
            f"Created image with size {pet_image.shape} and "
            f"voxel size {pet_image.voxel_sizes()}"
        )

        # ---------------------- PET CTAC Processing and U-map Generation ----------------------
        search_pattern = os.path.join(
            args.data_path, args.patient, "PET", "ctac", "*.dcm"
        )
        print(f"Searching for files with pattern: {search_pattern}")
        matching_files = glob.glob(search_pattern)
        if matching_files:
            pet_ctac = ImageData(matching_files[0])
            pet_ctac = pet_ctac.maximum(-1000)
        else:
            print("No matching CTAC files found for PET.")
            pet_ctac = None

        # Convert CTAC to attenuation (u-map) if the CTAC image was found.
        if pet_ctac is not None:
            pet_ctac.write(
                os.path.join(
                    args.output_path, 
                    args.patient, 
                    "PET", 
                    args.pet_sub_phantom,
                    "ctac_FoV_corrected.hv"
                )
            )
            try:
                subprocess.run(
                    [
                        "ctac_to_mu_values",
                        "-o",
                        os.path.join(
                            args.output_path, 
                            args.patient, 
                            "PET", 
                            args.pet_sub_phantom,
                            "umap.hv"
                        ),
                        "-i",
                        os.path.join(
                            args.output_path,
                            args.patient,
                            "PET", 
                            args.pet_sub_phantom,
                            "ctac_FoV_corrected.hv",
                        ),
                        "-j",
                        "/home/sam/devel/SIRF_builds/master/build/sources/STIR/src/config/ct_slopes.json",
                        "-m",
                        "GE",
                        "-v",
                        "80",
                        "-k",
                        "511",
                    ],
                    check=True,
                )
            except subprocess.CalledProcessError as e:
                print(f"Error converting PET CTAC to u-map: {e}")

        # Load the generated PET u-map and zoom to match the combined image template.
        pet_umap = ImageData(
            os.path.join(
                args.output_path, 
                args.patient, 
                "PET", 
                args.pet_sub_phantom,
                "umap.hv")
        )
        
        # we need to get rid of the offsets
        pet_umap_tmp = ImageData()
        pet_umap_offsets = pet_umap.get_geometrical_info().get_offset()
        offset_z = (
            (pet_umap.voxel_sizes()[0] * pet_umap.shape[0])
            - (pet_image.voxel_sizes()[0]
            * pet_image.shape[0])
        ) / 2
        pet_umap_tmp.initialise(pet_umap.shape, pet_umap.voxel_sizes(), (-offset_z, 0, 0))
        pet_umap_tmp.fill(pet_umap.as_array())
        pet_umap = pet_umap_tmp    
        
        pet_umap_zoomed2pet = pet_umap.zoom_image_as_template(
            pet_image, scaling="preserve_values"
        )

        if args.save_plots:
            plot_image_info(
                pet_umap,
                os.path.join(
                    args.output_path,
                    args.patient, 
                    "PET", 
                    args.pet_sub_phantom,
                    "umap_info.png"
                ),
                "PET u-map",
            )
            plot_image_info(
                pet_umap_zoomed2pet,
                os.path.join(
                    args.output_path, 
                    args.patient, 
                    "PET", 
                    args.pet_sub_phantom,
                    "umap_zoomed_info.png"
                ),
                "PET u-map zoomed to combined PET",
            )

    # ---------------------- SPECT Data Processing ----------------------
    
    # Load the SPECT CTAC image and threshold it.
    search_pattern = os.path.join(
        args.data_path, 
        args.patient, "SPECT", 
        args.spect_sub_phantom, 
        "ctac", 
        "*.dcm"
    )
    print(f"Searching for files with pattern: {search_pattern}")
    matching_files = glob.glob(search_pattern)
    if matching_files:
        spect_ctac = ImageData(matching_files[0])
        spect_ctac = spect_ctac.maximum(-1000)
    else:
        print("No matching CTAC files found for SPECT.")
        spect_ctac = None
        
    if spect_ctac is not None:
        spect_ctac.write(
            os.path.join(
                args.output_path, 
                args.patient, 
                "SPECT",
                args.spect_sub_phantom, 
                "ctac_FoV_corrected.hv"
            )
        )
    else:
        print("No matching CTAC files found for SPECT.")
        sys.exit(1)
    
    # Convert SPECT CTAC to u-map using Mediso-specific parameters
    
    
    try:
        subprocess.run(
            [
                "ctac_to_mu_values",
                "-o",
                os.path.join(args.output_path, 
                             args.patient, 
                             "SPECT", 
                             args.spect_sub_phantom, 
                             "umap.hv"
                ),
                "-i",
                os.path.join(
                    args.output_path,
                    args.patient,
                    "SPECT",
                    args.spect_sub_phantom, 
                    "ctac_FoV_corrected.hv",
                ),
                "-j",
                "/home/sam/devel/SIRF_builds/master/build/sources/STIR/src/config/ct_slopes.json",
                "-m",
                "Mediso",
                "-v",
                "120",
                "-k",
                "80",
            ],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"Error converting SPECT CTAC to u-map: {e}")

    # Load the SPECT u-map.
    spect_umap = ImageData(
        os.path.join(
            args.output_path, 
            args.patient, 
            "SPECT", 
            args.spect_sub_phantom, 
            "umap.hv"
        )
    
    )
    
    # Create a uniform SPECT image template from the sinogram.
    spect_sino = AcquisitionData(
        os.path.join(
            args.data_path, 
            args.patient, 
            "SPECT", 
            args.spect_sub_phantom, 
            "peak.hs"
        )
    )
    spect_image = create_spect_uniform_image(spect_sino)

    # Write the template image for the SPECT data.
    spect_image.write(
        os.path.join(
            args.output_path, 
            args.patient, 
            "SPECT", 
            args.spect_sub_phantom, 
            "template_image.hv"
        )
    )
    
    # we need to get rid of the offsets
    spect_umap_tmp = ImageData()
    spect_umap_offsets = spect_umap.get_geometrical_info().get_offset()
    offset_z = (
        (spect_umap.voxel_sizes()[0] * spect_umap.shape[0])
        - (spect_image.voxel_sizes()[0]
           * spect_image.shape[0])
    ) / 2
    spect_umap_tmp.initialise(spect_umap.shape, spect_umap.voxel_sizes(), (-offset_z, 0, 0))
    spect_umap_tmp.fill(spect_umap.as_array())
    spect_umap = spect_umap_tmp
    
    if args.save_plots:
        plot_image_info(
            spect_umap,
            os.path.join(
                args.output_path, 
                args.patient, 
                "SPECT", 
                args.spect_sub_phantom, 
                "umap_info.png"
            ),
            "SPECT u-map",
        )
        
    spect_umap_zoomed2spect = spect_umap.zoom_image_as_template(
        spect_image, scaling="preserve_values"
    )

    spect_umap_zoomed2spect.write(
        os.path.join(
            args.output_path, 
            args.patient, 
            "SPECT", 
            args.spect_sub_phantom, 
            "umap_zoomed.hv"
        )
    )

    if args.save_plots:
        plot_image_info(
            spect_umap_zoomed2spect,
            os.path.join(
                args.output_path, 
                args.patient, 
                "SPECT", 
                args.spect_sub_phantom, 
                "umap_zoomed2spect_info.png"
            ),
            "SPECT u-map zoomed to SPECT resolution",
        )

    # ---------------------- Registration of SPECT to PET ----------------------
    if pet_switch:
        reg = NiftyAladinSym()
        reg.set_reference_image(pet_umap_zoomed2pet)
        reg.set_floating_image(spect_umap_zoomed2spect)
        reg.set_parameter("SetPerformRigid", "1")
        reg.set_parameter("SetPerformAffine", "0")
        reg.process()

        # Save the transformation matrix and the registered SPECT u-map.
        transformation = reg.get_transformation_matrix_forward()
        transformation.write(
            os.path.join(
                args.output_path, 
                args.patient, 
                "SPECT", 
                args.spect_sub_phantom, 
                "spect2pet.tfm")
        )
        spect_umap_zoomed_registered = reg.get_output()
        spect_umap_zoomed_registered.write(
            os.path.join(
                args.output_path, 
                args.patient, 
                "SPECT", 
                args.spect_sub_phantom, 
                "umap_zoomed_registered.hv")
        )

        if args.save_plots:
            plot_image_info(
                spect_umap_zoomed_registered,
                os.path.join(
                    args.output_path, 
                    args.patient, 
                    "SPECT", 
                    args.spect_sub_phantom, 
                    "umap_zoomed_registered_info.png"
                ),
                "SPECT u-map registered to PET",
            )

        # ---------------------- Plot Difference Between PET and Registered SPECT ----------------------
        # normalise by 95th percentile of each image
        pet_array = pet_umap_zoomed2pet.as_array()
        pet_normed_array = pet_array / 0.096

        spect_array = spect_umap_zoomed_registered.as_array()
        spect_normed_array = spect_array / 0.16

        diff = pet_normed_array - spect_normed_array
        diff_minmax = max(abs(diff.min()), abs(diff.max())) / 2

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        im0 = axes[0].imshow(
            diff[diff.shape[0] // 2],
            vmin=-diff_minmax,
            vmax=diff_minmax,
            cmap="seismic",
        )
        axes[1].imshow(
            diff[:, diff.shape[1] // 2],
            vmin=-diff_minmax,
            vmax=diff_minmax,
            cmap="seismic",
        )
        axes[2].imshow(
            diff[:, :, diff.shape[2] // 2],
            vmin=-diff_minmax,
            vmax=diff_minmax,
            cmap="seismic",
        )
        # cbar on left of first plot
        cbar = fig.colorbar(
            im0, 
            ax=axes[0], 
            orientation="vertical", 
            location="left", 
            shrink=0.7
        )
        cbar.set_label("PET - SPECT")

        # sup title
        fig.suptitle("Normed PET - Normed Registered SPECT")

        # axis off
        for ax in axes:
            ax.axis("off")

        plt.savefig(
            os.path.join(
                args.output_path, 
                args.patient, 
                "SPECT", 
                args.spect_sub_phantom, 
                "umap_zoomed_diff.png"
            )
        )
        plt.savefig(
            os.path.join(
                args.output_path, 
                args.patient, 
                "PET", 
                args.pet_sub_phantom,
                "umap_zoomed_diff.png"
            )
        )


if __name__ == "__main__":
    main()
