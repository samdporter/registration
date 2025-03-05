import argparse
import glob
import os
import subprocess
import sys

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np
import nibabel as nib
from totalsegmentator.python_api import totalsegmentator
from cil.framework import BlockDataContainer
from sirf.Reg import NiftyAladinSym, NiftyF3dSym
from sirf.Reg import ImageData as RegImageData
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

    # Plot central slices along each dimension.
    axes[0].imshow(arr[arr.shape[0] // 2], cmap="gray")
    axes[1].imshow(arr[:, arr.shape[1] // 2], cmap="gray")
    axes[2].imshow(arr[:, :, arr.shape[2] // 2], cmap="gray")

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


def crop_ct_to_body(ct_image):
    """
    Crop a CT image to the body region using the TotalSegmentator.

    Args:
        ct_image (ImageData): The CT image to be cropped.

    Returns:
        ImageData: The cropped CT image.
    """
    # Convert the CT image to a Nifti file.
    ct_nifty = RegImageData(ct_image)
    ct_nifty.write("ct_image_tmp.nii")
    ct_nifty = nib.load("ct_image_tmp.nii")

    seg = totalsegmentator(ct_nifty, body_seg=True, task='body')
    data = (seg.get_fdata() == 1)

    data = np.rot90(data, axes=(0, 2))
    data = np.flip(data, axis=0)

    ct_arr = ct_image.as_array()
    ct_arr[~data] = -1000

    out = ct_image.clone()
    out.fill(ct_arr)

    return out

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
      6. Save displacement fields and registered images.
      7. Optionally plot and save the difference between the PET and registered SPECT images.
    """
    # Parse command-line arguments.
    parser = argparse.ArgumentParser(description="Register PET and SPECT images")
    parser.add_argument(
        "--data_path",
        type=str,
        default="/home/storage/prepared_data/oxford_patient_data",
        help="Path to the input data.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="/home/storage/prepared_data/oxford_patient_data",
        help="Path where output files will be saved.",
    )
    parser.add_argument(
        "--patient",
        type=str,
        default="sirt4",
        help="Patient name for data processing.",
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
        default="/home/sam/working/BSREM_PSMR_MIC_2024/tmp",
        help="Path to the temporary working directory.",
    )
    parser.add_argument(
        "--use_tof",
        action="store_true",
        help="Flag to use TOF sinograms.",
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
        default=288,
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

    # Use TOF or non-TOF data based on flag.
    tof_str = "tof" if args.use_tof else "non_tof"

    # ---------------------- PET Data Processing ----------------------
    # Load PET sinograms for different bed positions.
    bed_pos = ["f1b1", "f2b1"]
    pet_sinos = [
        AcquisitionData(
            os.path.join(
                args.data_path,
                args.patient,
                "PET",
                tof_str,
                f"prompts_{bp}.hs",
            )
        )
        for bp in bed_pos
    ]

    # Create uniform images for each sinogram.
    pet_images = [
        sino.create_uniform_image(1, args.xy_image_size) for sino in pet_sinos
    ]

    # Write template images for each bed position.
    for image, pos in zip(pet_images, bed_pos):
        image.write(
            os.path.join(
                args.output_path,
                args.patient,
                "PET",
                f"template_image_{pos}.hv",
            )
        )

    print(
        f"Created images with size {pet_images[0].as_array().shape} and "
        f"voxel sizes {pet_images[0].voxel_sizes()}"
    )

    # Get couch shifts from sinograms.
    pet_shifts = [get_couch_shift_from_sinogram(sino) for sino in pet_sinos]
    print(f"Shifts: {pet_shifts}")

    # Apply couch shift corrections to the images.
    pet_shift_ops = [
        CouchShiftOperator(image, shift)
        for image, shift in zip(pet_images, pet_shifts)
    ]
    pet_shifted_images = [
        op.direct(image)
        for op, image in zip(pet_shift_ops, pet_images)
    ]

    # Combine the shifted images into one.
    pet_combine_op = ImageCombineOperator(
        BlockDataContainer(*pet_shifted_images)
    )
    pet_combined_image = pet_combine_op.direct(
        BlockDataContainer(*pet_shifted_images)
    )
    print(
        f"Combined image with size {pet_combined_image.as_array().shape} and "
        f"voxel sizes {pet_combined_image.voxel_sizes()}"
    )

    # Create a template by filling the combined image.
    pet_combined_image.fill(1)
    pet_combined_image.write(
        os.path.join(
            args.output_path,
            args.patient,
            "PET",
            "combined_template_image.hv",
        )
    )

    # ---------------------- PET CTAC Processing and U-map Generation ----------------------
    search_pattern = os.path.join(
        args.data_path, args.patient, "PET", "ctac", "i42*CTDC*.img"
    )
    print(f"Searching for files with pattern: {search_pattern}")
    matching_files = glob.glob(search_pattern)
    if matching_files:
        pet_ctac = ImageData(matching_files[0])
        pet_ctac = pet_ctac.maximum(-1000)
        pet_ctac_crop = crop_ct_to_body(pet_ctac)
    else:
        print("No matching CTAC files found for PET.")
        pet_ctac = None

    # Convert CTAC to attenuation (u-map) if the CTAC image was found.
    if pet_ctac is not None:
        pet_ctac.write(
            os.path.join(
                args.output_path, args.patient, "PET", "ctac_FoV_corrected.hv"
            )
        )
        pet_ctac_crop.write(
            os.path.join(
                args.output_path, args.patient, "PET", "ctac_FoV_corrected_crop.hv"
            )
        )
        try:
            subprocess.run(
                [
                    "ctac_to_mu_values",
                    "-o",
                    os.path.join(args.output_path, args.patient, "PET", "umap.hv"),
                    "-i",
                    os.path.join(
                        args.output_path,
                        args.patient,
                        "PET",
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
        os.path.join(args.output_path, args.patient, "PET", "umap.hv")
    )
    pet_umap_zoomed2pet = pet_umap.zoom_image_as_template(
        pet_combined_image, scaling="preserve_values"
    )

    pet_ctac_crop_zoomed2pet = pet_ctac_crop.zoom_image_as_template(
        pet_combined_image, scaling="preserve_values"
    )
    pet_ctac_crop_zoomed2pet.write(
        os.path.join(
            args.output_path, args.patient, "PET", "ctac_FoV_corrected_crop_zoomed.hv"
        )
    )
    if args.save_plots:
        plot_image_info(
            pet_ctac_crop_zoomed2pet,
            os.path.join(
                args.output_path, args.patient, "PET", "ctac_FoV_corrected_crop_zoomed_info.png"
            ),
            "PET CTAC cropped to body and zoomed to PET resolution",
        )

    # Create u-maps for each shifted image by zooming and then unshifting.
    pet_umaps_bed_pos = [
        pet_umap_zoomed2pet.zoom_image_as_template(
            image, scaling="preserve_values"
        )
        for image in pet_shifted_images
    ]
    for pet_umap_item, pos, op in zip(pet_umaps_bed_pos, bed_pos, pet_shift_ops):
        pet_umap_item = op.adjoint(pet_umap_item)
        # Verify that geometrical information remains consistent.
        assert (
            pet_umap_item.get_geometrical_info().get_info()
            == pet_umap_item.get_geometrical_info().get_info()
        )
        pet_umap_item.write(
            os.path.join(
                args.output_path, args.patient, "PET", f"umap_{pos}.hv"
            )
        )

        if args.save_plots:
            plot_image_info(
                pet_umap_item,
                os.path.join(
                    args.output_path, args.patient, "PET", f"umap_{pos}_info.png"
                ),
                f"PET u-map zoomed to {pos}",
            )

    if args.save_plots:
        plot_image_info(
            pet_umap,
            os.path.join(args.output_path, args.patient, "PET", "umap_info.png"),
            "PET u-map",
        )
        plot_image_info(
            pet_umap_zoomed2pet,
            os.path.join(args.output_path, args.patient, "PET", "umap_zoomed_info.png"),
            "PET u-map zoomed to combined PET",
        )

    # ---------------------- SPECT Data Processing ----------------------
    # Load SPECT CTAC image from DICOM.
    search_pattern = os.path.join(
        args.data_path, 
        args.patient, 
        "SPECT", 
        "CTSIRT-ABDO-PELVISTOMOCT",
        "1.2.*.dcm",
    )
    print(f"Searching for files with pattern: {search_pattern}")
    matching_files = glob.glob(search_pattern)
    if matching_files:
        spect_ctac = ImageData(matching_files[0])
        spect_ctac = spect_ctac.maximum(-1000)
        spect_ctac_crop = crop_ct_to_body(spect_ctac)
    else:
        print("No matching CTAC files found for PET.")
        spect_ctac = None

    if spect_ctac is not None:
        spect_ctac.write(
            os.path.join(
                args.output_path, args.patient, "SPECT", "ctac_FoV_corrected.hv"
            )
        )
        spect_ctac_crop.write(
            os.path.join(
                args.output_path, args.patient, "SPECT", "ctac_FoV_corrected_crop.hv"
            )
        )
        if args.save_plots:
            plot_image_info(
                spect_ctac_crop,
                os.path.join(
                    args.output_path, args.patient, "SPECT", "ctac_FoV_corrected_crop_info.png"
                ),
                "SPECT CTAC cropped to body",
            )
    else:
        print("No matching CTAC files found for SPECT.")

    # Convert SPECT CTAC to u-map using Mediso-specific parameters.
    try:
        subprocess.run(
            [
                "ctac_to_mu_values",
                "-o",
                os.path.join(args.output_path, 
                             args.patient, 
                             "SPECT", 
                             "umap.hv"
                ),
                "-i",
                os.path.join(
                    args.output_path,
                    args.patient,
                    "SPECT",
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
        os.path.join(args.output_path, args.patient, "SPECT", "umap.hv")
    )
    if args.save_plots:
        plot_image_info(
            spect_umap,
            os.path.join(args.output_path, args.patient, "SPECT", "umap_info.png"),
            "SPECT u-map",
        )

    # Compute zoom factors to match SPECT u-map to PET resolution.
    zooms = tuple(
        b / a for a, b in zip(pet_umap.voxel_sizes(), spect_umap.voxel_sizes())
    )
    spect_umap_zoomed2pet = spect_umap.zoom_image(
        zooms=zooms, scaling="preserve_values"
    )
    np.savetxt(
        os.path.join(args.output_path, args.patient, "SPECT", "zooms_spect_ct2pet.csv"),
        zooms,
        delimiter=",",
    )
    spect_umap_zoomed2pet.write(
        os.path.join(args.output_path, args.patient, "SPECT", "umap_zoomed2pet.hv")
    )
    if args.save_plots:
        plot_image_info(
            spect_umap_zoomed2pet,
            os.path.join(
                args.output_path, args.patient, "SPECT", "umap_zoomed2pet_info.png"
            ),
            "SPECT u-map zoomed to PET resolution",
        )

    # Create a uniform SPECT image template from the sinogram.
    spect_sino = AcquisitionData(
        os.path.join(args.data_path, args.patient, "SPECT", "peak.hs")
    )
    spect_image_from_sinogram = create_spect_uniform_image(spect_sino)
    spect_image_from_sinogram.write(
        os.path.join(args.output_path, args.patient, "SPECT", "template_image.hv")
    )

    # Compute offset adjustments between the SPECT and PET u-maps.
    # This bit is a little bit horrible but necessary to get the correct offset
    spect_umap_offsets = spect_umap.get_geometrical_info().get_offset()
    offset_z = (
        (spect_umap.voxel_sizes()[0] * spect_umap.shape[0])
        - (spect_image_from_sinogram.voxel_sizes()[0]
           * spect_image_from_sinogram.shape[0])
    ) / 2
    spect_template_image = create_spect_uniform_image(
        spect_sino, origin=(-offset_z, 0, -spect_umap_offsets[0] * 2)
    )

    # Compute zoom factors to match the SPECT u-map to the SPECT template.
    zooms = tuple(
        b / a
        for a, b in zip(
            spect_template_image.voxel_sizes(), spect_umap.voxel_sizes()
        )
    )
    spect_umap_zoomed2spect = spect_umap.zoom_image(
        zooms=zooms, scaling="preserve_values"
    )
    np.savetxt(
        os.path.join(
            args.output_path, args.patient, "SPECT", "zooms_spect_ct2spect.csv"
        ),
        zooms,
        delimiter=",",
    )
    spect_ctac_crop_zoomed2spect = spect_ctac_crop.zoom_image(
        zooms=zooms, scaling="preserve_values"
    )

    def zoom_and_shift_image(image, template_image, reference_image, output_filepath, flip_axes=(0, 2)):
        """
        Process an image by applying a zero couch shift, zooming it to match a provided template,
        """
        # Apply a couch shift with zero shift.
        shift_op = CouchShiftOperator(image, 0)
        shifted_image = shift_op.direct(image)
        
        # Zoom the shifted image to match the template image (preserving values).
        zoomed_image = shifted_image.zoom_image_as_template(template_image, scaling="preserve_values")
        
        # Flip the image along the specified axes.
        flipped_array = np.flip(zoomed_image.as_array(), axis=flip_axes)
        zoomed_image.fill(flipped_array)
        
        # Adjust the offset manually by filling the reference image with the updated array.
        reference_image.fill(zoomed_image.as_array())
        
        # Clone the reference image to obtain the final processed image.
        final_image = reference_image.clone()
        
        # Save the final processed image.
        final_image.write(output_filepath)
        
        return final_image
    
    spect_umap_zoomed2spect = zoom_and_shift_image(
        spect_umap_zoomed2spect,
        spect_template_image, 
        spect_image_from_sinogram,
        os.path.join(args.output_path, args.patient, "SPECT", "umap_zoomed2spect.hv"),
        flip_axes=(0, 2)
    )

    spect_ctac_crop_zoomed2spect = zoom_and_shift_image(
        spect_ctac_crop_zoomed2spect,
        spect_template_image, 
        spect_image_from_sinogram,
        os.path.join(args.output_path, args.patient, "SPECT", "ctac_FoV_corrected_crop_zoomed2spect.hv"),
        flip_axes=(0, 2)
    )

    if args.save_plots:
        plot_image_info(
            spect_umap_zoomed2spect,
            os.path.join(
                args.output_path, args.patient, "SPECT", "umap_zoomed2spect_info.png"
            ),
            "SPECT u-map zoomed to SPECT resolution",
        )
        plot_image_info(
            spect_ctac_crop_zoomed2spect,
            os.path.join(
                args.output_path, args.patient, "SPECT", "ctac_FoV_corrected_crop_zoomed2spect_info.png"
            ),
            "SPECT CTAC cropped to body and zoomed to SPECT resolution",
        )

    # ---------------------- Registration of SPECT to PET ----------------------
    reg_rigid = NiftyAladinSym()
    reg_rigid.set_reference_image(pet_ctac_crop_zoomed2pet)
    reg_rigid.set_floating_image(spect_ctac_crop_zoomed2spect)
    reg_rigid.set_parameter("SetPerformRigid", "1")
    reg_rigid.set_parameter("SetPerformAffine", "0")
    reg_rigid.process()
    transform = reg_rigid.get_transformation_matrix_forward()

    if args.save_plots:
        spect_ctac_zoomed_registered_rigid = reg_rigid.get_output()
        plot_image_info(
            spect_ctac_zoomed_registered_rigid,
            os.path.join(
                args.output_path, args.patient, "SPECT", "ctac_FoV_corrected_crop_zoomed_registered_rigid_info.png"
            ),
            "SPECT u-map registered to PET (rigid)",
        )

    reg = NiftyF3dSym()
    reg.set_reference_image(pet_ctac_crop_zoomed2pet)
    reg.set_floating_image(spect_ctac_crop_zoomed2spect)
    reg.set_initial_affine_transformation(transform)
    reg.print_all_wrapped_methods()
    reg.process()

    # Save the displacement field.
    displacement = reg.get_displacement_field_forward()
    displacement.write(
        os.path.join(args.output_path, args.patient, "SPECT", "spect2pet")
    )

    if args.save_plots:
        plt.figure()
        for i in range(3):
            plt.subplot(3, 1, 1 + i)
            im_slice = displacement.as_array()[displacement.shape[0]//2,:,:,0,i]
            minmax = max(abs(im_slice.min()), abs(im_slice.max()))
            plt.imshow(im_slice, cmap="seismic", vmin=-minmax, vmax=minmax)
            plt.title(f"Displacement field in direction {i}")
        plt.savefig(
            os.path.join(
                args.output_path, args.patient, "SPECT", "displacement_field.png"
            )
        )
    spect_ctac_zoomed_registered = reg.get_output()
    spect_ctac_zoomed_registered.write(
        os.path.join(args.output_path, args.patient, "SPECT", "ctac_FoV_corrected_crop_zoomed_registered.hv")
    )

    if args.save_plots:
        plot_image_info(
            spect_ctac_zoomed_registered,
            os.path.join(
                args.output_path, args.patient, "SPECT", "ctac_FoV_corrected_crop_zoomed_registered_info.png"
            ),
            "SPECT u-map registered to PET",
        )

    # ---------------------- Plot Difference Between PET and Registered SPECT ----------------------
    pet_array = pet_ctac_crop_zoomed2pet.as_array()

    spect_array = spect_ctac_zoomed_registered.as_array()

    diff = pet_array - spect_array
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
            args.output_path, args.patient, "SPECT", "ctac_zoomed_reg_diff.png"
        )
    )
    plt.savefig(
        os.path.join(
            args.output_path, args.patient, "PET", "ctac_zoomed_reg_diff.png"
        )
    )


if __name__ == "__main__":
    main()
