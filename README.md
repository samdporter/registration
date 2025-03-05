# Registrations for Y90 synergistic reconstruction

A repository containing scripts to register phantom and patient PET/SPECT data with separate CT images. 

Requires DICOM CT images and PET/SPECT sinograms in STIR format

## Script Summary

- **SPECT Uniform Template:**
  - **Function:** `create_spect_uniform_image`
  - **Operation:** Creates a uniform image by doubling the z-voxel size and recalculating dimensions:
    \[
    \text{dims}[0] = \left\lfloor \frac{\text{dims}[0]}{2} \right\rfloor + (\text{dims}[0] \bmod 2), \quad \text{dims}[2] = \text{dims}[1]
    \]

- **PET Data Processing:**
  - Loads a PET sinogram to create a uniform PET image.
  - Processes PET CTAC by thresholding (max with -1000) and converting to a u-map using an external command (with GE parameters).
  - Adjusts the PET u-map’s z-offset:
    \[
    \text{offset}_z = \frac{V_z^{\text{u-map}} \cdot N_z^{\text{u-map}} - V_z^{\text{template}} \cdot N_z^{\text{template}}}{2}
    \]
  - Zooms the PET u-map to match the PET template; optionally generates diagnostic plots.

- **SPECT Data Processing:**
  - Loads and thresholds the SPECT CTAC image, then converts it to a u-map (using Mediso parameters).
  - Creates a uniform SPECT template from a sinogram.
  - Adjusts and zooms the SPECT u-map to remove offsets and match the SPECT template; optionally plots diagnostic info.

### Phantoms - NPL Anyscan

Uses non-rigid registraion
- **Registration:**
  - If PET data is available, performs rigid registration (via NiftyAladinSym) to align the SPECT u-map to the PET u-map.
  - Saves the transformation matrix and registered SPECT u-map.



### Patients - Oxford GE Discovery (670 & 710)

Uses rigid and non-rigid registration
- **Registration:**
  - **Rigid Registration:**  
    Use NiftyAladinSym to compute an initial affine transformation \( T \) aligning SPECT to PET.
  - **Non-Rigid Registration:**  
    Use NiftyF3dSym, starting from \( T \), to compute a displacement field \( D \) for finer alignment.
