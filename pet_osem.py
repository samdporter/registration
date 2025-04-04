from sirf.STIR import *
import os
import argparse

parser = argparse.ArgumentParser(description='Reconstruct with OSEM')
parser.add_argument('--data_path', type=str, default="/home/storage/triple_modality_antropomorphic_data/data/pet/cylindrical/001283/output_images", help='data path')
parser.add_argument('--output_path', type=str, default="/home/storage/triple_modality_antropomorphic_data/data/pet/cylindrical/001283/output_images", help='output path')
parser.add_argument('--suffix', type=str, default="none", help='suffix for the data files, f1b1, f2b1, both or none')
parser.add_argument('--rdp', action='store_true', help='use relative difference prior')

AcquisitionData.set_storage_scheme('memory')  # Store prompts in memory

def get_pet_data(path, suffix):
    pet_data = {}
    pet_data["acquisition_data"] = AcquisitionData(os.path.join(path, f"prompts{suffix}.hs"))
    pet_data["additive"] = AcquisitionData(os.path.join(path, f"additive_term{suffix}.hs"))
    pet_data["normalisation"] = AcquisitionData(os.path.join(path, f"mult_factors{suffix}.hs"))
    return pet_data

def get_pet_am(pet_data, gpu=True):
    if gpu:
        pet_am = AcquisitionModelUsingParallelproj()
    else:
        pet_am = AcquisitionModelUsingRayTracingMatrix()
        pet_am.set_num_tangential_LORs(10)
    asm = AcquisitionSensitivityModel(pet_data["normalisation"])
    pet_am.set_acquisition_sensitivity(asm)
    pet_am.set_additive_term(pet_data["additive"])
    return pet_am

def get_reconstructor(data, acq_model, initial_image, num_subsets, num_epochs):
    recon = OSMAPOSLReconstructor()
    obj_fun = make_Poisson_loglikelihood(acq_data=data, acq_model=acq_model)
    if args.rdp:
        prior = CudaRelativeDifferencePrior()
        prior = CudaRelativeDifferencePrior()
        prior.set_penalisation_factor(5)
        prior.set_up(initial_image)
        obj_fun.set_prior(prior)
    recon.set_objective_function(obj_fun)
    recon.set_num_subsets(num_subsets)
    recon.set_num_subiterations(num_subsets * num_epochs)
    recon.set_up(initial_image)
    return recon

def reconstruct_for_suffix(args, suffix):
    pet_data = get_pet_data(args.data_path, suffix)
    initial_image = pet_data["acquisition_data"].create_uniform_image(1, xy=288)
    pet_am = get_pet_am(pet_data, gpu=True)
    reconstructor = get_reconstructor(pet_data["acquisition_data"], pet_am, initial_image, 9, 5)
    reconstructor.reconstruct(initial_image)
    if args.rdp:
        suffix = suffix + "_rdp"
    output_path = os.path.join(args.output_path, f"pet_recon{suffix}.hv")
    output = reconstructor.get_output()
    gauss = SeparableGaussianImageFilter()
    gauss.set_fwhms((5, 5, 5))
    gauss.apply(output)
    output.write(output_path)
    print(f"Reconstruction complete for {suffix}. Output saved to {output_path}.")

def main(args):
    if args.suffix == "both":
        for suffix in ["f1b1", "f2b1"]:
            reconstruct_for_suffix(args, "_"+suffix)
    elif args.suffix == "none":
        reconstruct_for_suffix(args, "")
    else:
        reconstruct_for_suffix(args, "_"+args.suffix)

if __name__ == "__main__":
    args = parser.parse_args()
    # if output path does not exist, create it
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    main(args)