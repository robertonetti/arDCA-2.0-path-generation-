import argparse
from pathlib import Path
import torch

from adabmDCA.fasta import (
    get_tokens,
    write_fasta,
)
from arDCA import arDCA
from arDCA.parser import add_args_sample
from adabmDCA.dataset import DatasetDCA
from adabmDCA.utils import resample_sequences, get_device, get_dtype
from adabmDCA.functional import one_hot
from adabmDCA.stats import get_freq_single_point, get_freq_two_points, get_correlation_two_points


# import command-line input arguments
def create_parser():
    # Important arguments
    parser = argparse.ArgumentParser(description='Samples from a DCA model.')
    parser = add_args_sample(parser)
    
    return parser


def main():       
    
    # Parse arguments
    parser = create_parser()
    args = parser.parse_args()
    
    # Create output folder
    filename = Path(args.output)
    folder = filename.parent / Path(filename.name)
    # Create the folder where to save the samples
    folder.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "".join(["*"] * 10) + f" Sampling from arDCA model " + "".join(["*"] * 10) + "\n")
    # Set the device
    device = get_device(args.device)
    dtype = get_dtype(args.dtype)
    tokens = get_tokens(args.alphabet)
    
    # Check that the data file exists
    if args.data:
        if not Path(args.data).exists():
            raise FileNotFoundError(f"Data file {args.data} not found.")
    
    # Check that the parameters file exists
    if not Path(args.path_params).exists():
        raise FileNotFoundError(f"Parameters file {args.path_params} not found.")    
        
    # Import parameters
    print(f"Loading parameters from {args.path_params}...")
    params = torch.load(args.path_params)
    L, q = params["h"].shape
    model = arDCA(L=L, q=q).to(device=device, dtype=dtype)
    model.load_state_dict(params)
    
    # Sample from the model
    print(f"Sampling {args.ngen} sequences...")
    samples = model.sample(args.ngen, beta=args.beta)
    
    # If data is provided, compute the Pearson correlation coefficient
    if args.data:
        print(f"Loading data from {args.data}...")
        dataset = DatasetDCA(
            path_data=args.data,
            path_weights=args.weights,
            alphabet=tokens,
            clustering_th=args.clustering_seqid,
            no_reweighting=args.no_reweighting,
            device=device,
            dtype=dtype,
        )
        data = one_hot(dataset.data, num_classes=len(tokens)).to(dtype)
        weights = dataset.weights
        data_resampled = resample_sequences(data, weights, args.ngen)
    
        if args.pseudocount is None:
            args.pseudocount = 1. / weights.sum()
        print(f"Using pseudocount: {args.pseudocount}...")
    
        # Compute single and two-site frequencies of the data
        fi = get_freq_single_point(data=data_resampled, pseudo_count=args.pseudocount)
        fij = get_freq_two_points(data=data_resampled, pseudo_count=args.pseudocount)
        pi = get_freq_single_point(data=samples)
        pij = get_freq_two_points(data=samples)
        pearson, slope = get_correlation_two_points(fi=fi, pi=pi, fij=fij, pij=pij)
        print(f"Pearson correlation coefficient: {pearson:.3f}")
        print(f"Slope: {slope:.3f}")
    
    print("Saving the samples...")
    headers = [f"sequence {i+1}" for i in range(args.ngen)]
    write_fasta(
        fname=folder / Path(f"{args.label}_samples.fasta"),
        headers=headers,
        sequences=samples.argmax(-1).cpu().numpy(),
        numeric_input=True,
        remove_gaps=False,
        tokens=tokens,
    )
    
    print(f"Done, results saved in {str(folder)}")
    
    
if __name__ == "__main__":
    main()