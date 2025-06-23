from pathlib import Path
import argparse
import numpy as np
import sys
import torch
import torch.nn as nn
import torchmetrics
import matplotlib.pyplot as plt

# from adabmDCA.dataset import DatasetDCA
from adabmDCA.fasta import get_tokens
from adabmDCA.stats import get_freq_single_point, get_freq_two_points, get_correlation_two_points
from adabmDCA.utils import get_device, get_dtype
from adabmDCA.functional import one_hot

from arDCA import arDCA
from arDCA.parser import add_args_train
from arDCA.dataset import DatasetDCA


from typing import Optional
from typing import Tuple


# import command-line input arguments
def create_parser():
    parser = argparse.ArgumentParser(description='Train a DCA model.')
    parser = add_args_train(parser)    
    return parser


def main():
    # Load parser, training dataset and DCA model
    parser = create_parser()
    args = parser.parse_args()
    
    print("\n" + "".join(["*"] * 10) + f" Training arDCA model " + "".join(["*"] * 10) + "\n")
    # Set the device
    device = get_device(args.device)
    dtype = get_dtype(args.dtype)

    template = "{0:<30} {1:<50}"
    print(template.format("Input MSA:", str(args.data)))
    print(template.format("Output folder:", str(args.output)))
    print(template.format("Alphabet:", args.alphabet))
    print(template.format("Learning rate:", args.lr))
    print(template.format("L2 reg. for fields:", args.reg_h))
    print(template.format("L2 reg. for couplings:", args.reg_J))
    print(template.format("Entropic order:", "True" if not args.no_entropic_order else "False"))
    print(template.format("Convergence threshold:", args.epsconv))
    if args.pseudocount is not None:
        print(template.format("Pseudocount:", args.pseudocount))
    print(template.format("Random seed:", args.seed))
    print(template.format("Data type:", args.dtype))
    if args.path_graph is not None:
        print(template.format("Input graph:", str(args.path_graph)))
    if args.data_test is not None:
        print(template.format("Input test MSA:", str(args.data_test)))
    if args.batch_size is not None:
        print(template.format("batch size:", str(args.batch_size))) 
    print("\n")

    # Check if the data file exist
    if not Path(args.data).exists():
        raise FileNotFoundError(f"Data file {args.data} not found.")
    
    # Create the folder where to save the model
    folder = Path(args.output)
    folder.mkdir(parents=True, exist_ok=True)
    
    if args.label is not None:
        file_paths = {"params" : folder / Path(f"{args.label}_params.pth"),} 
    else:
        file_paths = { "params" : folder / Path(f"params.pth"),}
    
    # Import dataset
    print("Importing dataset...")
    sys.stdout.flush()
    dataset = DatasetDCA(
        path_data      = args.data,
        path_weights   = args.weights,
        alphabet       = args.alphabet,
        clustering_th  = args.clustering_seqid,
        no_reweighting = args.no_reweighting,
        device         = device,
        dtype          = dtype
        )

    # Compute statistics of the data
    L = dataset.get_num_residues()
    q = dataset.get_num_states()
    print("L: ", L, "q: ", q)

    # Import graph from file
    if args.path_graph:
        if not args.no_entropic_order:
            raise ValueError("Graph and entropic order cannot be done together")
        print("Importing graph...")
    graph = torch.load(args.path_graph) if args.path_graph else None
    print("\n")

    model = arDCA(L=L, q=q, graph=graph, model=args.mode).to(device=device) # model = arDCA(L=L, q=q).to(device=device, dtype=dtype)
    # tokens = get_tokens(args.alphabet)
    
    if args.mode == "third":
        print("Kind of task: predict ", args.mode, " sequence.")
        index = 2 * L // 3
        test_fn = model.test_prediction_third
    elif args.mode == "second":
        print("Kind of task: predict ", args.mode, " sequence.")
        index = L // 2
        test_fn = model.test_prediction_second
    else:
        print("Kind of task: normal arDCA.")
        index = L
        test_fn = None

    print("\n")

    # Save the weights if not already provided
    if args.weights is None:
        if args.label is not None:
            path_weights = folder / f"{args.label}_weights.dat"
        else:
            path_weights = folder / "weights.dat"
        np.savetxt(path_weights, dataset.weights.cpu().numpy())
        print(f"Weights saved in {path_weights}")
        
    # Set the random seed
    torch.manual_seed(args.seed)
    
    if args.pseudocount is None:
        args.pseudocount = 1. / dataset.get_effective_size()
        print(f"Pseudocount automatically set to {args.pseudocount}.")

    
    if args.batch_size is None: 
        data_oh = one_hot(dataset.data, num_classes=q).to(dtype)
        fi_target = get_freq_single_point(data=data_oh, weights=dataset.weights, pseudo_count=args.pseudocount)
        fij_target = get_freq_two_points(data=data_oh, weights=dataset.weights, pseudo_count=args.pseudocount)
        
    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    print("\n")
    data_test_oh = None
    if args.data_test is not None:
        dataset_test = DatasetDCA(
            path_data      = args.data_test,
            path_weights   = None,
            alphabet       = args.alphabet,
            clustering_th  = None,
            no_reweighting = args.no_reweighting,
            device         = device,
            dtype          = dtype,
        )
        if args.batch_size is None: 
            data_test_oh = one_hot(dataset_test.data, num_classes=q).to(dtype)

    if args.batch_size is not None:
        loss, losses, log_likelihoods, val_losses, val_log_likelihoods = model.fit_batch(
            X                  = dataset.data,
            weights            = dataset.weights,
            optimizer          = optimizer,
            max_epochs         = args.nepochs,
            epsconv            = args.epsconv,
            pseudo_count       = args.pseudocount,
            use_entropic_order = not args.no_entropic_order,
            fix_first_residue  = False,
            reg_h              = args.reg_h,
            reg_J              = args.reg_J,
            X_test             = dataset_test.data,
            batch_size         = args.batch_size,
            index              = index
        )
    else:
        loss, ro_fi_prediction, ro_fi_input, ro_cij_prediction, ro_cij_input, ro_cij_prediction_test, losses, val_losses, log_likelihoods, val_log_likelihoods = model.fit(
                    X                  =data_oh,
                    weights            =dataset.weights,
                    optimizer          =optimizer,
                    max_epochs         =args.nepochs,
                    epsconv            =args.epsconv,
                    pseudo_count       =args.pseudocount,
                    use_entropic_order =not args.no_entropic_order,
                    fix_first_residue  = False,
                    reg_h              =args.reg_h,
                    reg_J              =args.reg_J,
                    X_test             =data_test_oh,
                    fi_target          =fi_target,
                    fij_target         =fij_target,
                    index              =index,
        )


    # Save the model
    print("Saving the model...")
    torch.save(model.state_dict(), file_paths["params"])
    print(f"Model saved in {file_paths['params']}")

    if args.batch_size is None:
        # First plot: Pearson correlation
        plt.figure()                           # ← start a new figure
        plt.plot(ro_cij_prediction,      label="cij Train")
        plt.plot(ro_cij_prediction_test, label="cij Test")
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Pearson correlation")
        plt.title("Pearson correlation")
        plt.savefig(folder / "pearson.png")
        plt.close()                           # ← close it when you’re done

    # Second plot: Loss curves
    x_train = np.arange(len(losses))
    x_val = np.linspace(0, len(losses) - 1, len(val_losses))
    plt.figure()
    plt.plot(x_train,     losses,     label="Loss Train")
    plt.plot(x_val,       val_losses, label="Loss Validation")
    plt.plot(x_train,     log_likelihoods,     label="LogLike Train")
    plt.plot(x_val,       val_log_likelihoods, label="LogLike Validation")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss")
    plt.grid(True)
    plt.savefig(folder / "loss.png")
    plt.close()
            
        
    if args.batch_size is None:
        # Compute Accuracy
        if args.no_entropic_order is True: 
            data_oh_ordered = data_oh.clone()
            data_test_oh_ordered = data_test_oh.clone()
        else: 
            data_oh_ordered = data_oh[:, model.entropic_order.cpu().numpy(), :].clone()  
            data_test_oh_ordered = data_test_oh[:, model.entropic_order.cpu().numpy(), :].clone()

        err = model.test_fn(data_oh_ordered)
        print(f"TRAIN SET: Mean agreement between the predicted and the true sequence: {err}\n")
        if args.data_test is not None:
            err_test = model.test_fn(data_test_oh_ordered) 
            print(f"TEST SET: Mean agreement between the predicted and the true sequence: {err_test}\n")

        # Sample from the model and compute Cij pearson score
        if args.mode != "second" and args.mode != "third":
            samples = model.sample_autoregressive(data_test_oh_ordered[:, :1, :])
        else:
            samples = model.sample_autoregressive(data_test_oh_ordered[:, :index, :])

        pi = get_freq_single_point(samples[:, index:, :])
        pij = get_freq_two_points(samples[:, index:, :])

        test_weights = torch.ones(data_test_oh_ordered.shape[0]).to(data_test_oh_ordered.device)
        fi_target_pred  = get_freq_single_point(data=data_test_oh_ordered[:, index:, :], weights=test_weights, pseudo_count=args.pseudocount)
        fij_target_pred = get_freq_two_points(data=data_test_oh_ordered[:, index:, :],   weights=test_weights, pseudo_count=args.pseudocount)
    
        print(f"Loss: {loss:.3f}\n")
        
        pearson_third, _ = get_correlation_two_points(fi=fi_target_pred, fij=fij_target_pred, pi=pi, pij=pij)
        pearson_corr_fi = torchmetrics.functional.pearson_corrcoef(fi_target_pred.view(-1), pi.view(-1))
        print(f"Pearson correlation of the two-site statistics on the predicted sequence TEST: {pearson_third:.3f}\n")
        print(f"Pearson correlation of the single-site statistics on the predicted sequence TEST: {pearson_corr_fi:.3f}\n")
        
    


    
    
if __name__ == "__main__":
    main()