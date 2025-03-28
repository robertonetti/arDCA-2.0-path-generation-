import argparse
from pathlib import Path

def add_args_dca(parser : argparse.ArgumentParser) -> argparse.ArgumentParser:
    dca_args = parser.add_argument_group("General DCA arguments")
    dca_args.add_argument("-d", "--data",         type=Path,  required=True,        help="Filename of the dataset to be used for training the model.")
    dca_args.add_argument("-o", "--output",       type=Path,  default='DCA_model',  help="(Defaults to DCA_model). Path to the folder where to save the model.")
    # Optional arguments
    dca_args.add_argument("-l", "--label",        type=str,   default=None,         help="(Defaults to None). If provoded, adds a label to the output files inside the output folder.")
    dca_args.add_argument("--alphabet",           type=str,   default="protein",    help="(Defaults to protein). Type of encoding for the sequences. Choose among ['protein', 'rna', 'dna'] or a user-defined string of tokens.")
    # dca_args.add_argument("--no_entropic_order",  action="store_true",              help="If provided, the entropic order is not used for the sequences.")
    dca_args.add_argument("--no_entropic_order",  type=bool,  default=True,         help="If provided, the entropic order is not used for the sequences.")
    dca_args.add_argument("--data_test",          type=Path,  default=None,        help="Filename of the dataset to be used for training the model.")
    dca_args.add_argument("--lr",                 type=float, default=0.005,        help="(Defaults to 0.005). Learning rate.")
    dca_args.add_argument("--reg_h",              type=float, default=1e-6,         help="(Defaults to 1e-6). L2 regularization parameter for the fields.")
    dca_args.add_argument("--reg_J",              type=float, default=1e-4,         help="(Defaults to 1e-4). L2 regularization parameter for the couplings.")
    dca_args.add_argument("--epsconv",            type=float, default=1e-2,         help="(Defaults to 1e-2). Convergence threshold on the loss function.")
    dca_args.add_argument("--nepochs",            type=int,   default=1000,         help="(Defaults to 1000). Maximum number of epochs allowed.")
    dca_args.add_argument("--pseudocount",        type=float, default=None,         help="(Defaults to None). Pseudo count for the single and two-sites statistics. Acts as a regularization. If None, it is set to 1/Meff.")
    dca_args.add_argument("--seed",               type=int,   default=0,            help="(Defaults to 0). Seed for the random number generator.")
    dca_args.add_argument("--device",             type=str,   default="cuda",       help="(Defaults to cuda). Device to be used.")
    dca_args.add_argument("--dtype",              type=str,   default="float32",    help="(Defaults to float32). Data type to be used.")
    dca_args.add_argument("--path_graph",         type=str,   default=None,         help="(Defaults to None). Path to the model containing the graph used to train arDCA.")

    return parser


def add_args_reweighting(parser : argparse.ArgumentParser) -> argparse.ArgumentParser:
    weight_args = parser.add_argument_group("Sequence reweighting arguments")
    weight_args.add_argument("-w", "--weights",      type=Path,  default=None,         help="(Defaults to None). Path to the file containing the weights of the sequences. If None, the weights are computed automatically.")
    weight_args.add_argument("--clustering_seqid",   type=float, default=0.8,          help="(Defaults to 0.8). Sequence Identity threshold for clustering. Used only if 'weights' is not provided.")
    weight_args.add_argument("--no_reweighting",     action="store_true",              help="If provided, the reweighting of the sequences is not performed.")

    return parser


def add_args_train(parser : argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser = add_args_dca(parser)
    parser = add_args_reweighting(parser)
    
    return parser


def add_args_sample(parser : argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("-p", "--path_params",  type=Path,   required=True,      help="Path to the file containing the parameters of DCA model to sample from.")
    parser.add_argument("-o", "--output",       type=Path,   required=True,      help="Path to the folder where to save the output.")
    parser.add_argument("--ngen",               type=int,    required=True,      help="Number of sequences to be generated.") 
    
    # Optional arguments
    parser.add_argument("-d", "--data",         type=Path,   default=None,       help="(Defaults to None). Path to the file containing the data. If provided, the Pearson correlation coefficient is computed.")
    parser.add_argument("-l", "--label",        type=str,    default="sampling", help="(Defaults to 'sampling'). Label to be used for the output files.")
    parser.add_argument("--alphabet",           type=str,    default="protein",  help="(Defaults to protein). Type of encoding for the sequences. Choose among ['protein', 'rna', 'dna'] or a user-defined string of tokens.")
    parser.add_argument("--beta",               type=float,  default=1.0,        help="(Defaults to 1.0). Inverse temperature for the sampling.")
    parser.add_argument("--pseudocount",        type=float,  default=None,       help="(Defaults to None). Pseudocount for the single and two points statistics used during the training. If None, 1/Meff is used.")
    parser.add_argument("--device",             type=str,    default="cuda",     help="(Defaults to cuda). Device to perform computations on.")
    parser.add_argument("--dtype",              type=str,    default="float32",  help="(Defaults to float32). Data type to be used.")
    
    parser = add_args_reweighting(parser)
    
    return parser