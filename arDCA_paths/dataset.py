from typing import Any
from pathlib import Path
import numpy as np
from typing import Tuple

from torch.utils.data import Dataset
import torch

from adabmDCA.fasta import (
    get_tokens,
    encode_sequence, # import_from_fasta,
    compute_weights,
)

def import_from_fasta(
    fasta_name: str | Path,
    tokens: str | None = None,
    filter_sequences: bool = False,
    remove_duplicates: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Import sequences from a fasta file. The following operations are performed:
    - If 'tokens' is provided, encodes the sequences in numeric format.
    - If 'filter_sequences' is True, removes the sequences whose tokens are not present in the alphabet.
    - If 'remove_duplicates' is True, removes the duplicated sequences.

    Args:
        fasta_name (str | Path): Path to the fasta file.
        tokens (str | None, optional): Alphabet to be used for the encoding. If provided, encodes the sequences in numeric format.
        filter_sequences (bool, optional): If True, removes the sequences whose tokens are not present in the alphabet. Defaults to False.
        remove_duplicates (bool, optional): If True, removes the duplicated sequences. Defaults to True.

    Raises:
        RuntimeError: The file is not in fasta format.

    Returns:
        Tuple[np.ndarray, np.ndarray]: headers, sequences.
    """
    # Import headers and sequences
    sequences = []
    names = []
    seq = ''
    with open(fasta_name, 'r') as f:
        first_line = f.readline()
        if not first_line.startswith('>'):
            raise RuntimeError(f"The file {fasta_name} is not in a fasta format.")
        f.seek(0)
        for line in f:
            if not line.strip():
                continue
            if line.startswith('>'):
                if seq:
                    sequences.append(seq)
                header = line[1:].strip()
                names.append(header)
                seq = ''
            else:
                seq += line.strip()
    if seq:
        sequences.append(seq)
    
    # Filter sequences
    if filter_sequences:
        if tokens is None:
            raise ValueError("Argument 'tokens' must be provided if 'filter_sequences' is True.")
        tokens = get_tokens(tokens)
        tokens_list = [a for a in tokens]
        clean_names = []
        clean_sequences = []
        for n, s in zip(names, sequences):
            good_sequence = np.full(shape=(len(s),), fill_value=False)
            splitline = np.array([a for a in s])
            for token in tokens_list:
                good_sequence += (token == splitline)
            if np.all(good_sequence):
                if n == "":
                    n = "unknown_sequence"
                clean_names.append(n)
                clean_sequences.append(s)
            else:
                print(f"Unknown token found: removing sequence {n}")
        names = np.array(clean_names)
        sequences = np.array(clean_sequences)
        
    else:
        names = np.array(names)
        sequences = np.array(sequences)
    
    # Remove duplicates
    if remove_duplicates:
        sequences, unique_ids = np.unique(sequences, return_index=True)
        names = names[unique_ids]
    
    if (tokens is not None) and (len(sequences) > 0):
        sequences = encode_sequence(sequences, tokens)
    
    return names, sequences

class DatasetDCA(Dataset):
    """Dataset class for handling multi-sequence alignments data."""
    def __init__(
        self,
        path_data: str | Path,
        path_weights: str | Path | None = None,
        alphabet: str = "protein",
        clustering_th: float = 0.8,
        no_reweighting: bool = False,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
        message: bool = True,
    ):
        """Initialize the dataset.

        Args:
            path_data (str | Path): Path to multi sequence alignment in fasta format.
            path_weights (str | Path | None, optional): Path to the file containing the importance weights of the sequences. If None, the weights are computed automatically.
            alphabet (str, optional): Selects the type of encoding of the sequences. Default choices are ("protein", "rna", "dna"). Defaults to "protein".
            clustering_th (float, optional): Sequence identity threshold for clustering. Defaults to 0.8.
            no_reweighting (bool, optional): If True, the weights are not computed. Defaults to False.
            device (torch.device, optional): Device to be used. Defaults to "cpu".
            dtype (torch.dtype, optional): Data type of the dataset. Defaults to torch.float32.
            message (bool, optional): Print the import message. Defaults to True.
        """
        path_data = Path(path_data)
        self.names = None
        self.data = None
        self.device = device
        self.dtype = dtype
        
        # Select the proper encoding
        self.tokens = get_tokens(alphabet)
        print("tokens: ", self.tokens)
        # Automatically detects if the file is in fasta format and imports the data
        with open(path_data, "r") as f:
            first_line = f.readline()
        if first_line.startswith(">"):
            self.names, self.data = import_from_fasta(path_data, tokens=self.tokens, filter_sequences=True, remove_duplicates=False)
            self.data = torch.tensor(self.data, device=device, dtype=torch.int32)
            # Check if data is empty
            if len(self.data) == 0:
                raise ValueError(f"The input dataset is empty. Check that the alphabet is correct. Current alphabet: {alphabet}")
        else:
            raise KeyError("The input dataset is not in fasta format")
        
        # Computes the weights to be assigned to the data
        if no_reweighting:
            self.weights = torch.ones(len(self.data), device=device, dtype=dtype)
        elif path_weights is None:
            if message:
                print("Automatically computing the sequence weights...")
            self.weights = compute_weights(data=self.data, th=clustering_th, device=device, dtype=dtype)
        else:
            with open(path_weights, "r") as f:
                weights = [float(line.strip()) for line in f]
            self.weights = torch.tensor(weights, device=device, dtype=dtype)
        
        if message:
            print(f"Multi-sequence alignment imported: M = {self.data.shape[0]}, L = {self.data.shape[1]}, q = {self.get_num_states()}, M_eff = {int(self.weights.sum())}.")


    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx: int) -> Any:
        sample = self.data[idx]
        weight = self.weights[idx]
        return (sample, weight)
    
    
    def get_num_residues(self) -> int:
        """Returns the number of residues (L) in the multi-sequence alignment.

        Returns:
            int: Length of the MSA.
        """
        return self.data.shape[1]
    
    
    def get_num_states(self) -> int:
        """Returns the number of states (q) in the alphabet.

        Returns:
            int: Number of states.
        """
        return torch.max(self.data).item() + 1
    
    
    def get_effective_size(self) -> int:
        """Returns the effective size (Meff) of the dataset.

        Returns:
            int: Effective size of the dataset.
        """
        return int(self.weights.sum())
    
    
    def shuffle(self) -> None:
        """Shuffles the dataset.
        """
        perm = torch.randperm(len(self.data))
        self.data = self.data[perm]
        self.names = self.names[perm]
        self.weights = self.weights[perm]