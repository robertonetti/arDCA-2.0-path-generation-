import torch
import torch.nn as nn
from tqdm import tqdm
from adabmDCA.stats import get_freq_single_point, get_freq_two_points
from adabmDCA.io import save_params, load_params

def get_entropic_order(fi: torch.Tensor) -> torch.Tensor:
    """Returns the entropic order of the sites in the MSA.

    Args:
        fi (torch.Tensor): Single-site frequencies of the MSA.

    Returns:
        torch.Tensor: Entropic order of the sites.
    """
    site_entropy = -torch.sum(fi * torch.log(fi + 1e-10), dim=1)
   
    return torch.argsort(site_entropy, descending=False)

# Define the loss function
def loss_fn(
    model: nn.Module,
    X: torch.Tensor,
    weights: torch.Tensor,
    fi_target: torch.Tensor,
    fij_target: torch.Tensor,
    reg_h: float = 0.0,
    reg_J: float = 0.0,
) -> torch.Tensor:
    """Computes the negative log-likelihood of the model.
    
    Args:
        model (nn.Module): arDCA model.
        X (torch.Tensor): Input MSA one-hot encoded.
        weights (torch.Tensor): Weights of the sequences in the MSA.
        fi_target (torch.Tensor): Single-site frequencies of the MSA.
        fij_target (torch.Tensor): Pairwise frequencies of the MSA.
        reg_h (float, optional): L2 regularization for the biases. Defaults to 0.0.
        reg_J (float, optional): L2 regularization for the couplings. Defaults to 0.0.
    """
    n_samples, _, q = X.shape
    # normalize the weights
    weights = (weights / weights.sum())
    log_likelihood = 0
    for i in range(1, model.L):
        energy_i = (fi_target[i] @ model.h[i]) + (model.J[i, :, :i, :] * fij_target[i, :, :i, :]).sum()
        logZ_i = torch.logsumexp(model.h[i] + X[:, :i, :].view(n_samples, -1) @ model.J[i, :, :i, :].view(q, -1).mT, dim=-1) @ weights
        log_likelihood += energy_i - logZ_i
    
    return - log_likelihood + reg_h * torch.norm(model.h)**2 + reg_J * torch.norm(model.J)**2


class arDCA(nn.Module):
    def __init__(
        self,
        L: int,
        q: int,
    ):
        """Initializes the arDCA model. Either fi or L and q must be provided.

        Args:
            L (int): Number of residues in the MSA.
            q (int): Number of states for the categorical variables.
        """
        super(arDCA, self).__init__()
        self.L = L
        self.q = q
        self.h = nn.Parameter(torch.randn(L, q) * 1e-4)
        self.J = nn.Parameter(torch.randn(self.L, self.q, self.L, self.q) * 1e-4)
        # Mask for removing self-interactions
        self.mask = nn.Parameter(torch.ones(self.L, self.q, self.L, self.q), requires_grad=False)
        for i in range(self.L):
            self.mask[i, :, i, :] = 0
        self.remove_autocorr()
        self.revert_entropic_order = None
        
    def remove_autocorr(self):
        """Removes the self-interactions from the model."""
        self.J.data = self.J.data * self.mask.data

    def forward(
        self,
        X: torch.Tensor,
        beta: float = 1.0,
    ) -> torch.Tensor:
        """Predicts the probability of next token given the previous ones.
        
        Args:
            X (torch.Tensor): Input MSA one-hot encoded.
            beta (float, optional): Inverse temperature. Defaults to 1.0.
            
        Returns:
            torch.Tensor: Probability of the next token.
        """
        # X has to be a 3D tensor of shape (n_samples, l, q) with l < L
        if X.dim() != 3:
            raise ValueError("X must be a 3D tensor")
        if X.shape[1] >= self.L:
            raise ValueError("X must have a second dimension smaller than L")
        n_samples, residue_idx = X.shape[0], X.shape[1]

        J_ar = self.J[residue_idx, :, :residue_idx, :].view(self.q, -1)
        X_ar = X.view(n_samples, -1)
        logit_i = self.h[residue_idx] + torch.einsum("ij,nj->ni", J_ar, X_ar)
        prob_i = torch.softmax(beta * logit_i, dim=-1)
        
        return prob_i
    
    
    def sample(
        self,
        n_samples: int,
        beta: float = 1.0,
    ) -> torch.Tensor:
        """Samples from the model.
        
        Args:
            n_samples (int): Number of samples to generate.
            beta (float, optional): Inverse temperature. Defaults to 1.0.
            
        Returns:
            torch.Tensor: Generated samples.
        """       
        X = torch.zeros(n_samples, self.L, self.q, dtype=self.h.dtype, device=self.h.device)
        X_init = torch.multinomial(torch.softmax(self.h[0], dim=-1), num_samples=n_samples, replacement=True) # (n_samples,)
        X[:, 0, :] = nn.functional.one_hot(X_init, self.q).to(dtype=self.h.dtype)
        
        for i in range(1, self.L):
            prob_i = self.forward(X[:, :i, :], beta=beta)
            sample_i = torch.multinomial(prob_i, num_samples=1).squeeze()
            X[:, i] = nn.functional.one_hot(sample_i, self.q).to(dtype=self.h.dtype)
        if self.revert_entropic_order is not None:
            X = X[:, self.revert_entropic_order, :]
            
        return X
    
    
    def fit(
        self,
        X: torch.Tensor,
        weights: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        max_epochs: int = 10000,
        epsconv: float = 1e-4,
        pseudo_count: float = 0.0,
        use_entropic_order: bool = True,
        fix_first_residue: bool = False,
        reg_h: float = 0.0,
        reg_J: float = 0.0,
    ) -> None:
        """Fits the model to the data.
        
        Args:
            X (torch.Tensor): Input MSA one-hot encoded.
            weights (torch.Tensor): Weights of the sequences in the MSA.
            optimizer (torch.optim.Optimizer): Optimizer to use.
            max_epochs (int, optional): Maximum number of epochs. Defaults to 1000.
            epsconv (float, optional): Convergence threshold. Defaults to 1e-4.
            target_pearson (float, optional): Target Pearson correlation. Defaults to 0.95.
            pseudo_count (float, optional): Pseudo-count for the frequencies. Defaults to 0.0.
            n_samples (int, optional): Number of samples to generate for computing the pearson. Defaults to 10000.
            use_entropic_order (bool, optional): Whether to use the entropic order. Defaults to True.
            fix_first_residue (bool, optional): Fix the position of the first residue so that it is not sorted by entropy.
                Used when the first residue encodes for the label. Defaults to False.
        """
        fi = get_freq_single_point(X, weights=weights, pseudo_count=pseudo_count)
        if use_entropic_order:
            if fix_first_residue:
                entropic_order = get_entropic_order(fi[1:])
                entropic_order = torch.cat([torch.tensor([0], device=entropic_order.device), entropic_order + 1])
            else:
                entropic_order = get_entropic_order(fi)
            self.revert_entropic_order = torch.argsort(entropic_order)
            X = X[:, entropic_order, :]
        else:
            self.revert_entropic_order = None
        # Target frequencies, if entropic order is used, the frequencies are sorted
        fi_target = get_freq_single_point(X, weights=weights, pseudo_count=pseudo_count)
        fij_target = get_freq_two_points(X, weights=weights, pseudo_count=pseudo_count)
        self.h.data = torch.log(fi_target + 1e-10)
        
        # Training loop
        pbar = tqdm(
            total=max_epochs,
            colour="red",
            dynamic_ncols=True,
            leave=False,
            ascii="-#",
        )
        pbar.set_description(f"Loss: inf")
        prev_loss = float("inf")
        for _ in range(max_epochs):
            optimizer.zero_grad()
            loss = loss_fn(self, X, weights, fi_target, fij_target, reg_h=reg_h, reg_J=reg_J)
            loss.backward()
            optimizer.step()
            self.remove_autocorr()
            pbar.update(1)
            pbar.set_description(f"Loss: {loss.item():.3f}")
            if torch.abs(loss - prev_loss) < epsconv:
                break
            prev_loss = loss
        pbar.close()
        
    def save(
        self,
        path: str,
        tokens: str,
    ) -> None:
        """Saves the parameters of the model.
        
        Args:
            path (str): Path to the file where to save the parameters.
        """
        params = {
            "bias" : self.h.detach(),
            "coupling_matrix" : self.J.detach(),
        }
        mask_save = torch.ones((self.L, self.q, self.L, self.q), dtype=torch.bool)
        idx1_rm, idx2_rm = torch.triu_indices(self.L, self.L, offset=0)
        mask_save[idx1_rm, :, idx2_rm, :] = 0
        save_params(
            fname=path,
            params=params,
            mask=mask_save,
            tokens=tokens,
        )
        
    def load(
        self,
        path: str,
        tokens: str,
    ) -> None:
        params = load_params(
            fname=path,
            tokens=tokens,
            device=self.h.device,
            dtype=self.h.dtype,
        )
        self.h.data = params["bias"]
        self.J.data = params["coupling_matrix"]
        # Set to zero the upper triangular part of the couplings
        mask = torch.ones((self.L, self.q, self.L, self.q), dtype=torch.bool, device=self.h.device)
        idx1_rm, idx2_rm = torch.triu_indices(self.L, self.L, offset=0)
        mask[idx1_rm, :, idx2_rm, :] = 0
        self.J.data = self.J.data * mask

    