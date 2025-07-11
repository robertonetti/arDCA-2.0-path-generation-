import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from tqdm import tqdm
# from adabmDCA.stats import get_freq_single_point, get_freq_two_points
from adabmDCA.io import save_params, load_params
from adabmDCA.functional import one_hot
from adabmDCA.stats import get_correlation_two_points, get_freq_single_point, get_freq_two_points

import torch
import torch.nn.functional as F
import torchmetrics

def get_freq_single_point_batches(X, weights, pseudo_count, batch_size, num_classes, device):

    M_tot = X.shape[0]
    fi = torch.zeros((X.shape[1], num_classes), dtype=torch.float32, device=device)
    for start in range(0, M_tot, batch_size):
        end = min(start + batch_size, M_tot)
        batch = X[start:end]      # shape: (batch_actual_size, L)
        batch_weights = weights[start:end] 
        batch_oh = one_hot(batch, num_classes=num_classes).to(dtype=torch.float32, device=device)  # shape: (batch_actual_size, L, q)
        fi += get_freq_single_point(batch_oh, weights=batch_weights, pseudo_count=pseudo_count)
    return fi / (M_tot // batch_size + 1)  # Average over batches


def get_entropic_order(fi: torch.Tensor) -> torch.Tensor:
    """Returns the entropic order of the sites in the MSA.

    Args:
        fi (torch.Tensor): Single-site frequencies of the MSA.

    Returns:
        torch.Tensor: Entropic order of the sites.
    """
    site_entropy = -torch.sum(fi * torch.log(fi + 1e-10), dim=1)
   
    return torch.argsort(site_entropy, descending=False)



def get_entropic_order_with_inverse(fi: torch.Tensor, index: int) -> (torch.Tensor, torch.Tensor):
    """
    Returns two vectors:
      1) A full index ordering of the MSA sites such that:
         - Sites 0 to index-1 remain in their original order
         - Sites from index onward are sorted ascending by their site entropy
      2) The inverse permutation that restores the original site order from the sorted order.

    Args:
        fi (torch.Tensor): Single-site frequencies of the MSA. Shape: (n_sites, n_states)
        index (int): Starting site index to reorder by entropy.

    Returns:
        order (torch.Tensor): A 1D tensor of length n_sites containing a permutation of [0, ..., n_sites-1].
        inverse_order (torch.Tensor): A 1D tensor such that order[inverse_order] = torch.arange(n_sites).
    """
    # Compute the entropy for each site: H_i = -sum_j fi_ij * log(fi_ij)
    site_entropy = -torch.sum(fi * torch.log(fi + 1e-10), dim=1)
    n_sites = site_entropy.size(0)
    # Keep original order for sites before 'index'
    prefix_indices = torch.arange(min(index, n_sites), device=fi.device)
    # Sort the remaining sites by entropy
    if index < n_sites:
        suffix_entropy = site_entropy[index:]
        sorted_suffix = torch.argsort(suffix_entropy, descending=False) + index
    else:
        sorted_suffix = torch.tensor([], dtype=torch.long, device=fi.device)
    # Concatenate prefix (unchanged) with sorted suffix
    order = torch.cat([prefix_indices, sorted_suffix], dim=0)
    # Build inverse permutation: inverse_order[order[i]] = i
    inverse_order = torch.empty(n_sites, dtype=torch.long, device=fi.device)
    inverse_order[order] = torch.arange(n_sites, device=fi.device)

    return order, inverse_order


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
    
    return - log_likelihood + reg_h * torch.norm(model.h)**2 + reg_J * torch.norm(model.J)**2, - log_likelihood

# Define the loss function
def loss_third_fn(
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
    for i in range(2*model.L//3, model.L):
        energy_i = (fi_target[i] @ model.h[i]) + (model.J[i, :, :i, :] * fij_target[i, :, :i, :]).sum()
        logZ_i = torch.logsumexp(model.h[i] + X[:, :i, :].view(n_samples, -1) @ model.J[i, :, :i, :].view(q, -1).mT, dim=-1) @ weights
        log_likelihood += energy_i - logZ_i
    
    return - log_likelihood + reg_h * torch.norm(model.h)**2 + reg_J * torch.norm(model.J)**2, - log_likelihood

# Define the loss function
def loss_second_fn(
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
    for i in range(model.L//2, model.L):
        energy_i = (fi_target[i] @ model.h[i]) + (model.J[i, :, :i, :] * fij_target[i, :, :i, :]).sum()
        logZ_i = torch.logsumexp(model.h[i] + X[:, :i, :].view(n_samples, -1) @ model.J[i, :, :i, :].view(q, -1).mT, dim=-1) @ weights
        log_likelihood += energy_i - logZ_i
    
    return - log_likelihood + reg_h * torch.norm(model.h)**2 + reg_J * torch.norm(model.J)**2, - log_likelihood


class EarlyStopping:
    def __init__(self, patience=5, epsconv=0.01):
        """
        patience: How many epochs to wait before stopping if no improvement.
        min_delta: Minimum change in the monitored value to qualify as an improvement.
        """
        self.patience = patience
        self.epsconv = epsconv
        self.best_loss = float("inf")
        self.counter = 0
    
    def __call__(self, loss):
        if loss < self.best_loss - self.epsconv:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1
        if self.counter >= self.patience:
            return True
        else:
            return False

class arDCA(nn.Module):
    def __init__(
        self,
        L: int,
        q: int,
        graph: dict = None,
        model: str = "third",
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
        # Mask to initialize the correct graph
        if graph is None:
            graph = {'J': torch.ones(self.L, self.q, self.L, self.q, dtype=torch.bool), 'h': torch.ones(self.L, self.q, dtype=torch.bool)}
        self.graph_J = nn.Parameter(graph['J'], requires_grad=False)
        self.graph_h = nn.Parameter(graph['h'], requires_grad=False)
        self.restore_graph()
        # Sorting of the sites to the MSA ordering
        self.sorting = nn.Parameter(torch.arange(self.L), requires_grad=False)
        self.entropic_order = nn.Parameter(torch.empty(self.L), requires_grad=False)
        self.inverse_entropic_order = nn.Parameter(torch.empty(self.L), requires_grad=False)
        

        if model == "third":
            self.loss_fn = loss_third_fn
            self.test_fn = self.test_prediction_third
        elif model == "second":
            self.loss_fn = loss_second_fn
            self.test_fn = self.test_prediction_second
        else: 
            self.loss_fn = loss_fn 
            self.test_fn = self.test_prediction_second
        print(f"arDCA model initialized with {model}")

        
    def remove_autocorr(self):
        """Removes the self-interactions from the model."""
        self.J.data = self.J.data * self.mask.data

    def restore_graph(self):
            """Removes the interactions from the model which are not present in the graph."""
            self.J.data = self.J.data * self.graph_J.data
            self.h.data = self.h.data * self.graph_h.data

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
        
        # MSA ordering
        X = X[:, self.sorting, :]
            
        return X

    def sample_autoregressive(
        self,
        X: torch.Tensor,
        beta: float = 1.0,):
        """Predict using arDCA by sequentially filling the last third of the sequence."""
        
        l = X.size(1)
        n_samples = X.size(0)
        X_pred = torch.zeros(n_samples, self.L, self.q, dtype=self.h.dtype, device=self.h.device)
        X_pred[:, :l, :] = X.clone()  
        for i in range(l, self.L):
            prob_i = self.forward(X_pred[:, :i, :], beta=beta)
            sample_i = torch.multinomial(prob_i, num_samples=1).squeeze(1)
            X_pred[:, i, :] = nn.functional.one_hot(sample_i, num_classes=self.q).to(dtype=X_pred.dtype)

        return X_pred

    def compute_mean_error(
        self, 
        X1: torch.Tensor, 
        X2: torch.Tensor):
        """Compute the mean agreement between two predictions."""
        return (X1.argmax(dim=-1) == X2.argmax(dim=-1)).float().mean(dim=0)

    def predict_third_ML(
        self,
        X: torch.Tensor,
        beta: float = 1.0,):
        """Predict using arDCA by sequentially filling the last third of the sequence."""
        X_pred = X.clone()
        l = self.L // 3
        for i in range(self.L - l, self.L):
            prob = self.forward(X_pred[:, :i, :], beta=beta)
            X_pred[:, i] = nn.functional.one_hot(prob.argmax(dim=1), self.q).to(dtype=self.h.dtype)

        return X_pred


    def predict_second_ML(
        self,
        X: torch.Tensor,
        beta: float = 1.0,):
        """Predict using arDCA by sequentially filling the last third of the sequence."""
        X_pred = X.clone()
        l = self.L // 2
        for i in range(self.L - l, self.L):
            prob = self.forward(X_pred[:, :i, :], beta=beta)
            X_pred[:, i] = nn.functional.one_hot(prob.argmax(dim=1), self.q).to(dtype=self.h.dtype)

        return X_pred


    def test_prediction_third(
        self, 
        X_data: torch.Tensor):
        X_pred = torch.zeros_like(X_data)
        X_pred = self.predict_third_ML(X_data)

        return self.compute_mean_error(X_pred[:, 2 * self.L // 3:, :], X_data[:, 2 * self.L // 3:, :]).mean().item()

    def test_prediction_second(
        self, 
        X_data: torch.Tensor):
        X_pred = torch.zeros_like(X_data)
        X_pred = self.predict_second_ML(X_data)
        
        return self.compute_mean_error(X_pred[:, self.L // 2:, :], X_data[:,  self.L // 2:, :]).mean().item()

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
        X_test: torch.Tensor = None,
        fij_target: torch.Tensor = None,
        fi_target: torch.Tensor = None,
        index: int = None,
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

        

        ro_fi_prediction, ro_fi_input, ro_cij_prediction, ro_cij_input, ro_cij_prediction_test = [], [], [], [], []

        
        
        # Set Entropic Order if required
        fi = get_freq_single_point(X, weights=weights, pseudo_count=pseudo_count)
        if use_entropic_order:
            if fix_first_residue:
                entropic_order, inverse_entropic_order = get_entropic_order_with_inverse(fi[1:], index)
                device = entropic_order.device
                entropic_order = torch.cat([torch.tensor([0], device=device), entropic_order + 1])
                inverse_entropic_order = torch.cat([torch.tensor([0], device=device), inverse_entropic_order + 1])
            else:
                entropic_order, inverse_entropic_order = get_entropic_order_with_inverse(fi, index)

            self.sorting.data = torch.argsort(entropic_order)
            self.entropic_order.data = entropic_order 
            self.inverse_entropic_order.data =  inverse_entropic_order 
            X = X[:, entropic_order, :]
            X_test = X_test[:, entropic_order, :] if X_test is not None else None

        # Target frequencies, if entropic order is used, the frequencies are sorted
        fi_target = get_freq_single_point(X, weights=weights, pseudo_count=pseudo_count)
        fij_target = get_freq_two_points(X, weights=weights,  pseudo_count=pseudo_count)

        if X_test is not None:
            weights_test = torch.ones(X_test.shape[0]).to(device=X_test.device)
            fi_test = get_freq_single_point(X_test, weights=weights_test, pseudo_count=pseudo_count)
            fij_test = get_freq_two_points(X_test,  weights=weights_test, pseudo_count=pseudo_count)

        self.h.data = torch.log(fi_target + 1e-10)
        callback = EarlyStopping(patience=50, epsconv=epsconv)

        # Set Updating Bar
        pbar = tqdm(
            total=max_epochs,
            colour="red",
            dynamic_ncols=True,
            leave=False,
            ascii="-#",
            desc="Loss: inf"
        )
        metrics = {'Train Accuracy': "0.0"}
        if X_test is not None:
            metrics.update({'Test Accuracy': "0.0",
                'Shuffled Test Accuracy': "0.0"})
            val_losses, val_log_likelihoods = [], []
        pbar.set_postfix(metrics)


        # Training Loop
        losses, log_likelihoods = [], []
        for epoch in range(max_epochs):
            optimizer.zero_grad()
            loss, log_likelihood = self.loss_fn(self, X, weights, fi_target, fij_target, reg_h=reg_h, reg_J=reg_J)
            loss_value = loss.item()
            if loss_value < 0:
                raise ValueError("Negative loss encountered. Try to increase the regularization.")
            losses.append(loss_value)
            log_likelihoods.append(log_likelihood.item())

            loss.backward()
            optimizer.step()

            self.restore_graph()
            self.remove_autocorr()

            pbar.update()  # defaults to +1
            pbar.set_description(f"Loss: {loss_value:.3f}")


            if epoch % 10 == 0:
                # Compute Accuracy
                metrics = {'Train Accuracy': f"{self.test_fn(X):.5f}"}
                if X_test is not None:
                    X_test_shuffle = X_test.clone()
                    X_test_shuffle[:, index:, :] = X_test[torch.randperm(X_test.size(0)), index:, :]
                    metrics.update({'Test Accuracy':           f"{self.test_fn(X_test):.5f}",
                                    'Shuffled Test Accuracy':  f"{self.test_fn(X_test_shuffle):.5f}"})
                pbar.set_postfix(metrics)
                # Compute Training Pearson Cij
                samples = self.sample_autoregressive(X[:, :index, :])
                data, data_target = samples[:, index:, :], X[:, index:, :]
                pi, pij  = get_freq_single_point(data=data, weights=weights, pseudo_count=pseudo_count), get_freq_two_points(  data=data, weights=weights, pseudo_count=pseudo_count)
                fi_target_pred, fij_target_pred  = get_freq_single_point(data=data_target, weights=weights, pseudo_count=pseudo_count), get_freq_two_points(  data=data_target, weights=weights, pseudo_count=pseudo_count)
                pearson_cij_prediction, _ = get_correlation_two_points(fi=fi_target_pred, fij=fij_target_pred, pi=pi, pij=pij)
                metrics['Pearson cij (Train)'] = f"{pearson_cij_prediction:.5f}"
                ro_cij_prediction.append(pearson_cij_prediction)


                if X_test is not None:
                    with torch.no_grad():
                        val_loss, val_log_likelihood = self.loss_fn(self, X_test, weights_test, fi_test, fij_test, reg_h=reg_h, reg_J=reg_J)
                        val_losses.append(val_loss.item())
                        val_log_likelihoods.append(val_log_likelihood.item())
                        metrics['Val Loss'] = f"{val_loss:.5f}"

                    samples = self.sample_autoregressive(X_test[:, :index, :]) 
                    data, data_target = samples[:, index:, :], X_test[:, index:, :]
                    pi, pij = get_freq_single_point(data), get_freq_two_points(data)
                    fi_target_pred, fij_target_pred = get_freq_single_point(data=data_target, weights=weights_test, pseudo_count=pseudo_count), get_freq_two_points(data=data_target, weights=weights_test, pseudo_count=pseudo_count)
                    pearson_cij_prediction_test, _ = get_correlation_two_points(fi=fi_target_pred, fij=fij_target_pred, pi=pi, pij=pij)
                    metrics['Pearson cij (Test)'] = f"{pearson_cij_prediction_test:.5f}"
                    ro_cij_prediction_test.append(pearson_cij_prediction_test)
                pbar.set_postfix(metrics)

            if callback(loss):
                break

        pbar.close()
        return loss, ro_fi_prediction, ro_fi_input, ro_cij_prediction, ro_cij_input, ro_cij_prediction_test, losses, val_losses, log_likelihoods, val_log_likelihoods
        









    def fit_batch(
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
        X_test: torch.Tensor = None,
        batch_size: int = 500,
        index: int = None,
    ) -> None:
        """
        Adatta il modello ai dati, suddividendo il training e il test in mini batch.

        Args:
            X (torch.Tensor): Input MSA one-hot encoded.
            weights (torch.Tensor): Pesi delle sequenze nell'MSA.
            optimizer (torch.optim.Optimizer): Ottimizzatore da utilizzare.
            max_epochs (int, opzionale): Numero massimo di epoche.
            epsconv (float, opzionale): Soglia di convergenza.
            pseudo_count (float, opzionale): Pseudo-count per le frequenze.
            use_entropic_order (bool, opzionale): Se utilizzare l'ordine entropico.
            fix_first_residue (bool, opzionale): Se fissare il primo residuo.
            reg_h (float, opzionale): Regolarizzazione sugli bias.
            reg_J (float, opzionale): Regolarizzazione sulle interazioni.
            X_test (torch.Tensor, opzionale): Dati di test (senza pesi).
            batch_size (int, opzionale): Dimensione dei mini batch.
        """       
        

        # Riordinamento entropico se richiesto
        fi = get_freq_single_point_batches(X, weights=weights, pseudo_count=pseudo_count, batch_size=batch_size, num_classes=self.q, device=self.h.device)
        if use_entropic_order:
            if fix_first_residue:
                entropic_order, inverse_entropic_order = get_entropic_order_with_inverse(fi[1:], index)
                device = entropic_order.device
                entropic_order = torch.cat([torch.tensor([0], device=device), entropic_order + 1])
                inverse_entropic_order = torch.cat([torch.tensor([0], device=device), inverse_entropic_order + 1])
            else:
                entropic_order, inverse_entropic_order = get_entropic_order_with_inverse(fi, index)
            self.sorting.data = torch.argsort(entropic_order)
            self.entropic_order.data = entropic_order 
            self.inverse_entropic_order.data =  inverse_entropic_order 
            X = X[:, entropic_order]
            X_test = X_test[:, entropic_order] if X_test is not None else None

        # Calcolo delle frequenze target (fi e fij)
        fi_target = get_freq_single_point_batches(X, weights=weights, pseudo_count=pseudo_count, batch_size=batch_size, num_classes=self.q, device=self.h.device)
        self.h.data = torch.log(fi_target + 1e-10)
        
        # Creazione dei DataLoader per training e test
        train_dataset = TensorDataset(X, weights)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        if X_test is not None:
            # Assumiamo che per il test non siano necessari pesi.
            test_dataset = TensorDataset(X_test)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        callback = EarlyStopping(patience=5, epsconv=epsconv)
        
       # Set Updating Bar
        pbar = tqdm(
            total=max_epochs,
            colour="red",
            dynamic_ncols=True,
            leave=False,
            ascii="-#",
            desc="Loss: inf"
        )
        metrics = {'Train Accuracy': "0.0"}
        if X_test is not None:
            metrics.update({'Test Accuracy': "0.0"})
            val_losses, val_log_likelihoods = [], []
        pbar.set_postfix(metrics)


        losses, log_likelihoods = [], []
        losses_val, log_likelihoods_val = [], []
        for epoch in range(max_epochs):
            epoch_loss, epoch_log_likelihood = 0.0, 0.0
            # Ciclo sui mini batch del training
            for batch_X, batch_weights in train_loader:
                optimizer.zero_grad()
                batch_X = one_hot(batch_X, num_classes=self.q).to(self.h.dtype) 
                fij_target = get_freq_two_points(batch_X, weights=batch_weights, pseudo_count=pseudo_count)
                # Calcolo della loss sul mini batch
                loss, log_likelihood = self.loss_fn(self, batch_X, batch_weights, fi_target, fij_target, reg_h=reg_h, reg_J=reg_J)
                if loss.item() < 0:
                    raise ValueError("Negative loss encountered. Try to increase the regularization.")
                loss.backward()
                optimizer.step()

                self.restore_graph()
                self.remove_autocorr()

                # Accumulo della loss pesata per il numero di campioni nel batch
                epoch_loss += loss.item() * batch_X.size(0)
                epoch_log_likelihood += log_likelihood.item() * batch_X.size(0)
            
            # Calcolo della loss media sull’intero training set
            epoch_loss /= len(train_dataset)
            epoch_log_likelihood /= len(train_dataset)

            losses.append(epoch_loss)
            log_likelihoods.append(epoch_log_likelihood)
            # Aggiornamento della barra di progresso
            pbar.update()  # defaults to +1
            pbar.set_description(f"Loss: {epoch_loss:.3f}")
            
            if epoch % 10 == 0:
                # Compute Accuracy
                with torch.no_grad():

                    train_acc = 0 
                    for X_batch in train_loader:
                        X_batch = one_hot(X_batch[0], num_classes=self.q).to(dtype=self.h.dtype)
                        train_acc += self.test_fn(X_batch)
                    train_acc /= len(train_loader)  # Calcola l'accuratezza media sul training set

                    # Valutazione sul test set se definito
                    if X_test is not None:
                        test_acc = 0 
                        epoch_loss_val, epoch_log_likelihood_val = 0, 0
                        for X_batch in test_loader:
                            batch_weights = torch.ones(X_batch[0].size(0), device=X_batch[0].device)
                            X_batch = one_hot(X_batch[0], num_classes=self.q).to(dtype=self.h.dtype)
                            test_acc += self.test_fn(X_batch)

                            fij_target = get_freq_two_points(X_batch, weights=batch_weights, pseudo_count=pseudo_count)
                            loss_val, log_likelihood_val = self.loss_fn(self, X_batch, batch_weights, fi_target, fij_target, reg_h=reg_h, reg_J=reg_J)

                            epoch_loss_val += loss_val.item() * X_batch.size(0)
                            epoch_log_likelihood_val += log_likelihood_val.item() * X_batch.size(0)

                        epoch_loss_val /= len(test_dataset)
                        epoch_log_likelihood_val /= len(test_dataset)

                        losses_val.append(epoch_loss_val)
                        log_likelihoods_val.append(epoch_log_likelihood_val)

                        metrics['Val Loss'] = f"{epoch_loss_val:.5f}"

                        # Unisci tutti i batch in un unico tensore
                        test_acc /= len(test_loader)
                        metrics = {'Train Accuracy': f"{train_acc:.5f}", 'Test Accuracy': f"{test_acc:.5f}"}
                    else:
                        metrics = {'Train Accuracy': f"{train_acc:.5f}"}
                    
                    pbar.set_postfix(metrics)
                
            if callback(torch.tensor(epoch_loss)):
                break

        pbar.close()
        return epoch_loss, losses, log_likelihoods, losses_val, log_likelihoods_val
