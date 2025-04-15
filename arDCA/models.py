import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from tqdm import tqdm
# from adabmDCA.stats import get_freq_single_point, get_freq_two_points
from adabmDCA.io import save_params, load_params
from adabmDCA.functional import one_hot
from adabmDCA.stats import get_freq_two_points, get_correlation_two_points


import torch
import torch.nn.functional as F

def get_freq_single_point_batches(
    data: torch.Tensor,
    weights: torch.Tensor | None = None,
    pseudo_count: float = 0.0,
    batch_size: int = 32,
    num_classes: int | None = None,
    device: torch.device = torch.device("cuda"),
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Calcola le frequenze a singolo punto elaborando i dati per batch.
    
    Args:
        data (torch.Tensor): Dati grezzi, ad esempio di forma (N, L) dove N è il numero di sequenze.
        weights (torch.Tensor | None): Vettore dei pesi per ciascuna sequenza (forma (N,)) oppure None.
        pseudo_count (float): Pseudo count da applicare (default 0.0).
        batch_size (int): Dimensione dei batch per elaborare i dati.
        num_classes (int | None): Numero di classi per l'encoding one-hot. Deve essere specificato.
        device (torch.device): Dispositivo su cui eseguire i calcoli (default "cpu").
        dtype (torch.dtype): Tipo di dato usato per i calcoli (default torch.float32).
        
    Returns:
        torch.Tensor: Frequenze a singolo punto (shape (L, num_classes)).
        
    Raises:
        ValueError: Se non viene passato il parametro num_classes.
    """
    if num_classes is None:
        raise ValueError("Il parametro num_classes deve essere specificato per la codifica one-hot.")
        
    total_weight = 0.0
    frequency_accum = None
    N = data.shape[0]  # Numero totale di sequenze

    for i in range(0, N, batch_size):
        # Estrae il batch e lo porta sul device specificato
        batch = data[i : i + batch_size].to(device)
        
        # Codifica one-hot: da (batch_size, L) a (batch_size, L, num_classes)
        one_hot_batch = one_hot(batch, num_classes=num_classes).to(dtype)
        
        if weights is not None:
            batch_weights = weights[i : i + batch_size].to(device, dtype=dtype)
        else:
            # Se non specificati, il peso è 1 per ogni sequenza
            batch_weights = torch.ones(batch.shape[0], device=device, dtype=dtype)
            
        # Adattiamo il shape dei pesi in modo da poterne effettuare la broadcast
        batch_weights = batch_weights.view(-1, 1, 1)
        
        # Calcola la somma pesata del batch lungo la dimensione delle sequenze (dim=0)
        batch_freq = (one_hot_batch * batch_weights).sum(dim=0)  # shape: (L, num_classes)
        
        # Accumula le frequenze calcolate per il batch
        if frequency_accum is None:
            frequency_accum = batch_freq
        else:
            frequency_accum += batch_freq
        
        # Accumula il peso totale (non normalizzato) per il batch
        total_weight += batch_weights.sum()
    
    # Frequenza complessiva: normale accumulazione pesata
    overall_freq = frequency_accum / total_weight
    
    # Clampea (anche se in questo caso non dovrebbero esserci negativi) e applica il pseudo count
    overall_freq.clamp_(min=0.0)
    overall_freq = (1.0 - pseudo_count) * overall_freq + (pseudo_count / num_classes)
    
    return overall_freq



# @torch.jit.script
# def _get_freq_single_point(
#     data: torch.Tensor,
#     weights: torch.Tensor,
#     pseudo_count: float,
# ) -> torch.Tensor:    
#     _, _, q = data.shape
#     frequencies = (data * weights).sum(dim=0)
#     # Set to zero the negative frequencies. Used for the reintegration.
#     torch.clamp_(frequencies, min=0.0)

#     return (1. - pseudo_count) * frequencies + (pseudo_count / q)

# def get_freq_single_point(
#     data: torch.Tensor,
#     weights: torch.Tensor | None = None,
#     pseudo_count: float = 0.0,
# ) -> torch.Tensor:
#     """Computes the single point frequencies of the input MSA.
#     Args:
#         data (torch.Tensor): One-hot encoded data array.
#         weights (torch.Tensor | None, optional): Weights of the sequences.
#         pseudo_count (float, optional): Pseudo count to be added to the frequencies. Defaults to 0.0.
    
#     Raises:
#         ValueError: If the input data is not a 3D tensor.

#     Returns:
#         torch.Tensor: Single point frequencies.
#     """
#     if data.dim() != 3:
#         raise ValueError(f"Expected data to be a 3D tensor, but got {data.dim()}D tensor instead")
#     M = len(data)
#     if weights is not None:
#         norm_weights = weights.reshape(M, 1, 1) / weights.sum()
#     else:
#         norm_weights = torch.ones((M, 1, 1), device=data.device, dtype=data.dtype) / M
    
#     return _get_freq_single_point(data, norm_weights, pseudo_count)

# @torch.jit.script
# def _get_freq_two_points(
#     data: torch.Tensor,
#     weights: torch.Tensor,
#     pseudo_count: float,
# ) -> torch.Tensor:
    
#     M, L, q = data.shape
#     data_oh = data.reshape(M, q * L)
    
#     fij = (data_oh * weights).T @ data_oh
#     # Apply the pseudo count
#     fij = (1. - pseudo_count) * fij + (pseudo_count / q**2)
#     # Diagonal terms must represent the single point frequencies
#     fi = get_freq_single_point(data, weights, pseudo_count).ravel()
#     # Apply the pseudo count on the single point frequencies
#     fij_diag = (1. - pseudo_count) * fi + (pseudo_count / q)
#     # Set the diagonal terms of fij to the single point frequencies
#     fij = torch.diagonal_scatter(fij, fij_diag, dim1=0, dim2=1)
#     # Set to zero the negative frequencies. Used for the reintegration.
#     torch.clamp_(fij, min=0.0)
    
#     return fij.reshape(L, q, L, q)


# def get_freq_two_points(
#     data: torch.Tensor,
#     weights: torch.Tensor | None = None,
#     pseudo_count: float = 0.0,
# ) -> torch.Tensor:
#     """
#     Computes the 2-points statistics of the input MSA.

#     Args:
#         data (torch.Tensor): One-hot encoded data array.
#         weights (torch.Tensor | None, optional): Array of weights to assign to the sequences of shape.
#         pseudo_count (float, optional): Pseudo count for the single and two points statistics. Acts as a regularization. Defaults to 0.0.
    
#     Raises:
#         ValueError: If the input data is not a 3D tensor.

#     Returns:
#         torch.Tensor: Matrix of two-point frequencies of shape (L, q, L, q).
#     """
#     if data.dim() != 3:
#         raise ValueError(f"Expected data to be a 3D tensor, but got {data.dim()}D tensor instead")
    
#     M = len(data)
#     if weights is not None:
#         norm_weights = weights.reshape(M, 1) / weights.sum()
#     else:
#         norm_weights = torch.ones((M, 1), device=data.device, dtype=data.dtype) / M
    
#     return _get_freq_two_points(data, norm_weights, pseudo_count)




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
    
    return - log_likelihood + reg_h * torch.norm(model.h)**2 + reg_J * torch.norm(model.J)**2

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
    
    return - log_likelihood + reg_h * torch.norm(model.h)**2 + reg_J * torch.norm(model.J)**2


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
        if model == "third":
            self.loss_fn = loss_third_fn
            self.test_fn = self.test_prediction_third
        elif model == "second":
            self.loss_fn = loss_second_fn
            self.test_fn = self.test_prediction_second
        
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
            self.sorting.data = torch.argsort(entropic_order)
            X = X[:, entropic_order, :]

        # Target frequencies, if entropic order is used, the frequencies are sorted
        fi_target = get_freq_single_point(X, weights=weights, pseudo_count=pseudo_count)
        fij_target = get_freq_two_points(X, weights=weights, pseudo_count=pseudo_count)


        self.h.data = torch.log(fi_target + 1e-10)
        callback = EarlyStopping(patience=50, epsconv=epsconv)

        # Training loop
        pbar = tqdm(
            total=max_epochs,
            colour="red",
            dynamic_ncols=True,
            leave=False,
            ascii="-#",
        )
        pbar.set_description(f"Loss: inf")
        metrics = {'Train Accuracy': f"{0.0}"}
        if X_test is not None:
            metrics['Test Accuracy'] = f"{0.0}"
        pbar.set_postfix(metrics)

        prev_loss = float("inf")

        for epoch in range(max_epochs):
            optimizer.zero_grad()
            loss = self.loss_fn(self, X, weights, fi_target, fij_target, reg_h=reg_h, reg_J=reg_J)
            if loss < 0:
                raise ValueError("Negative loss encountered. Try to increase the regularization.")
            loss.backward()
            optimizer.step()
            self.restore_graph()
            self.remove_autocorr()
            
            pbar.update(1)
            pbar.set_description(f"Loss: {loss.item():.3f}")
            if epoch % 10 == 0:
                metrics = {'Train Accuracy': f"{self.test_fn(X):.5f}"}
                if X_test is not None:
                    metrics['Test Accuracy'] = f"{self.test_fn(X_test):.5f}"
                pbar.set_postfix(metrics)
                    
            # if torch.abs(loss - prev_loss) < epsconv:
            #     break
            # prev_loss = loss
                            

            if callback(loss):
                break
        pbar.close()
        return loss
        

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
        batch_size: int = 500 #32
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
        if use_entropic_order:
            if fix_first_residue:
                entropic_order = get_entropic_order(fi[1:])
                entropic_order = torch.cat([torch.tensor([0], device=entropic_order.device), entropic_order + 1])
            else:
                entropic_order = get_entropic_order(fi)
            self.sorting.data = torch.argsort(entropic_order)
            X = X[:, entropic_order, :]
        
        # Calcolo delle frequenze target (fi e fij)
        fi_target = get_freq_single_point_batches(X, weights=weights, pseudo_count=pseudo_count, batch_size=batch_size, num_classes=self.q, device=self.h.device) # get_freq_single_point(X, weights=weights, pseudo_count=pseudo_count)
        self.h.data = torch.log(fi_target + 1e-10)
        
        # Creazione dei DataLoader per training e test
        train_dataset = TensorDataset(X, weights)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        if X_test is not None:
            # Assumiamo che per il test non siano necessari pesi.
            test_dataset = TensorDataset(X_test)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        callback = EarlyStopping(patience=5, epsconv=epsconv)
        
        pbar = tqdm(total=max_epochs, colour="red", dynamic_ncols=True, leave=False, ascii="-#")
        pbar.set_description(f"Loss: inf")
        metrics = {'Train Accuracy': f"{0.0}"}
        if X_test is not None:
            metrics['Test Accuracy'] = f"{0.0}"
        pbar.set_postfix(metrics)
        
        prev_loss = float("inf")
        for epoch in range(max_epochs):
            epoch_loss = 0.0
            # Ciclo sui mini batch del training
            for batch_X, batch_weights in train_loader:
                optimizer.zero_grad()
                batch_X = one_hot(batch_X, num_classes=self.q).to(self.h.dtype) 
                fij_target = get_freq_two_points(batch_X, weights=batch_weights, pseudo_count=pseudo_count)
                # Calcolo della loss sul mini batch
                loss = self.loss_fn(self, batch_X, batch_weights, fi_target, fij_target, reg_h=reg_h, reg_J=reg_J)
                if loss.item() < 0:
                    raise ValueError("Negative loss encountered. Try to increase the regularization.")
                loss.backward()
                optimizer.step()
                self.restore_graph()
                self.remove_autocorr()
                # Accumulo della loss pesata per il numero di campioni nel batch
                epoch_loss += loss.item() * batch_X.size(0)
            
            # Calcolo della loss media sull’intero training set
            epoch_loss /= len(train_dataset)
            pbar.update(1)
            pbar.set_description(f"Loss: {epoch_loss:.3f}")
            
            if epoch % 10 == 0:
                # Valutazione sulla parte di training
                with torch.no_grad():
                    train_acc = 0 
                    for X_batch in train_loader:
                        X_batch = one_hot(X_batch[0], num_classes=self.q).to(dtype=self.h.dtype)
                        train_acc += self.test_fn(X_batch)
                    train_acc /= len(train_loader)  # Calcola l'accuratezza media sul training set

                    # Valutazione sul test set se definito
                    if X_test is not None:
                        test_acc = 0 
                        for X_batch in test_loader:
                            X_batch = one_hot(X_batch[0], num_classes=self.q).to(dtype=self.h.dtype)
                            test_acc += self.test_fn(X_batch)
                        # Unisci tutti i batch in un unico tensore
                        test_acc /= len(test_loader)
                        metrics = {'Train Accuracy': f"{train_acc:.5f}", 'Test Accuracy': f"{test_acc:.5f}"}
                    else:
                        metrics = {'Train Accuracy': f"{train_acc:.5f}"}
                    
                    pbar.set_postfix(metrics)
                
            if callback(torch.tensor(epoch_loss)):
                break
        pbar.close()
        return epoch_loss
