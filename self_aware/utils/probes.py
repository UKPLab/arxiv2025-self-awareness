import torch
import torch.nn as nn
from torch.nn import functional as F


class SparseProbe(nn.Module):
    """Sparse probing with random orthonormal matrix"""

    def __init__(self, input_dim, bias_init=0.0, non_linearity=None, M=None):
        super().__init__()
        self.input_dim = input_dim
        self.bias_init = bias_init
        self.non_linearity = non_linearity

        # Create orthonormal matrix for feature projection
        if M is not None:
            self.M = M
        else:
            self.M = SparseProbe.get_orthonormal(self.input_dim, seed=123)  # Default seed, will be overridden

        # Initialize trainable parameters
        self.w = nn.Parameter(torch.randn(self.input_dim))  # Sparse vector for feature selection
        self.w_c = nn.Parameter(torch.randn(self.input_dim))  # Classifier weights
        self.b = nn.Parameter(torch.tensor([self.bias_init]))  # Classifier bias

        if self.non_linearity:
            self.non_linearity_fn = eval(self.non_linearity)
        else:
            self.non_linearity_fn = lambda x: x  # Identity function if none specified

    def forward(self, x):
        """Perform forward pass through the classifier"""
        if self.M.device != self.w.device:
            self.M = self.M.to(self.w.device)
        x_hat = (x @ self.M) * (self.M.T @ self.w)
        x_hat = self.non_linearity_fn(x_hat)
        logits = torch.sum(self.w_c * x_hat, dim=-1) + self.b
        return logits

    def encode(self, x):
        if self.M.device != self.w.device:
            self.M = self.M.to(self.w.device)
        x_hat = (x @ self.M) * (self.M.T @ self.w)
        # x_hat = self.non_linearity_fn(x_hat)
        # x_hat = self.w_c * x_hat
        return x_hat

    @staticmethod
    def compute_loss(logits, labels, model, alpha=None, beta=None):
        """Compute the total loss including BCE and L1 regularization for sparse probing"""
        bce_loss = F.binary_cross_entropy_with_logits(logits, labels)

        # Add L1 regularization for sparse probing
        if alpha and beta:
            # L1 regularization terms
            l1_w = alpha * torch.norm(model.w, p=1)  # ||w||_1
            l1_wc = beta * torch.norm(model.w_c, p=1)  # ||w_c||_1

            # Total loss with regularization
            total_loss = bce_loss + l1_w + l1_wc
            return total_loss
        else:
            # Use only BCE loss if regularization parameters not specified
            return bce_loss

    @staticmethod
    def get_orthonormal(n, seed):
        """Creates an orthonormal matrix with shape [n, n]"""
        torch.manual_seed(seed)
        random_matrix = torch.randn(n, n)
        Q, _ = torch.linalg.qr(random_matrix)
        if torch.det(Q) < 0:
            Q[:, 0] = -Q[:, 0]  # Ensure it's a proper rotation matrix
        return Q


class RegressorProbe(nn.Module):
    """A simple logistic regression model"""

    def __init__(self, input_dim, hidden_dim=None, bias_init=0.0):
        super().__init__()
        self.input_dim = input_dim
        self.bias_init = bias_init

        self.fc1 = nn.Linear(self.input_dim, 1)

        if self.bias_init != 0.0:
            self.fc1.bias.data.fill_(self.bias_init)

    def forward(self, x):
        x = self.fc1(x)
        return x.squeeze(-1)  # Return [B] instead of [B, 1]

    def encode(self, x):
        return x @ self.fc1.weight.T

    @staticmethod
    def compute_loss(logits, labels):
        """Compute the binary cross entropy loss"""
        bce_loss = F.binary_cross_entropy_with_logits(logits, labels)
        return bce_loss


class NNProbe(nn.Module):
    """A simple 2-layer neural network"""

    def __init__(self, non_linearity, input_dim, hidden_dim=None, bias_init=0.0):
        super().__init__()
        self.input_dim = input_dim
        self.bias_init = bias_init
        self.non_linearity = non_linearity

        if hidden_dim:
            self.hidden_dim = hidden_dim
        else:
            self.hidden_dim = 2 * self.input_dim

        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.non_linearity_fn = eval(self.non_linearity)
        self.fc2 = nn.Linear(self.hidden_dim, 1)

        # Initialize with a small bias to break symmetry
        if self.bias_init != 0.0:
            self.fc2.bias.data.fill_(self.bias_init)

    def forward(self, x):
        x = self.fc1(x)
        x = self.non_linearity_fn(x)
        x = self.fc2(x)
        return x.squeeze(-1)  # Return [B] instead of [B, 1]

    def encode(self, x):
        x = self.fc1(x)
        # x = self.non_linearity_fn(x)
        return x.squeeze(-1)  # Return [B] instead of [B, 1]

    @staticmethod
    def compute_loss(logits, labels):
        """Compute the binary cross entropy loss"""
        bce_loss = F.binary_cross_entropy_with_logits(logits, labels)
        return bce_loss

