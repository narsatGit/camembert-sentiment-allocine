import torch.nn as nn


class ReseauTest(nn.Module):
    r"""Réseau de neurones Perceptron Multi-Couches pour la classification de sentiment binaire.

    Attributes:
        couches (nn.Sequential): Séquence des couches linéaires et activations (ReLU, Sigmoid).
    """

    def __init__(self, input_dim):
        r"""Initialise les couches du réseau de neurones.

        Args:
            input_dim (int): Taille du vocabulaire d'entrée (dimension des vecteurs Bag-of-Words).
        """
        super().__init__()
        self.couches = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        r"""Réalise le passage avant (forward pass) à travers le réseau.

        Args:
            x (torch.Tensor): Tenseur d'entrée de forme (batch_size, input_dim).

        Returns:
            torch.Tensor: Probabilités prédites de forme (batch_size, 1).
        """
        return self.couches(x)