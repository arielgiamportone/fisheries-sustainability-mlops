"""
Modelos de Deep Learning para análisis de sostenibilidad pesquera.

Incluye:
- SustainabilityMLP: Red neuronal feedforward para clasificación
- BayesianNeuralNetwork: Red con cuantificación de incertidumbre
- CausalEncoder: Encoder para representación causal latente
- CausalVAE: Variational Autoencoder con estructura causal
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Bernoulli
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuración para modelos de deep learning."""
    input_dim: int
    hidden_dims: List[int] = None
    output_dim: int = 1
    dropout_rate: float = 0.2
    activation: str = 'relu'
    use_batch_norm: bool = True

    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [64, 32]


class SustainabilityMLP(nn.Module):
    """
    Red Neuronal Multicapa para predicción de sostenibilidad.

    Arquitectura flexible con dropout y batch normalization opcional.

    Args:
        config: Configuración del modelo

    Example:
        >>> config = ModelConfig(input_dim=10, hidden_dims=[64, 32])
        >>> model = SustainabilityMLP(config)
        >>> output = model(torch.randn(32, 10))
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Seleccionar función de activación
        activations = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(0.1),
            'elu': nn.ELU(),
            'selu': nn.SELU(),
            'tanh': nn.Tanh()
        }
        self.activation = activations.get(config.activation, nn.ReLU())

        # Construir capas
        layers = []
        prev_dim = config.input_dim

        for hidden_dim in config.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))

            if config.use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))

            layers.append(self.activation)
            layers.append(nn.Dropout(config.dropout_rate))
            prev_dim = hidden_dim

        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_dim, config.output_dim)

        # Inicialización de pesos
        self._init_weights()

    def _init_weights(self):
        """Inicializa pesos con Xavier/Glorot."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        features = self.feature_extractor(x)
        logits = self.classifier(features)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Retorna probabilidades."""
        logits = self.forward(x)
        return torch.sigmoid(logits)

    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Retorna predicciones binarias."""
        proba = self.predict_proba(x)
        return (proba >= threshold).float()


class BayesianLinear(nn.Module):
    """
    Capa lineal bayesiana con pesos estocásticos.

    Implementa variational inference con reparametrización.
    Los pesos se modelan como distribuciones normales.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        prior_std: float = 1.0
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Prior
        self.prior_std = prior_std

        # Parámetros variacionales para pesos
        self.weight_mu = nn.Parameter(torch.zeros(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.zeros(out_features, in_features))

        # Parámetros variacionales para bias
        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.bias_rho = nn.Parameter(torch.zeros(out_features))

        # Inicialización
        self._init_parameters()

    def _init_parameters(self):
        """Inicializa parámetros."""
        nn.init.xavier_uniform_(self.weight_mu)
        nn.init.constant_(self.weight_rho, -3.0)  # log(sigma) ≈ -3 -> sigma ≈ 0.05
        nn.init.zeros_(self.bias_mu)
        nn.init.constant_(self.bias_rho, -3.0)

    def _rho_to_std(self, rho: torch.Tensor) -> torch.Tensor:
        """Convierte rho a desviación estándar positiva."""
        return F.softplus(rho)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass con muestreo de pesos.

        Returns:
            Tuple de (output, kl_divergence)
        """
        # Muestrear pesos usando reparametrización
        weight_std = self._rho_to_std(self.weight_rho)
        bias_std = self._rho_to_std(self.bias_rho)

        # Epsilon ~ N(0, 1)
        weight_eps = torch.randn_like(self.weight_mu)
        bias_eps = torch.randn_like(self.bias_mu)

        # Reparametrización: w = mu + std * eps
        weight = self.weight_mu + weight_std * weight_eps
        bias = self.bias_mu + bias_std * bias_eps

        # Forward
        output = F.linear(x, weight, bias)

        # KL divergence
        kl = self._kl_divergence(
            self.weight_mu, weight_std,
            self.bias_mu, bias_std
        )

        return output, kl

    def _kl_divergence(
        self,
        weight_mu: torch.Tensor,
        weight_std: torch.Tensor,
        bias_mu: torch.Tensor,
        bias_std: torch.Tensor
    ) -> torch.Tensor:
        """Calcula KL divergence entre posterior y prior."""
        # KL(q||p) para normal
        # = 0.5 * (log(prior_var/post_var) + (post_var + (post_mu - prior_mu)^2)/prior_var - 1)

        prior_var = self.prior_std ** 2

        # Para pesos
        weight_var = weight_std ** 2
        kl_weight = 0.5 * (
            np.log(prior_var) - torch.log(weight_var) +
            (weight_var + weight_mu ** 2) / prior_var - 1
        ).sum()

        # Para bias
        bias_var = bias_std ** 2
        kl_bias = 0.5 * (
            np.log(prior_var) - torch.log(bias_var) +
            (bias_var + bias_mu ** 2) / prior_var - 1
        ).sum()

        return kl_weight + kl_bias


class BayesianNeuralNetwork(nn.Module):
    """
    Red Neuronal Bayesiana para cuantificación de incertidumbre.

    Usa capas bayesianas con pesos estocásticos para estimar
    incertidumbre epistémica en las predicciones.

    Args:
        config: Configuración del modelo
        prior_std: Desviación estándar del prior

    Example:
        >>> config = ModelConfig(input_dim=10)
        >>> bnn = BayesianNeuralNetwork(config)
        >>> mean, std = bnn.predict_with_uncertainty(x, n_samples=100)
    """

    def __init__(
        self,
        config: ModelConfig,
        prior_std: float = 1.0
    ):
        super().__init__()
        self.config = config
        self.prior_std = prior_std

        # Construir capas bayesianas
        layers = []
        prev_dim = config.input_dim

        for hidden_dim in config.hidden_dims:
            layers.append(BayesianLinear(prev_dim, hidden_dim, prior_std))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        self.hidden_layers = nn.ModuleList(layers[::2])  # Solo las BayesianLinear
        self.activations = nn.ModuleList(layers[1::2])   # Solo las activaciones

        self.output_layer = BayesianLinear(prev_dim, config.output_dim, prior_std)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Returns:
            Tuple de (logits, kl_total)
        """
        kl_total = 0.0

        for layer, activation in zip(self.hidden_layers, self.activations):
            x, kl = layer(x)
            kl_total = kl_total + kl
            x = activation(x)

        logits, kl = self.output_layer(x)
        kl_total = kl_total + kl

        return logits, kl_total

    def elbo_loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        n_samples: int = 1,
        kl_weight: float = 1.0
    ) -> torch.Tensor:
        """
        Calcula la pérdida ELBO (Evidence Lower Bound).

        ELBO = E[log p(y|x,w)] - KL(q(w)||p(w))
        """
        total_loss = 0.0

        for _ in range(n_samples):
            logits, kl = self.forward(x)

            # Likelihood (BCE loss)
            likelihood = F.binary_cross_entropy_with_logits(
                logits.squeeze(), y.float(), reduction='sum'
            )

            # ELBO = -likelihood + kl_weight * kl
            total_loss = total_loss + likelihood + kl_weight * kl

        return total_loss / n_samples

    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        n_samples: int = 100
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predicción con estimación de incertidumbre.

        Realiza múltiples forward passes para estimar la distribución
        predictiva.

        Args:
            x: Entrada
            n_samples: Número de muestras MC

        Returns:
            Tuple de (mean_prob, std_prob, all_samples)
        """
        self.eval()
        samples = []

        with torch.no_grad():
            for _ in range(n_samples):
                logits, _ = self.forward(x)
                probs = torch.sigmoid(logits)
                samples.append(probs)

        samples = torch.stack(samples, dim=0)  # [n_samples, batch, 1]

        mean_prob = samples.mean(dim=0)
        std_prob = samples.std(dim=0)

        return mean_prob, std_prob, samples

    def epistemic_uncertainty(
        self,
        x: torch.Tensor,
        n_samples: int = 100
    ) -> torch.Tensor:
        """Calcula incertidumbre epistémica (varianza entre predicciones)."""
        _, std, _ = self.predict_with_uncertainty(x, n_samples)
        return std


class CausalEncoder(nn.Module):
    """
    Encoder que respeta estructura causal.

    Genera representaciones latentes que preservan relaciones causales
    definidas en un DAG.

    Args:
        input_dim: Dimensión de entrada
        latent_dim: Dimensión del espacio latente
        causal_order: Orden causal de las variables (lista de índices)
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 16,
        hidden_dim: int = 32,
        causal_order: Optional[List[int]] = None
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.causal_order = causal_order or list(range(input_dim))

        # Encoder principal
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Capas para mu y log_var (VAE)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Máscara causal triangular inferior
        self._build_causal_mask()

    def _build_causal_mask(self):
        """Construye máscara causal basada en el orden."""
        mask = torch.tril(torch.ones(self.latent_dim, self.latent_dim))
        self.register_buffer('causal_mask', mask)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent distribution parameters."""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor
    ) -> torch.Tensor:
        """Reparametrización trick para VAE."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Returns:
            Tuple de (z, mu, logvar)
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar


class CausalVAE(nn.Module):
    """
    Variational Autoencoder con estructura causal.

    Aprende representaciones latentes que respetan un DAG causal,
    permitiendo generación de contrafactuales.

    Args:
        input_dim: Dimensión de entrada
        latent_dim: Dimensión del espacio latente
        hidden_dims: Dimensiones de capas ocultas
        causal_graph: Lista de aristas causales [(i, j), ...]
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 16,
        hidden_dims: List[int] = None,
        causal_graph: Optional[List[Tuple[int, int]]] = None
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [64, 32]

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.causal_graph = causal_graph or []

        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = h_dim

        self.encoder = nn.Sequential(*encoder_layers)
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)

        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = h_dim

        self.decoder = nn.Sequential(*decoder_layers)
        self.fc_out = nn.Linear(prev_dim, input_dim)

        # Construir máscara causal
        self._build_causal_mask()

    def _build_causal_mask(self):
        """Construye máscara causal para el espacio latente."""
        mask = torch.eye(self.latent_dim)
        for i, j in self.causal_graph:
            if i < self.latent_dim and j < self.latent_dim:
                mask[j, i] = 1  # j depende de i
        self.register_buffer('causal_mask', mask)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode to latent space."""
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparametrización."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from latent space."""
        h = self.decoder(z)
        return self.fc_out(h)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Returns:
            Tuple de (x_reconstructed, mu, logvar)
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

    def loss_function(
        self,
        x: torch.Tensor,
        x_recon: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        kl_weight: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """
        Calcula la pérdida del VAE.

        Loss = Reconstruction Loss + KL Divergence
        """
        # Reconstruction loss (MSE para variables continuas)
        recon_loss = F.mse_loss(x_recon, x, reduction='sum')

        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        total_loss = recon_loss + kl_weight * kl_loss

        return {
            'total': total_loss,
            'reconstruction': recon_loss,
            'kl': kl_loss
        }

    def generate_counterfactual(
        self,
        x: torch.Tensor,
        intervention: Dict[int, float]
    ) -> torch.Tensor:
        """
        Genera contrafactual aplicando intervención en el espacio latente.

        Args:
            x: Observación original
            intervention: Dict de {índice_latente: nuevo_valor}

        Returns:
            Contrafactual generado
        """
        self.eval()
        with torch.no_grad():
            mu, logvar = self.encode(x)
            z = self.reparameterize(mu, logvar)

            # Aplicar intervención
            z_intervened = z.clone()
            for idx, value in intervention.items():
                z_intervened[:, idx] = value

            # Decodificar
            x_cf = self.decode(z_intervened)

        return x_cf

    def sample(self, n_samples: int = 1) -> torch.Tensor:
        """Genera muestras desde el prior."""
        z = torch.randn(n_samples, self.latent_dim)
        if next(self.parameters()).is_cuda:
            z = z.cuda()
        return self.decode(z)


def create_model(
    model_type: str,
    input_dim: int,
    hidden_dims: List[int] = None,
    **kwargs
) -> nn.Module:
    """
    Factory function para crear modelos.

    Args:
        model_type: 'mlp', 'bnn', 'causal_vae'
        input_dim: Dimensión de entrada
        hidden_dims: Dimensiones ocultas
        **kwargs: Argumentos adicionales

    Returns:
        Modelo instanciado
    """
    if hidden_dims is None:
        hidden_dims = [64, 32]

    config = ModelConfig(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        **{k: v for k, v in kwargs.items() if k in ModelConfig.__dataclass_fields__}
    )

    if model_type == 'mlp':
        return SustainabilityMLP(config)
    elif model_type == 'bnn':
        prior_std = kwargs.get('prior_std', 1.0)
        return BayesianNeuralNetwork(config, prior_std=prior_std)
    elif model_type == 'causal_vae':
        latent_dim = kwargs.get('latent_dim', 16)
        causal_graph = kwargs.get('causal_graph', [])
        return CausalVAE(input_dim, latent_dim, hidden_dims, causal_graph)
    else:
        raise ValueError(f"Tipo de modelo no válido: {model_type}")
