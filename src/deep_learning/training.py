"""
Funciones de entrenamiento para modelos de Deep Learning.

Incluye:
- Trainer: Clase para entrenamiento con callbacks
- EarlyStopping: Callback para detención temprana
- Funciones de entrenamiento y evaluación
- DataLoaders para datos de sostenibilidad
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from pathlib import Path
import json

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
from sklearn.preprocessing import LabelEncoder, StandardScaler


@dataclass
class TrainingConfig:
    """Configuración de entrenamiento."""
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    early_stopping_patience: int = 10
    val_split: float = 0.2
    device: str = 'auto'
    save_best: bool = True
    verbose: bool = True

    def __post_init__(self):
        if self.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'


@dataclass
class TrainingHistory:
    """Historial de entrenamiento."""
    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    train_metrics: List[Dict] = field(default_factory=list)
    val_metrics: List[Dict] = field(default_factory=list)
    best_epoch: int = 0
    best_val_loss: float = float('inf')

    def to_dict(self) -> Dict:
        return {
            'train_loss': self.train_loss,
            'val_loss': self.val_loss,
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics,
            'best_epoch': self.best_epoch,
            'best_val_loss': self.best_val_loss
        }

    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


class EarlyStopping:
    """
    Callback para detención temprana.

    Detiene el entrenamiento cuando la métrica de validación
    no mejora por un número de épocas especificado.
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = 'min',
        verbose: bool = True
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose

        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_state = None

    def __call__(
        self,
        score: float,
        model: nn.Module
    ) -> bool:
        """
        Verifica si debe detenerse.

        Args:
            score: Métrica actual
            model: Modelo actual

        Returns:
            True si debe detenerse
        """
        if self.mode == 'min':
            current_score = -score
        else:
            current_score = score

        if self.best_score is None:
            self.best_score = current_score
            self.best_model_state = model.state_dict().copy()
        elif current_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = current_score
            self.best_model_state = model.state_dict().copy()
            self.counter = 0

        return self.early_stop

    def restore_best_model(self, model: nn.Module):
        """Restaura el mejor modelo."""
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)


class Trainer:
    """
    Clase para entrenamiento de modelos de Deep Learning.

    Maneja el loop de entrenamiento, validación, callbacks y logging.

    Example:
        >>> trainer = Trainer(model, config)
        >>> history = trainer.fit(train_loader, val_loader)
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        loss_fn: Optional[nn.Module] = None,
        optimizer: Optional[optim.Optimizer] = None
    ):
        self.model = model.to(config.device)
        self.config = config
        self.device = config.device

        # Loss function
        self.loss_fn = loss_fn or nn.BCEWithLogitsLoss()

        # Optimizer
        self.optimizer = optimizer or optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        # Scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )

        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.early_stopping_patience,
            verbose=config.verbose
        )

        self.history = TrainingHistory()

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None
    ) -> TrainingHistory:
        """
        Entrena el modelo.

        Args:
            train_loader: DataLoader de entrenamiento
            val_loader: DataLoader de validación (opcional)

        Returns:
            Historial de entrenamiento
        """
        for epoch in range(self.config.epochs):
            # Training
            train_loss, train_metrics = self._train_epoch(train_loader)
            self.history.train_loss.append(train_loss)
            self.history.train_metrics.append(train_metrics)

            # Validation
            if val_loader is not None:
                val_loss, val_metrics = self._validate_epoch(val_loader)
                self.history.val_loss.append(val_loss)
                self.history.val_metrics.append(val_metrics)

                # Learning rate scheduling
                self.scheduler.step(val_loss)

                # Early stopping
                if self.early_stopping(val_loss, self.model):
                    if self.config.verbose:
                        print(f"Early stopping at epoch {epoch + 1}")
                    break

                # Track best
                if val_loss < self.history.best_val_loss:
                    self.history.best_val_loss = val_loss
                    self.history.best_epoch = epoch

            # Logging
            if self.config.verbose and (epoch + 1) % 10 == 0:
                self._log_epoch(epoch, train_loss, val_loss if val_loader else None)

        # Restore best model
        if val_loader is not None:
            self.early_stopping.restore_best_model(self.model)

        return self.history

    def _train_epoch(self, loader: DataLoader) -> Tuple[float, Dict]:
        """Ejecuta una época de entrenamiento."""
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        for batch_x, batch_y in loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            self.optimizer.zero_grad()

            # Forward
            outputs = self.model(batch_x)

            # Handle BNN output (logits, kl)
            if isinstance(outputs, tuple):
                logits, kl = outputs
                loss = self.loss_fn(logits.squeeze(), batch_y.float())
                loss = loss + 0.01 * kl / len(loader.dataset)
            else:
                logits = outputs
                loss = self.loss_fn(logits.squeeze(), batch_y.float())

            # Backward
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            # Predictions
            with torch.no_grad():
                probs = torch.sigmoid(logits.squeeze())
                preds = (probs >= 0.5).float()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())

        avg_loss = total_loss / len(loader)
        metrics = self._compute_metrics(all_labels, all_preds)

        return avg_loss, metrics

    def _validate_epoch(self, loader: DataLoader) -> Tuple[float, Dict]:
        """Ejecuta validación."""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                outputs = self.model(batch_x)

                if isinstance(outputs, tuple):
                    logits, kl = outputs
                    loss = self.loss_fn(logits.squeeze(), batch_y.float())
                    loss = loss + 0.01 * kl / len(loader.dataset)
                else:
                    logits = outputs
                    loss = self.loss_fn(logits.squeeze(), batch_y.float())

                total_loss += loss.item()

                probs = torch.sigmoid(logits.squeeze())
                preds = (probs >= 0.5).float()

                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())

        avg_loss = total_loss / len(loader)
        metrics = self._compute_metrics(all_labels, all_preds, all_probs)

        return avg_loss, metrics

    def _compute_metrics(
        self,
        y_true: List,
        y_pred: List,
        y_prob: Optional[List] = None
    ) -> Dict:
        """Calcula métricas de clasificación."""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0)
        }

        if y_prob is not None:
            try:
                metrics['auc_roc'] = roc_auc_score(y_true, y_prob)
            except ValueError:
                metrics['auc_roc'] = 0.5

        return metrics

    def _log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: Optional[float]
    ):
        """Log de época."""
        msg = f"Epoch {epoch + 1}/{self.config.epochs} - Train Loss: {train_loss:.4f}"
        if val_loss is not None:
            msg += f" - Val Loss: {val_loss:.4f}"
        print(msg)

    def predict(self, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """Realiza predicciones."""
        self.model.eval()
        all_preds = []
        all_probs = []

        with torch.no_grad():
            for batch_x, _ in loader:
                batch_x = batch_x.to(self.device)
                outputs = self.model(batch_x)

                if isinstance(outputs, tuple):
                    logits, _ = outputs
                else:
                    logits = outputs

                probs = torch.sigmoid(logits.squeeze())
                preds = (probs >= 0.5).float()

                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        return np.array(all_preds), np.array(all_probs)

    def save_model(self, path: str):
        """Guarda el modelo."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }, path)

    def load_model(self, path: str):
        """Carga el modelo."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


def prepare_data_loaders(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 32,
    val_split: float = 0.2,
    test_split: float = 0.0,
    random_state: int = 42
) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
    """
    Prepara DataLoaders para entrenamiento.

    Args:
        X: Features
        y: Labels
        batch_size: Tamaño de batch
        val_split: Proporción de validación
        test_split: Proporción de test
        random_state: Semilla

    Returns:
        Tuple de (train_loader, val_loader, test_loader)
    """
    torch.manual_seed(random_state)

    # Convertir a tensores
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y)

    dataset = TensorDataset(X_tensor, y_tensor)

    # Split
    n_samples = len(dataset)
    n_test = int(n_samples * test_split)
    n_val = int(n_samples * val_split)
    n_train = n_samples - n_val - n_test

    if n_test > 0:
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [n_train, n_val, n_test]
        )
    else:
        train_dataset, val_dataset = random_split(dataset, [n_train, n_val])
        test_dataset = None

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size) if n_val > 0 else None
    test_loader = DataLoader(test_dataset, batch_size=batch_size) if test_dataset else None

    return train_loader, val_loader, test_loader


def prepare_sustainability_data(
    df: pd.DataFrame,
    target: str = 'Sustainable',
    exclude_cols: List[str] = None
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Prepara datos de sostenibilidad para Deep Learning.

    Args:
        df: DataFrame con datos
        target: Columna objetivo
        exclude_cols: Columnas a excluir

    Returns:
        Tuple de (X, y, feature_names)
    """
    exclude = exclude_cols or []
    exclude.append(target)

    # Separar features y target
    feature_cols = [c for c in df.columns if c not in exclude]
    X = df[feature_cols].copy()
    y = df[target].values

    # Encodear variables categóricas
    for col in X.select_dtypes(include=['object', 'category']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    # Escalar
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, feature_cols


def train_model(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    config: Optional[TrainingConfig] = None
) -> Tuple[nn.Module, TrainingHistory]:
    """
    Función de conveniencia para entrenar un modelo.

    Args:
        model: Modelo a entrenar
        X: Features
        y: Labels
        config: Configuración de entrenamiento

    Returns:
        Tuple de (modelo entrenado, historial)
    """
    config = config or TrainingConfig()

    train_loader, val_loader, _ = prepare_data_loaders(
        X, y,
        batch_size=config.batch_size,
        val_split=config.val_split
    )

    trainer = Trainer(model, config)
    history = trainer.fit(train_loader, val_loader)

    return model, history


def evaluate_model(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    device: str = 'cpu'
) -> Dict:
    """
    Evalúa un modelo.

    Args:
        model: Modelo entrenado
        X: Features
        y: Labels
        device: Dispositivo

    Returns:
        Diccionario con métricas
    """
    model.eval()
    model = model.to(device)

    X_tensor = torch.FloatTensor(X).to(device)
    y_tensor = torch.LongTensor(y).to(device)

    with torch.no_grad():
        outputs = model(X_tensor)
        if isinstance(outputs, tuple):
            logits, _ = outputs
        else:
            logits = outputs

        probs = torch.sigmoid(logits.squeeze())
        preds = (probs >= 0.5).float()

    y_true = y_tensor.cpu().numpy()
    y_pred = preds.cpu().numpy()
    y_prob = probs.cpu().numpy()

    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'auc_roc': roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5,
        'confusion_matrix': confusion_matrix(y_true, y_pred)
    }
