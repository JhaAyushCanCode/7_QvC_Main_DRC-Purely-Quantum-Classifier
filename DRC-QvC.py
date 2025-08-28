import pennylane as qml
from pennylane import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import time
import warnings
from typing import List, Tuple, Dict, Any
import logging
import os
import ast  # For parsing string representations of lists

# Import transformers for BERT
from transformers import AutoTokenizer, AutoModel

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Set double precision for PennyLane stability
torch.set_default_dtype(torch.float64)

# Configuration
N_QUBITS = 12
N_LAYERS = 3
N_LABELS = 28
BATCH_SIZE = 16  # Reduced for memory efficiency
EPOCHS = 100
LEARNING_RATE = 0.01
PATIENCE = 15  # Early stopping patience
MIN_DELTA = 1e-4  # Minimum improvement for early stopping

# Dataset paths - UPDATE THESE PATHS TO MATCH YOUR DIRECTORY
DATA_DIR = r"C:\Users\Admin\.spyder-py3\QvC-3_docs"
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
VAL_PATH   = os.path.join(DATA_DIR, "val.csv")
TEST_PATH  = os.path.join(DATA_DIR, "test.csv")

# Use GPU for classical components, CPU for quantum operations
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
QUANTUM_DEVICE = torch.device("cpu")  # Keep quantum operations on CPU
logger.info(f"Using device: {DEVICE} for classical operations, CPU for quantum operations")

# Ensure reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# GoEmotions label names for interpretability
GOEMOTIONS_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral"
]

def parse_labels(label_str):
    """Parse string representation of labels list"""
    if isinstance(label_str, str):
        try:
            return ast.literal_eval(label_str)
        except:
            # Fallback parsing if ast.literal_eval fails
            return eval(label_str)
    return label_str

def labels_to_multi_hot(labels_list):
    """Convert list of binary labels to multi-hot vector"""
    vec = np.array(labels_list, dtype=np.float32)
    return vec

def compute_pos_weights(y_train: np.ndarray) -> torch.Tensor:
    """Compute per-label positive class weights from training data."""
    pos_counts = np.sum(y_train, axis=0)
    neg_counts = len(y_train) - pos_counts
    
    # Avoid division by zero
    pos_weights = np.where(pos_counts > 0, neg_counts / pos_counts, 1.0)
    
    # Clip extreme weights to avoid instability
    pos_weights = np.clip(pos_weights, 0.1, 10.0)
    
    logger.info(f"Computed pos_weights - min: {pos_weights.min():.3f}, max: {pos_weights.max():.3f}, mean: {pos_weights.mean():.3f}")
    return torch.tensor(pos_weights, dtype=torch.float64, device=torch.device("cpu"))  # Keep on CPU for quantum operations

class DimensionalityReduction(nn.Module):
    """Reduces BERT embeddings to qubit-compatible dimensions."""
    
    def __init__(self, input_dim: int = 768, output_dim: int = N_QUBITS):
        super(DimensionalityReduction, self).__init__()
        # Simplified architecture to reduce overfitting risk
        self.network = nn.Sequential(
            nn.Linear(input_dim, output_dim * 2),
            nn.LayerNorm(output_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim * 2, output_dim),
            nn.Tanh()  # Bounded output for stable quantum encoding
        )
        
        # Xavier initialization for better training
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

# Define quantum device
dev = qml.device("default.qubit", wires=N_QUBITS)

@qml.qnode(dev, interface="torch", diff_method="adjoint")
def quantum_circuit(inputs: torch.Tensor, weights: List[torch.Tensor]) -> List[torch.Tensor]:
    """
    Quantum circuit with data re-uploading and observable bank for multi-label output.
    
    Args:
        inputs: Tensor of shape (N_QUBITS,) with values in [-1, 1]
        weights: List of weight tensors for each layer
    
    Returns:
        List of expectation values for different observables
    """
    # Data re-uploading with parameterized layers
    for layer_idx in range(N_LAYERS):
        # Encode data with angle embedding
        qml.AngleEmbedding(inputs, wires=range(N_QUBITS), rotation='Y')
        
        # Apply parameterized entangling layer
        qml.StronglyEntanglingLayers(weights[layer_idx], wires=range(N_QUBITS))
    
    # Observable bank: Create N_LABELS observables from N_QUBITS qubits
    expectations = []
    
    # Strategy 1: Single qubit Z observables (first N_QUBITS outputs)
    for i in range(min(N_QUBITS, N_LABELS)):
        expectations.append(qml.expval(qml.PauliZ(i)))
    
    # Strategy 2: Two-qubit ZZ observables for next outputs
    remaining = N_LABELS - len(expectations)
    if remaining > 0:
        for i in range(min(remaining, N_QUBITS - 1)):
            q1, q2 = i, (i + 1) % N_QUBITS
            expectations.append(qml.expval(qml.PauliZ(q1) @ qml.PauliZ(q2)))
    
    # Strategy 3: Single qubit X observables if still needed
    remaining = N_LABELS - len(expectations)
    if remaining > 0:
        for i in range(min(remaining, N_QUBITS)):
            expectations.append(qml.expval(qml.PauliX(i)))
    
    return expectations[:N_LABELS]

def init_quantum_weights() -> nn.ParameterList:
    """Initialize quantum circuit weights with small random values to avoid barren plateaus."""
    weights = []
    
    for layer_idx in range(N_LAYERS):
        # Small angle initialization to avoid barren plateaus
        # StronglyEntanglingLayers expects shape: (n_layers, n_wires, 3)
        layer_weights = torch.normal(0, 0.1, size=(1, N_QUBITS, 3), dtype=torch.float64)  # Double precision
        weights.append(nn.Parameter(layer_weights))
    
    return nn.ParameterList(weights)

class PureQuantumModel(nn.Module):
    """Pure quantum model for multi-label classification."""
    
    def __init__(self):
        super(PureQuantumModel, self).__init__()
        self.dim_reduction = DimensionalityReduction()
        self.quantum_weights = init_quantum_weights()
        
        # Direct logit scaling (no double sigmoid)
        self.output_scales = nn.Parameter(torch.ones(N_LABELS, dtype=torch.float64))
        self.output_biases = nn.Parameter(torch.zeros(N_LABELS, dtype=torch.float64))
        
        # Store observable mapping for interpretability
        self.observables_map = self._get_observables_map()
        
        logger.info(f"Model initialized with {self.count_parameters()} parameters")
        logger.info(f"Observable mapping: {self.observables_map}")
    
    def _get_observables_map(self) -> List[Tuple[str, str]]:
        """Get the mapping of observables to emotion labels for interpretability."""
        observables_map = []
        
        # Single qubit Z observables
        for i in range(min(N_QUBITS, N_LABELS)):
            observables_map.append((GOEMOTIONS_LABELS[i], f"Z_{i}"))
        
        # Two-qubit ZZ observables
        remaining = N_LABELS - len(observables_map)
        if remaining > 0:
            for i in range(min(remaining, N_QUBITS - 1)):
                q1, q2 = i, (i + 1) % N_QUBITS
                emotion_idx = len(observables_map)
                if emotion_idx < len(GOEMOTIONS_LABELS):
                    observables_map.append((GOEMOTIONS_LABELS[emotion_idx], f"Z_{q1}Z_{q2}"))
        
        # Single qubit X observables
        remaining = N_LABELS - len(observables_map)
        if remaining > 0:
            for i in range(min(remaining, N_QUBITS)):
                emotion_idx = len(observables_map)
                if emotion_idx < len(GOEMOTIONS_LABELS):
                    observables_map.append((GOEMOTIONS_LABELS[emotion_idx], f"X_{i}"))
        
        return observables_map
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the quantum model.
        
        Args:
            x: Input tensor of shape (batch_size, 768)
        
        Returns:
            Logits tensor of shape (batch_size, N_LABELS)
        """
        batch_size = x.size(0)
        
        # Reduce dimensionality (keep on GPU for speed)
        x_reduced = self.dim_reduction(x)  # Shape: (batch_size, N_QUBITS)
        
        # Move to CPU for quantum processing
        x_reduced_cpu = x_reduced.cpu()
        
        # Process through quantum circuit (sample by sample)
        expectations = torch.zeros(batch_size, N_LABELS, dtype=torch.float64)
        
        for i in range(batch_size):
            try:
                # Get quantum expectations for this sample
                circuit_output = quantum_circuit(x_reduced_cpu[i], self.quantum_weights)
                expectations[i] = torch.stack(circuit_output)
            except Exception as e:
                logger.error(f"Quantum circuit error for sample {i}: {e}")
                # Fallback to zeros if quantum circuit fails
                expectations[i] = torch.zeros(N_LABELS, dtype=torch.float64)
        
        # Move back to GPU and convert directly to logits (no double sigmoid)
        expectations = expectations.to(x.device)
        logits = expectations * self.output_scales.to(x.device) + self.output_biases.to(x.device)
        
        return logits

class EarlyStopping:
    """Early stopping utility class."""
    
    def __init__(self, patience: int = 15, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, val_score: float) -> bool:
        if self.best_score is None:
            self.best_score = val_score
        elif val_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_score
            self.counter = 0
        
        return self.early_stop

def optimize_thresholds(y_true: np.ndarray, y_pred_proba: np.ndarray) -> np.ndarray:
    """Optimize per-label classification thresholds for better F1 score."""
    thresholds = np.zeros(N_LABELS)
    
    for label_idx in range(N_LABELS):
        best_threshold = 0.5
        best_f1 = 0.0
        
        # Grid search for optimal threshold
        for threshold in np.arange(0.1, 0.9, 0.05):
            y_pred_thresh = (y_pred_proba[:, label_idx] > threshold).astype(int)
            f1 = f1_score(y_true[:, label_idx], y_pred_thresh, zero_division=0)
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        thresholds[label_idx] = best_threshold
    
    logger.info(f"Optimized thresholds - min: {thresholds.min():.3f}, max: {thresholds.max():.3f}, mean: {thresholds.mean():.3f}")
    return thresholds

def create_data_loaders(X_train: np.ndarray, y_train: np.ndarray, 
                       X_val: np.ndarray, y_val: np.ndarray) -> Tuple[DataLoader, DataLoader]:
    """Create data loaders with proper memory management."""
    
    # Create datasets but keep on CPU initially
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float64),
        torch.tensor(y_train, dtype=torch.float64)
    )
    
    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float64),
        torch.tensor(y_val, dtype=torch.float64)
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=2 if DEVICE.type == 'cuda' else 0,  # Use workers for GPU
        pin_memory=DEVICE.type == 'cuda',  # Pin memory for GPU transfers
        persistent_workers=DEVICE.type == 'cuda' and True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=2 if DEVICE.type == 'cuda' else 0,
        pin_memory=DEVICE.type == 'cuda',
        persistent_workers=DEVICE.type == 'cuda' and True
    )
    
    return train_loader, val_loader

def train_model(model: PureQuantumModel, train_loader: DataLoader, val_loader: DataLoader,
               pos_weights: torch.Tensor) -> Tuple[List[float], List[float], List[float], List[float], np.ndarray]:
    """Train the quantum model with proper error handling and monitoring."""
    
    # Loss function with computed pos_weights (move to main device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights.to(DEVICE))
    
    # Optimizer with L2 weight decay to prevent overfitting
    optimizer = Adam([
        {'params': model.dim_reduction.parameters(), 'lr': LEARNING_RATE, 'weight_decay': 1e-4},
        {'params': model.quantum_weights, 'lr': LEARNING_RATE * 0.1, 'weight_decay': 1e-5},  # Lower decay for quantum
        {'params': [model.output_scales, model.output_biases], 'lr': LEARNING_RATE * 0.5, 'weight_decay': 1e-4}
    ])
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10, verbose=True
    )
    
    # Early stopping
    early_stopping = EarlyStopping(patience=PATIENCE, min_delta=MIN_DELTA)
    
    # Tracking lists
    train_losses, val_losses = [], []
    train_f1s, val_f1s = [], []
    
    best_val_f1 = 0.0
    best_model_state = None
    
    logger.info("Starting training...")
    
    for epoch in range(EPOCHS):
        # Training phase
        model.train()
        epoch_train_loss = 0.0
        train_proba_list, train_labels_list = [], []
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]", leave=False)
        
        for batch_idx, (batch_X, batch_y) in enumerate(train_pbar):
            try:
                # Move batch to device
                batch_X = batch_X.to(DEVICE)
                batch_y = batch_y.to(DEVICE)
                
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                
                # Check for NaN loss
                if torch.isnan(loss):
                    logger.warning(f"NaN loss detected at epoch {epoch+1}, batch {batch_idx}")
                    continue
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                epoch_train_loss += loss.item()
                
                # Store probabilities for F1 calculation (not thresholded predictions)
                with torch.no_grad():
                    proba = torch.sigmoid(outputs).detach().cpu().numpy().astype(np.float64)
                    train_proba_list.append(proba)
                    train_labels_list.append(batch_y.cpu().numpy())
                
                # Update progress bar
                train_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'LR': f'{optimizer.param_groups[0]["lr"]:.6f}'
                })
                
            except Exception as e:
                logger.error(f"Training error at epoch {epoch+1}, batch {batch_idx}: {e}")
                continue
        
        # Calculate training metrics
        epoch_train_loss /= len(train_loader)
        train_losses.append(epoch_train_loss)
        
        if train_proba_list:
            train_proba = np.vstack(train_proba_list)
            train_labels = np.vstack(train_labels_list)
            # Use standard 0.5 threshold for training F1 (for consistency)
            train_preds = (train_proba > 0.5).astype(int)
            train_f1 = f1_score(train_labels, train_preds, average='macro', zero_division=0)
            train_f1s.append(train_f1)
        else:
            train_f1s.append(0.0)
        
        # Validation phase
        model.eval()
        epoch_val_loss = 0.0
        val_proba_list, val_labels_list = [], []
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]", leave=False)
            
            for batch_X, batch_y in val_pbar:
                try:
                    batch_X = batch_X.to(DEVICE)
                    batch_y = batch_y.to(DEVICE)
                    
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    
                    if not torch.isnan(loss):
                        epoch_val_loss += loss.item()
                    
                    # Store probabilities (not thresholded predictions)
                    proba = torch.sigmoid(outputs).detach().cpu().numpy().astype(np.float64)
                    val_proba_list.append(proba)
                    val_labels_list.append(batch_y.cpu().numpy())
                    
                except Exception as e:
                    logger.error(f"Validation error at epoch {epoch+1}: {e}")
                    continue
        
        # Calculate validation metrics with optimized thresholds
        epoch_val_loss /= len(val_loader)
        val_losses.append(epoch_val_loss)
        
        if val_proba_list:
            val_proba = np.vstack(val_proba_list)
            val_labels = np.vstack(val_labels_list)
            
            # Optimize thresholds on validation set
            optimal_thresholds = optimize_thresholds(val_labels, val_proba)
            
            # Apply optimized thresholds
            val_preds_optimal = (val_proba > optimal_thresholds).astype(int)
            val_f1 = f1_score(val_labels, val_preds_optimal, average='macro', zero_division=0)
            val_f1s.append(val_f1)
            
            # Also calculate F1 with standard threshold for comparison
            val_preds_std = (val_proba > 0.5).astype(int)
            val_f1_std = f1_score(val_labels, val_preds_std, average='macro', zero_division=0)
            
            logger.info(f"Validation F1 - Standard (0.5): {val_f1_std:.4f}, Optimized: {val_f1:.4f}")
        else:
            val_f1s.append(0.0)
            optimal_thresholds = np.full(N_LABELS, 0.5)  # Fallback
        
        # Update learning rate
        scheduler.step(val_f1s[-1])
        
        # Save best model
        if val_f1s[-1] > best_val_f1:
            best_val_f1 = val_f1s[-1]
            best_model_state = model.state_dict().copy()
            logger.info(f"New best validation F1: {best_val_f1:.4f}")
        
        # Log epoch results
        logger.info(f"Epoch {epoch+1}/{EPOCHS} | "
                   f"Train Loss: {epoch_train_loss:.4f}, Train F1: {train_f1s[-1]:.4f} | "
                   f"Val Loss: {epoch_val_loss:.4f}, Val F1: {val_f1s[-1]:.4f}")
        
        # Early stopping check
        if early_stopping(val_f1s[-1]):
            logger.info(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    # Load best model
    if best_model_state is not None:
        # Careful loading to maintain device placement
        current_devices = {
            'dim_reduction': next(model.dim_reduction.parameters()).device,
            'quantum_weights': [p.device for p in model.quantum_weights],
            'output_scales': model.output_scales.device,
            'output_biases': model.output_biases.device
        }
        
        model.load_state_dict(best_model_state)
        
        # Restore proper device placement after loading
        model.dim_reduction.to(current_devices['dim_reduction'])
        model.output_scales = model.output_scales.to(current_devices['output_scales'])
        model.output_biases = model.output_biases.to(current_devices['output_biases'])
        
        for i, p in enumerate(model.quantum_weights):
            if p.device.type != 'cpu':
                model.quantum_weights[i] = nn.Parameter(p.detach().to('cpu'), requires_grad=True)
        
        logger.info(f"Loaded best model with validation F1: {best_val_f1:.4f}")
    
    return train_losses, val_losses, train_f1s, val_f1s, optimal_thresholds

def evaluate_on_test(model: PureQuantumModel, X_test: np.ndarray, y_test: np.ndarray, 
                    optimal_thresholds: np.ndarray) -> Dict[str, float]:
    """Evaluate the trained model on test set with frozen thresholds."""
    
    # Create test loader
    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float64),
        torch.tensor(y_test, dtype=torch.float64)
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=0,  # No workers for final evaluation
        pin_memory=False
    )
    
    model.eval()
    test_proba_list, test_labels_list = [], []
    
    logger.info("Evaluating on test set...")
    
    with torch.no_grad():
        for batch_X, batch_y in tqdm(test_loader, desc="Test Evaluation"):
            try:
                batch_X = batch_X.to(DEVICE)
                batch_y = batch_y.to(DEVICE)
                
                outputs = model(batch_X)
                proba = torch.sigmoid(outputs).detach().cpu().numpy().astype(np.float64)
                
                test_proba_list.append(proba)
                test_labels_list.append(batch_y.cpu().numpy())
                
            except Exception as e:
                logger.error(f"Test evaluation error: {e}")
                continue
    
    if not test_proba_list:
        logger.error("No test predictions collected!")
        return {}
    
    # Calculate test metrics
    test_proba = np.vstack(test_proba_list)
    test_labels = np.vstack(test_labels_list)
    
    # Apply optimized thresholds from validation
    test_preds_optimal = (test_proba > optimal_thresholds).astype(int)
    test_preds_std = (test_proba > 0.5).astype(int)
    
    # Calculate different F1 scores
    test_f1_macro_optimal = f1_score(test_labels, test_preds_optimal, average='macro', zero_division=0)
    test_f1_micro_optimal = f1_score(test_labels, test_preds_optimal, average='micro', zero_division=0)
    test_f1_macro_std = f1_score(test_labels, test_preds_std, average='macro', zero_division=0)
    test_f1_micro_std = f1_score(test_labels, test_preds_std, average='micro', zero_division=0)
    
    results = {
        'test_f1_macro_optimal': test_f1_macro_optimal,
        'test_f1_micro_optimal': test_f1_micro_optimal,
        'test_f1_macro_std': test_f1_macro_std,
        'test_f1_micro_std': test_f1_micro_std
    }
    
    logger.info(f"Test Results:")
    logger.info(f"  Macro F1 (optimized thresholds): {test_f1_macro_optimal:.4f}")
    logger.info(f"  Micro F1 (optimized thresholds): {test_f1_micro_optimal:.4f}")
    logger.info(f"  Macro F1 (standard 0.5): {test_f1_macro_std:.4f}")
    logger.info(f"  Micro F1 (standard 0.5): {test_f1_micro_std:.4f}")
    
    return results

def main():
    """Main training function."""
    try:
        # Load prepared datasets from CSV files
        logger.info("Loading prepared datasets from CSV files...")
        
        try:
            df_train = pd.read_csv(TRAIN_PATH)
            df_val   = pd.read_csv(VAL_PATH)
            df_test  = pd.read_csv(TEST_PATH)
            
            # Parse labels from string format
            df_train["labels"] = df_train["labels"].apply(parse_labels)
            df_val["labels"]   = df_val["labels"].apply(parse_labels)
            df_test["labels"]  = df_test["labels"].apply(parse_labels)
            
            # Convert to multi-hot format
            df_train["multi_hot"] = df_train["labels"].apply(labels_to_multi_hot)
            df_val["multi_hot"]   = df_val["labels"].apply(labels_to_multi_hot)
            df_test["multi_hot"]  = df_test["labels"].apply(labels_to_multi_hot)
            
            logger.info(f"Dataset sizes - Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")
            
        except FileNotFoundError as e:
            logger.error(f"Error: Could not find dataset files. Please run prepare_data.py first.")
            logger.error(f"Expected files: {TRAIN_PATH}, {VAL_PATH}, {TEST_PATH}")
            logger.error(f"Error details: {e}")
            return
        except Exception as e:
            logger.error(f"Error loading datasets: {e}")
            return

        # Load BERT encoder
        logger.info("Loading BERT model...")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        bert = AutoModel.from_pretrained("bert-base-uncased").to(DEVICE)

        def bert_embed_batch(texts, batch_size=32):
            """Generate BERT embeddings for a list of texts in batches."""
            all_embeddings = []
            for i in tqdm(range(0, len(texts), batch_size), desc="Computing embeddings"):
                batch_texts = texts[i:i+batch_size]
                encodings = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt", max_length=128)
                encodings = {k: v.to(DEVICE) for k, v in encodings.items()}
                with torch.no_grad():
                    embeddings = bert(**encodings).last_hidden_state[:, 0, :].cpu()  # CLS token
                all_embeddings.append(embeddings)
            return torch.cat(all_embeddings, dim=0)

        # Generate BERT embeddings
        logger.info("Generating BERT embeddings...")
        X_train = bert_embed_batch(df_train["text"].tolist()).numpy()
        X_val = bert_embed_batch(df_val["text"].tolist()).numpy()
        X_test = bert_embed_batch(df_test["text"].tolist()).numpy()
        
        # Convert multi-hot labels to numpy arrays
        y_train = np.stack(df_train["multi_hot"].values)
        y_val = np.stack(df_val["multi_hot"].values)
        y_test = np.stack(df_test["multi_hot"].values)

        logger.info(f"Data loaded - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")      
        
        # Compute pos_weights from training data
        pos_weights = compute_pos_weights(y_train)
        
        # Create data loaders
        train_loader, val_loader = create_data_loaders(X_train, y_train, X_val, y_val)
        
        # Initialize model with proper device placement
        model = PureQuantumModel()
        model.dim_reduction.to(DEVICE)
        
        # Ensure quantum weights stay on CPU
        for i, p in enumerate(model.quantum_weights):
            if p.device.type != 'cpu':
                model.quantum_weights[i] = nn.Parameter(p.detach().to('cpu'), requires_grad=True)
                
        logger.info(f"Model architecture:\n{model}")
        
        # Train model
        train_losses, val_losses, train_f1s, val_f1s, optimal_thresholds = train_model(
            model, train_loader, val_loader, pos_weights
        )
        
        # Save results
        results = pd.DataFrame({
            'epoch': range(1, len(train_losses) + 1),
            'train_loss': train_losses,
            'val_loss': val_losses,
            'train_f1': train_f1s,
            'val_f1': val_f1s
        })
        
        os.makedirs('results', exist_ok=True)
        results.to_csv('results/pure_quantum_results.csv', index=False)
        logger.info("Results saved to results/pure_quantum_results.csv")
        
        # Create plots
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.plot(results['epoch'], results['train_loss'], label='Train Loss', alpha=0.8)
        plt.plot(results['epoch'], results['val_loss'], label='Val Loss', alpha=0.8)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 3, 2)
        plt.plot(results['epoch'], results['train_f1'], label='Train F1', alpha=0.8)
        plt.plot(results['epoch'], results['val_f1'], label='Val F1', alpha=0.8)
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.title('Training and Validation F1 Score')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 3, 3)
        plt.plot(results['train_f1'], results['val_f1'], alpha=0.6, marker='o', markersize=3)
        plt.xlabel('Train F1')
        plt.ylabel('Val F1')
        plt.title('Train vs Validation F1')
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect correlation')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/pure_quantum_training.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Evaluate on test set
        test_results = evaluate_on_test(model, X_test, y_test, optimal_thresholds)
        pd.Series(test_results).to_csv('results/pure_quantum_test_metrics.csv')
        
        # Save model
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': {
                'N_QUBITS': N_QUBITS,
                'N_LAYERS': N_LAYERS,
                'N_LABELS': N_LABELS,
                'LEARNING_RATE': LEARNING_RATE
            },
            'results': results.to_dict('list'),
            'best_val_f1': max(val_f1s) if val_f1s else 0.0,
            'test_results': test_results,
            'observables_map': model.observables_map
        }, 'results/pure_quantum_model.pth')
        
        logger.info(f"Training completed! Best validation F1: {max(val_f1s):.4f}")
        logger.info("Model and results saved in 'results/' directory")
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise

if __name__ == "__main__":
    main()