import gc
import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from aim import Run
from hydra.utils import instantiate, get_class
from omegaconf import DictConfig
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.cuda.amp import GradScaler, autocast
from torch.nn import functional as F
from tqdm import tqdm
from urartu.common.action import Action
from urartu.common.dataset import Dataset
from urartu.common.device import Device
from self_aware.utils.utils import set_random_seeds, get_required_init_args, get_loc_from_hook_name
from self_aware.utils.context_perturbator import ContextPertrubator
from datasets import Dataset as HFDataset
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
from collections import defaultdict
from self_aware.utils.context_perturbator import DatasetContext, ContextPerturbatorFactory


# Configure logging and set random seeds for reproducibility
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, cfg: DictConfig, aim_run: Run) -> None:
        # Performance config parameters
        self.use_amp = getattr(self.task_cfg, "use_amp", True)  # Enable AMP by default
        self.validation_interval = getattr(self.task_cfg, "validation_interval", 2)  # Validate every N epochs
        self.batch_features_cache_size = getattr(self.task_cfg, "batch_features_cache_size", 50)  # Maximum batches to cache

        # Memory optimization parameters
        self.memory_efficient_mode = getattr(self.task_cfg, "memory_efficient_mode", False)
        self.process_layers_sequentially = getattr(self.task_cfg, "process_layers_sequentially", False)
        self.gradient_accumulation_steps = getattr(self.task_cfg, "gradient_accumulation_steps", 1)
        self.offload_to_cpu = getattr(self.task_cfg, "offload_to_cpu", False)
        self.layer_subsample_factor = getattr(self.task_cfg, "layer_subsample_factor", 1)

        # Initialize performance-related components
        self.scaler = GradScaler()
        self.feature_cache = {}  # Cache for features

        # Enable high precision matrix multiplications
        torch.set_float32_matmul_precision("high")

        self.probe_cls = get_class(self.task_cfg.probe.type._target_)

    def initialize(self):
        set_random_seeds(self.action_cfg.seed)

        # Initialize model
        self.model = instantiate(
            self.task_cfg.model.type,
            self.task_cfg.model.name,
            device=Device.get_device(),
            n_devices=torch.cuda.device_count(),
        )

        # Get all hook names first
        all_hook_names = [hook_name for hook_name in self.model.hook_dict.keys() if "resid" in hook_name]
        all_hook_names = [hook_name for hook_name in all_hook_names if "post" in hook_name or "pre" in hook_name]

        # Subsample the hooks based on the subsample factor
        if self.layer_subsample_factor > 1:
            self.hook_names = all_hook_names[:: self.layer_subsample_factor]
            logger.info(f"Layer subsampling enabled. Processing {len(self.hook_names)} out of {len(all_hook_names)} layers.")
        else:
            self.hook_names = all_hook_names

        self.aim_run["hook_names"] = self.hook_names
        self.aim_run["total_hook_names"] = len(all_hook_names)
        self.aim_run["processed_hook_names"] = len(self.hook_names)

        # Initialize dataset with pin_memory for faster transfers
        self.dataset = Dataset.get_dataset(self.task_cfg.dataset)

        # Create and apply perturbations if needed
        context = DatasetContext(dataset=self.dataset, task_cfg=self.task_cfg, dataset_counts=getattr(self, "dataset_counts", None))
        perturbator = ContextPerturbatorFactory.create_perturbator(context)
        if perturbator:
            logger.info("Applying context perturbations to dataset")

        self.random_baselines = {
            "train": max(
                Counter(self.dataset.dataset["train"]["type"])["known"],
                Counter(self.dataset.dataset["train"]["type"])["forgotten"],
            )
            / len(self.dataset.dataset["train"]),
            "test": max(
                Counter(self.dataset.dataset["test"]["type"])["known"],
                Counter(self.dataset.dataset["test"]["type"])["forgotten"],
            )
            / len(self.dataset.dataset["test"]),
        }

        self.aim_run["dataset_size"] = {
            "train_size": {
                "total": len(self.dataset.dataset["train"]),
                "known": Counter(self.dataset.dataset["train"]["type"])["known"],
                "forgotten": Counter(self.dataset.dataset["train"]["type"])["forgotten"],
            },
            "test_size": {
                "total": len(self.dataset.dataset["test"]),
                "known": Counter(self.dataset.dataset["test"]["type"])["known"],
                "forgotten": Counter(self.dataset.dataset["test"]["type"])["forgotten"],
            },
        }

        # Load dataset counts if available
        counts_path = Path(self.task_cfg.dataset.data_files).joinpath(".counts.json")
        if counts_path.exists():
            with open(counts_path, "r") as f:
                self.dataset_counts = json.load(f)
                self.aim_run["dataset_counts"] = self.dataset_counts

        self.dataloaders = self.dataset.get_dataloader(
            self.task_cfg.dataset.dataloader,
            self.model.tokenizer,
            return_attrs=True,
        )

        # Initialize parameters
        self.device = Device.get_device()
        self.d_model = self.model.cfg.d_model

        # Store hyperparameters from config
        self.weight_decay = self.task_cfg.weight_decay if hasattr(self.task_cfg, "weight_decay") else 0.0

    def _get_pos_from_hook_index(self, hook_idx):
        return "high" if hook_idx / len(self.hook_names) > 0.5 else "low"

    @property
    def checkpoint_dir(self):
        """Get the checkpoint directory path"""
        return Path(self.cfg.run_dir).joinpath("checkpoint")

    def _prepare_batch(self, batch):
        """Move batch to device with non_blocking for faster transfers"""
        return {k: v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}

    def _get_label(self, batch):
        # top_k_ratio > bottom_k_ratio -> known (0)
        # else -> forgotten (1)
        return (
            1
            - (
                torch.tensor([float(i) for i in batch["top_k_ratio"]], device=self.device)
                > torch.tensor([float(i) for i in batch["bottom_k_ratio"]], device=self.device)
            ).float()
        )

    def _get_token_lengths(self, batch):
        """Get the lengths of input tokens for each example in the batch"""
        return torch.tensor([seq.size(0) for seq in batch["input_ids"]], device=self.device)

    def _get_features_at_position(self, batch, layer_name):
        """Extract features at the specified position from the model"""
        all_features = self._get_all_features(batch)
        features = all_features[layer_name]

        # Move back to device if they were offloaded
        if isinstance(features, torch.Tensor) and features.device != self.device:
            return features.to(self.device)
        return features

    def _process_batch(self, batch, layer_hook_name, probe_model, optimizer, is_training=True, accumulation_step=0):
        """Process a single batch for a specific layer with forward pass, loss computation, and optionally backprop."""
        batch_labels = self._get_label(batch)
        x = self._get_features_at_position(batch, layer_hook_name)

        # Get token lengths for point-biserial correlation calculation
        token_lengths = self._get_token_lengths(batch)

        # Ensure feature tensor is on the same device as the model
        if x.device != next(probe_model.parameters()).device:
            x = x.to(next(probe_model.parameters()).device)

        # Ensure consistent batch size between features and labels (for last batch issues)
        batch_size = min(x.size(0), batch_labels.size(0), token_lengths.size(0))
        if x.size(0) != batch_labels.size(0) or x.size(0) != token_lengths.size(0):
            logger.info(
                f"Last batch size mismatch: features={x.size(0)}, labels={batch_labels.size(0)}, token_lengths={token_lengths.size(0)}. Using {batch_size} samples."
            )
            x = x[:batch_size]
            batch_labels = batch_labels[:batch_size]
            token_lengths = token_lengths[:batch_size]

        # Make sure batch_labels and token_lengths are on the same device as x
        if batch_labels.device != x.device:
            batch_labels = batch_labels.to(x.device)
        if token_lengths.device != x.device:
            token_lengths = token_lengths.to(x.device)

        # Use autocast for mixed precision training if enabled
        with autocast(enabled=self.use_amp):
            # Forward pass through the neural network
            logits = probe_model(x)
            probs = torch.sigmoid(logits)
            # Clip probabilities to avoid numerical instability
            probs = torch.clamp(probs, min=1e-7, max=1 - 1e-7)

            # Compute loss
            if SparseProbe.__name__ == self.probe_cls.__name__:
                loss = self.probe_cls.compute_loss(logits, batch_labels, probe_model)
            else:
                loss = self.probe_cls.compute_loss(logits, batch_labels)

            # Scale loss for gradient accumulation
            if self.gradient_accumulation_steps > 1 and is_training:
                loss = loss / self.gradient_accumulation_steps

        # Backprop if in training mode
        if is_training:
            if self.use_amp:
                # Use mixed precision training
                self.scaler.scale(loss).backward()

                # Only update weights on the last accumulation step
                if (accumulation_step + 1) % self.gradient_accumulation_steps == 0:
                    self.scaler.step(optimizer)
                    self.scaler.update()
                    optimizer.zero_grad(set_to_none=True)
            else:
                # Original behavior
                loss.backward()

                # Only update weights on the last accumulation step
                if (accumulation_step + 1) % self.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

        # Process metrics
        with torch.no_grad():
            batch_probs = probs.detach().cpu().numpy()
            batch_labels_np = batch_labels.cpu().numpy()
            batch_token_lengths = token_lengths.cpu().numpy()

        # Clear some memory
        del x, logits, probs
        if self.memory_efficient_mode:
            torch.cuda.empty_cache()

        # Store batch metadata along with predictions
        return {
            "loss": loss.item() * (self.gradient_accumulation_steps if is_training else 1),
            "probs": batch_probs,
            "preds": (batch_probs > 0.5),
            "labels": batch_labels_np,
            "token_lengths": batch_token_lengths,
            "entity_name": batch["entity_name"],
            "entity_type": batch["entity_type"],
            "relation": batch["relation"],
            "attribute": batch["attribute"],
            "template": batch["template"],
            "template_full": batch["template_full"],
        }

    def _process_data_split(
        self,
        dataloader,
        layer_models,
        layer_optimizers,
        subset,
        epoch,
        is_training=True,
    ):
        """Process a full data split (train/test) across all layers."""
        # Initialize dictionaries to store metrics
        all_metrics = {
            name: {
                "loss": [],
                "preds": [],
                "probs": [],
                "labels": [],
                "token_lengths": [],
                "entity_name": [],
                "entity_type": [],
                "relation": [],
                "attribute": [],
                "template": [],
                "template_full": [],
            }
            for name in self.hook_names
        }

        # Process batches
        desc = f"{'Training' if is_training else 'Validation'} epoch {epoch+1}/{self.task_cfg.num_epochs}"

        # Process layers sequentially to save memory
        if self.process_layers_sequentially:
            for hook_idx, layer_hook_name in enumerate(self.hook_names):
                layer_desc = f"{desc} - Layer {hook_idx+1}/{len(self.hook_names)}"
                logger.info(f"Processing {layer_hook_name}")

                # Move current model to device
                model = layer_models[layer_hook_name]
                if next(model.parameters()).device != self.device:
                    model.to(self.device)

                # Initialize optimizer for this layer
                optimizer = layer_optimizers[layer_hook_name]
                if is_training:
                    optimizer.zero_grad(set_to_none=True)

                for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc=layer_desc):
                    batch = self._prepare_batch(batch)

                    # Process batch with gradient accumulation
                    accumulation_step = batch_idx % self.gradient_accumulation_steps
                    batch_metrics = self._process_batch(
                        batch,
                        layer_hook_name,
                        model,
                        optimizer,
                        is_training,
                        accumulation_step,
                    )

                    # Store metrics for this layer
                    for metric_name, value in batch_metrics.items():
                        all_metrics[layer_hook_name][metric_name].append(value)

                    # Clear cache every batch in memory efficient mode
                    if self.memory_efficient_mode:
                        self.feature_cache = {}
                        torch.cuda.empty_cache()

                # Additional clean-up after each layer
                torch.cuda.empty_cache()

                # Explicitly move model to CPU if in extremely memory-constrained environment
                if self.memory_efficient_mode and hook_idx < len(self.hook_names) - 1:
                    model.cpu()
        else:
            for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc=desc):
                batch = self._prepare_batch(batch)

                # Precompute features for all layers to avoid redundant forward passes
                _ = self._get_all_features(batch)

                # Process each layer independently
                for hook_idx, layer_hook_name in enumerate(self.hook_names):
                    # Process batch with gradient accumulation
                    accumulation_step = batch_idx % self.gradient_accumulation_steps
                    batch_metrics = self._process_batch(
                        batch,
                        layer_hook_name,
                        layer_models[layer_hook_name],
                        layer_optimizers[layer_hook_name],
                        is_training,
                        accumulation_step,
                    )

                    # Store metrics for this layer
                    for metric_name, value in batch_metrics.items():
                        all_metrics[layer_hook_name][metric_name].append(value)

                # Clear the cache every few batches to prevent memory issues
                if batch_idx % 5 == 0 or self.memory_efficient_mode:
                    self.feature_cache = {}
                    torch.cuda.empty_cache()

        return all_metrics

    def _initialize_layer_models(self):
        """Initialize a neural network for each layer with diverse initialization."""
        layer_models = {}
        layer_optimizers = {}

        # Load only the first model to GPU in memory efficient mode
        current_device = self.device

        # Use a different seed for each layer
        for i, layer_hook_name in enumerate(self.hook_names):
            # In memory efficient mode, only first model on GPU initially
            if self.process_layers_sequentially and i > 0:
                current_device = torch.device("cpu")

            layer_seed = self.action_cfg.seed + i * 100
            torch.manual_seed(layer_seed)

            # Initialize model with a bias slightly different for odd vs even layers
            bias_init = 0.1 * (-1 if i % 2 == 0 else 1)

            # Create a neural network for this layer
            required_args = get_required_init_args(self.probe_cls)

            init_kwargs = {
                "input_dim": self.d_model,
                "bias_init": bias_init,
            }
            if "non_linearity" in required_args:
                init_kwargs["non_linearity"] = self.task_cfg.probe.non_linearity

            probe_model = instantiate(self.task_cfg.probe.type, **init_kwargs).to(current_device)

            layer_models[layer_hook_name] = probe_model

            # Create optimizer with layer-specific learning rate
            # Vary learning rate slightly for each layer to encourage different solutions
            layer_lr = self.task_cfg.lr * (1.1 - 0.2 * (i / len(self.hook_names)))

            layer_optimizers[layer_hook_name] = optim.Adam(
                probe_model.parameters(),
                lr=layer_lr,
                weight_decay=self.weight_decay,
            )

        return layer_models, layer_optimizers

    def _get_all_features(self, batch):
        """Extract features for all layers at once to avoid redundant forward passes"""
        batch_id = id(batch)

        # Return cached features if available
        if batch_id in self.feature_cache:
            return self.feature_cache[batch_id]

        features = {}

        # Get features at the entity's last token position for each layer
        sole_template_tokenized = self.model.tokenizer(batch["template"])["input_ids"]
        if self.task_cfg.cls_token_index == "entity_last":
            sole_entities_type_name_tokenized = self.model.tokenizer(
                [f"{entity_type} {entity_name}" for entity_type, entity_name in zip(batch["entity_type"], batch["entity_name"])]
            )["input_ids"]
            last_indices_tensor = torch.tensor(
                [
                    len(sole_template) - 1 - list(reversed(sole_template)).index(sole_entity[-1])
                    for sole_template, sole_entity in zip(sole_template_tokenized, sole_entities_type_name_tokenized)
                ]
            )
        elif isinstance(self.task_cfg.cls_token_index, int):
            last_indices_tensor = torch.tensor([len(sole_template) + self.task_cfg.cls_token_index for sole_template in sole_template_tokenized])
        else:
            raise NotImplementedError(f"cls_token_index: {self.task_cfg.cls_token_index} is not defined")

        try:
            # Try to get features for all layers at once
            with torch.no_grad():
                _, cache = self.model.run_with_cache(batch["input_ids"], prepend_bos=True, names_filter=lambda name: name in set(self.hook_names))

            # Extract features for all layers
            for layer_name in self.hook_names:
                x = cache[layer_name]  # [B, T, E]
                features_tensor = x[torch.arange(x.shape[0], device=x.device), last_indices_tensor.to(x.device)].clone()  # [B, D]

                # Optionally offload features to CPU to save GPU memory
                if self.offload_to_cpu:
                    features[layer_name] = features_tensor.cpu()
                else:
                    features[layer_name] = features_tensor

            # Clean up to free memory
            del cache
            torch.cuda.empty_cache()

        except torch.cuda.OutOfMemoryError as e:
            logger.warning("CUDA out of memory error. Splitting layers into two halves to reduce memory usage.")
            torch.cuda.empty_cache()  # Clear any partial allocations

            # Process in two halves
            mid_point = len(self.hook_names) // 2
            first_half = self.hook_names[:mid_point]
            second_half = self.hook_names[mid_point:]

            # Process first half
            logger.info(f"Processing first half of layers ({len(first_half)} layers)...")
            try:
                with torch.no_grad():
                    _, cache = self.model.run_with_cache(
                        batch["input_ids"], prepend_bos=True, return_type=None, names_filter=lambda name: name in set(first_half)
                    )

                # Extract features for the first half
                for layer_name in first_half:
                    if layer_name in cache:
                        x = cache[layer_name]
                        features_tensor = x[torch.arange(x.shape[0], device="cpu"), last_indices_tensor.to("cpu")].clone()

                        # Optionally offload features to CPU to save GPU memory
                        if self.offload_to_cpu:
                            features[layer_name] = features_tensor.cpu()
                        else:
                            features[layer_name] = features_tensor

                # Clean up to free memory
                del cache
                torch.cuda.empty_cache()

            except RuntimeError as e1:
                logger.error(f"CUDA OOM even with half the layers. Try reducing batch size or model size. Error: {str(e1)}")
                raise

            # Process second half
            logger.info(f"Processing second half of layers ({len(second_half)} layers)...")
            try:
                with torch.no_grad():
                    _, cache = self.model.run_with_cache(
                        batch["input_ids"], prepend_bos=True, return_type=None, names_filter=lambda name: name in set(second_half)
                    )

                # Extract features for the second half
                for layer_name in second_half:
                    if layer_name in cache:
                        x = cache[layer_name]
                        features_tensor = x[torch.arange(x.shape[0], device="cpu"), last_indices_tensor.to("cpu")].clone()

                        # Optionally offload features to CPU to save GPU memory
                        if self.offload_to_cpu:
                            features[layer_name] = features_tensor.cpu()
                        else:
                            features[layer_name] = features_tensor

                # Clean up to free memory
                del cache
                torch.cuda.empty_cache()

            except RuntimeError as e2:
                logger.error(f"CUDA OOM with second half of layers. Try reducing batch size or model size. Error: {str(e2)}")
                raise

        # Cache the features
        self.feature_cache[batch_id] = features

        # Limit cache size to prevent memory issues
        if len(self.feature_cache) > self.batch_features_cache_size:
            # Remove oldest items
            keys_to_remove = list(self.feature_cache.keys())[: -self.batch_features_cache_size]
            for key in keys_to_remove:
                del self.feature_cache[key]

        return features

    def save_checkpoint_per_layer(self, epoch, layer_models, checkpoint_step):
        """Save model checkpoint for each layer"""
        for layer_hook_name, model in layer_models.items():
            if hasattr(model, "non_linearity") and hasattr(model, "hidden_dim"):
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'epoch': epoch,
                    'layer': layer_hook_name,
                    'checkpoint_step': checkpoint_step,
                    "init_cfgs": {
                        "input_dim": model.input_dim,
                        "bias_init": model.bias_init,
                        "non_linearity": model.non_linearity,
                        "hidden_dim": model.hidden_dim,
                    },
                }
            else:
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'epoch': epoch,
                    'layer': layer_hook_name,
                    'checkpoint_step': checkpoint_step,
                    "init_cfgs": {
                        "input_dim": model.input_dim,
                        "bias_init": model.bias_init,
                    },
                }

            save_path = self.checkpoint_dir / f"checkpoint_{checkpoint_step}_{epoch}_{layer_hook_name}.pt"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint, save_path)
