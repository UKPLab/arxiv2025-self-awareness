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
import pandas as pd
from torch.cuda.amp import GradScaler, autocast
from torch.nn import functional as F
from tqdm import tqdm
from urartu.common.action import Action
from urartu.common.dataset import Dataset
from urartu.common.device import Device
from self_aware.utils.utils import set_random_seeds, get_required_init_args, get_loc_from_hook_name
from self_aware.utils.trainer import Trainer
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
from collections import defaultdict


# Configure logging and set random seeds for reproducibility
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ProbeTrainer(Action, Trainer):
    def __init__(self, cfg: DictConfig, aim_run: Run) -> None:
        Action.__init__(self, cfg, aim_run)
        Trainer.__init__(self, cfg, aim_run)

        # Add storage for final metrics
        self.final_metrics = {'train': None, 'test': None}

    def initialize(self):
        Trainer.initialize()

        # Initialize neural networks and optimizers for each layer
        self.layer_models, self.layer_optimizers = self._initialize_layer_models()

    def _track_metrics(self, all_metrics, subset, epoch):
        """Track metrics for all layers for the given subset (train/test)."""

        # Store all predictions and probabilities for comparison across layers
        all_layers_preds = {}
        all_layers_probs = {}
        reference_labels = None

        # Save metrics for each layer to track at the end
        last_epoch_metrics = {}

        # Track random baseline for this subset
        random_baseline = self.random_baselines[subset]
        if epoch == self.task_cfg.num_epochs - 1 and self.layer_subsample_factor == 1:
            # Track random baseline as a reference line for layer-indexed metrics
            for step in range(len(self.hook_names)):
                baseline_context = {"subset": subset, "type": "random_baseline"}
                self.aim_run.track(
                    random_baseline,
                    name="accuracy_by_layer",
                    step=step,
                    context=baseline_context,
                )

        # Track random baseline for epoch-based metrics
        baseline_context = {"subset": subset, "type": "random_baseline"}
        self.aim_run.track(random_baseline, name="accuracy", epoch=epoch, context=baseline_context)

        for layer_hook_name in self.hook_names:
            # Get the original hook index for proper visualization
            if self.layer_subsample_factor > 1:
                # Get all hook names to find original index
                all_hook_names = [hook_name for hook_name in self.model.hook_dict.keys() if "resid" in hook_name]
                all_hook_names = [hook_name for hook_name in all_hook_names if "post" in hook_name or "pre" in hook_name]
                original_hook_idx = all_hook_names.index(layer_hook_name)
                hook_idx = self.hook_names.index(layer_hook_name)
            else:
                original_hook_idx = self.hook_names.index(layer_hook_name)
                hook_idx = original_hook_idx

            # Calculate aggregated metrics
            all_preds = np.concatenate(all_metrics[layer_hook_name]["preds"])
            all_probs = np.concatenate(all_metrics[layer_hook_name]["probs"])
            all_labs = np.concatenate(all_metrics[layer_hook_name]["labels"])

            # Store for later comparison
            all_layers_preds[layer_hook_name] = all_preds
            all_layers_probs[layer_hook_name] = all_probs

            if reference_labels is None:
                reference_labels = all_labs

            # Get base context dictionary for aim tracking
            context = {
                "subset": subset,
                "layer_hook_name": layer_hook_name,
                "loc": get_loc_from_hook_name(layer_hook_name),
                "pos": self._get_pos_from_hook_index(original_hook_idx),  # Use original index for position
                "type": "model",  # To distinguish from random baseline
            }

            # Calculate accuracy and ROC AUC
            accuracy = accuracy_score(all_labs, all_preds)
            auc_roc = roc_auc_score(all_labs, all_probs)

            # Calculate mean loss
            loss_values = all_metrics[layer_hook_name]["loss"]
            mean_loss = sum(loss_values) / len(loss_values)

            # Store current metrics for tracking by layer index later
            if epoch >= self.task_cfg.num_epochs - 3:  # Store only the last 3 epochs
                if layer_hook_name not in last_epoch_metrics:
                    last_epoch_metrics[layer_hook_name] = {
                        "accuracy": [],
                        "auc_roc": [],
                        "loss": [],
                        "pb_corr": [],
                        "pb_pval": [],
                    }
                last_epoch_metrics[layer_hook_name]["accuracy"].append(accuracy)
                last_epoch_metrics[layer_hook_name]["auc_roc"].append(auc_roc)
                last_epoch_metrics[layer_hook_name]["loss"].append(mean_loss)

            # Track metrics by epoch as usual
            self.aim_run.track(accuracy, name="accuracy", epoch=epoch, context=context)
            self.aim_run.track(auc_roc, name="auc_roc", epoch=epoch, context=context)
            self.aim_run.track(mean_loss, name="loss", epoch=epoch, context=context)

            # Track metrics by layer index only on the final epoch
            if epoch == self.task_cfg.num_epochs - 1:
                # Use original hook index as step for proper visualization
                step = original_hook_idx

                # Calculate means of the last 3 epochs' metrics
                if layer_hook_name in last_epoch_metrics:
                    mean_accuracy = sum(last_epoch_metrics[layer_hook_name]["accuracy"]) / len(last_epoch_metrics[layer_hook_name]["accuracy"])
                    mean_auc_roc = sum(last_epoch_metrics[layer_hook_name]["auc_roc"]) / len(last_epoch_metrics[layer_hook_name]["auc_roc"])
                    mean_loss_value = sum(last_epoch_metrics[layer_hook_name]["loss"]) / len(last_epoch_metrics[layer_hook_name]["loss"])

                    # Log final layer metrics including point-biserial correlation
                    logger.info(f"Layer {layer_hook_name} final metrics (mean of last 3 epochs):")
                    logger.info(f"  Accuracy: {mean_accuracy:.4f} (random baseline: {random_baseline:.4f})")
                    logger.info(f"  AUC-ROC: {mean_auc_roc:.4f}")
                    logger.info(f"  Loss: {mean_loss_value:.4f}")

                    # Track by layer index
                    self.aim_run.track(
                        mean_accuracy,
                        name="accuracy_by_layer",
                        step=step,
                        context=context,
                    )
                    self.aim_run.track(
                        mean_auc_roc,
                        name="auc_roc_by_layer",
                        step=step,
                        context=context,
                    )
                    self.aim_run.track(
                        mean_loss_value,
                        name="loss_by_layer",
                        step=step,
                        context=context,
                    )

        # Save final metrics to a JSON file
        if epoch == self.task_cfg.num_epochs - 1:
            final_metrics = {
                layer_hook_name: {
                    "accuracy": sum(last_epoch_metrics[layer_hook_name]["accuracy"]) / len(last_epoch_metrics[layer_hook_name]["accuracy"]),
                    "auc_roc": sum(last_epoch_metrics[layer_hook_name]["auc_roc"]) / len(last_epoch_metrics[layer_hook_name]["auc_roc"]),
                    "loss": sum(last_epoch_metrics[layer_hook_name]["loss"]) / len(last_epoch_metrics[layer_hook_name]["loss"]),
                }
                for layer_hook_name in last_epoch_metrics
            }

            # Calculate mean and std across all layers
            mean_std_metrics = {
                "mean_accuracy": np.mean([metrics["accuracy"] for metrics in final_metrics.values()]),
                "std_accuracy": np.std([metrics["accuracy"] for metrics in final_metrics.values()]),
                "mean_auc_roc": np.mean([metrics["auc_roc"] for metrics in final_metrics.values()]),
                "std_auc_roc": np.std([metrics["auc_roc"] for metrics in final_metrics.values()]),
                "mean_loss": np.mean([metrics["loss"] for metrics in final_metrics.values()]),
                "std_loss": np.std([metrics["loss"] for metrics in final_metrics.values()]),
            }

            # Combine final metrics, mean/std metrics, and random baselines
            combined_metrics = {
                "final_metrics": final_metrics,
                "mean_std_metrics": mean_std_metrics,
                "random_baseline": random_baseline,
                "subset": subset,
            }

            # Store metrics for both train and test/validation subsets
            if not hasattr(self, 'all_combined_metrics'):
                self.all_combined_metrics = {}
            self.all_combined_metrics[subset] = combined_metrics

            # Save all combined metrics to a JSON file at the end of the last epoch
            if len(self.all_combined_metrics) == 2:  # Assuming two subsets: train and test/validation
                metrics_path = Path(self.cfg.run_dir) / "final_metrics.json"
                with open(metrics_path, "w") as f:
                    json.dump(self.all_combined_metrics, f, indent=4)
                logger.info(f"Final metrics saved to {metrics_path}")

            # Save metrics for the final epoch
            if epoch == self.task_cfg.num_epochs - 1:
                # Store the metrics for this subset
                self.final_metrics[subset] = {
                    layer_hook_name: {
                        "accuracy": last_epoch_metrics[layer_hook_name]["accuracy"][-1],
                        "auc_roc": last_epoch_metrics[layer_hook_name]["auc_roc"][-1],
                        "loss": last_epoch_metrics[layer_hook_name]["loss"][-1],
                    }
                    for layer_hook_name in last_epoch_metrics
                }

                # Only plot when we have both train and test metrics
                if self.final_metrics['train'] is not None and self.final_metrics['test'] is not None:
                    self.plot_combined_accuracy_per_layer(epoch)

            # Store predictions and labels for the last epoch
            if epoch == self.task_cfg.num_epochs - 1:
                # Get the last layer name (assuming hook_names is ordered)
                last_layer = self.hook_names[-1]

                # Store the predictions and labels
                self.final_layer_outputs = {
                    'layer_name': last_layer,
                    'predictions': np.concatenate(all_metrics[last_layer]["preds"]),
                    'probabilities': np.concatenate(all_metrics[last_layer]["probs"]),
                    'labels': np.concatenate(all_metrics[last_layer]["labels"]),
                    'entity_name': np.concatenate(all_metrics[last_layer]["entity_name"]),
                    'entity_type': np.concatenate(all_metrics[last_layer]["entity_type"]),
                    'relation': np.concatenate(all_metrics[last_layer]["relation"]),
                    'attribute': np.concatenate(all_metrics[last_layer]["attribute"]),
                    'template': np.concatenate(all_metrics[last_layer]["template"]),
                    'template_full': np.concatenate(all_metrics[last_layer]["template_full"]),
                    'correct_prediction': np.concatenate(all_metrics[last_layer]["preds"]) == np.concatenate(all_metrics[last_layer]["labels"]),
                    'subset': subset,
                }

                # Save to a numpy file for later analysis
                output_dir = Path(self.cfg.run_dir) / "final_outputs"
                output_dir.mkdir(parents=True, exist_ok=True)

                np.savez(
                    output_dir / f"final_layer_outputs_{subset}.npz",
                    predictions=self.final_layer_outputs['predictions'],
                    probabilities=self.final_layer_outputs['probabilities'],
                    labels=self.final_layer_outputs['labels'],
                    entity_name=self.final_layer_outputs['entity_name'],
                    entity_type=self.final_layer_outputs['entity_type'],
                    relation=self.final_layer_outputs['relation'],
                    attribute=self.final_layer_outputs['attribute'],
                    template=self.final_layer_outputs['template'],
                    template_full=self.final_layer_outputs['template_full'],
                    correct_prediction=self.final_layer_outputs['correct_prediction'],
                )

                logger.info(f"Saved final layer outputs for {subset} to {output_dir}")

                # Create a dictionary to store predictions per entity
                entity_predictions = defaultdict(lambda: {'correct': False, 'wrong': False})

                # For each prediction, store whether it was correct or wrong for that entity
                for i in range(len(self.final_layer_outputs['entity_name'])):
                    entity = self.final_layer_outputs['entity_name'][i]
                    if self.final_layer_outputs['correct_prediction'][i]:
                        entity_predictions[entity]['correct'] = True
                    else:
                        entity_predictions[entity]['wrong'] = True

                # Count entities that have both correct and wrong predictions
                mixed_prediction_entities = sum(1 for entity in entity_predictions.values() if entity['correct'] and entity['wrong'])

                total_unique_entities = len(entity_predictions)

                logger.info(
                    f"The model errs on at least one attribute and is correct on at least one attribute for {mixed_prediction_entities} out of {total_unique_entities} unique entities."
                )

    def _process_data_split(
        self,
        dataloader,
        layer_models,
        layer_optimizers,
        subset,
        epoch,
        is_training=True,
    ):
        all_metrics = Trainer._process_data_split(dataloader, layer_models, layer_optimizers, subset, epoch, is_training)
        # Track all metrics
        self._track_metrics(all_metrics, subset, epoch)

        # Clear cache at the end of processing
        self.feature_cache = {}
        torch.cuda.empty_cache()

        return all_metrics

    def run(self):
        # Training loop
        self.model.eval()  # Set model to evaluation mode since we're only using it for features

        # Print initial predictions before training to verify diverse starting points
        with torch.no_grad():
            batch = next(iter(self.dataloaders["test"]))
            batch = self._prepare_batch(batch)

            logger.info("Initial predictions before training:")
            for layer_hook_name in self.hook_names:
                model = self.layer_models[layer_hook_name]
                # Ensure model is on the correct device before inference
                model_device = next(model.parameters()).device

                x = self._get_features_at_position(batch, layer_hook_name)
                # Ensure x is on the same device as the model
                if x.device != model_device:
                    x = x.to(model_device)

                # Forward pass
                logits = model(x)
                probs = torch.sigmoid(logits)

                # Log initial prediction statistics
                mean_prob = probs.mean().item()
                std_prob = probs.std().item()
                positive_rate = (probs > 0.5).float().mean().item()

                logger.info(f"Layer {layer_hook_name} - Initial: mean_prob={mean_prob:.4f}, std={std_prob:.4f}, pos_rate={positive_rate:.4f}")

                # Clear memory
                if self.memory_efficient_mode:
                    del x, logits, probs
                    torch.cuda.empty_cache()

        logger.info(f"Will validate every {self.validation_interval} epochs (set to 1 for original behavior)")

        for epoch in range(self.task_cfg.num_epochs):
            self.current_epoch = epoch  # Add this line
            # Training
            self._process_data_split(
                self.dataloaders["train"],
                self.layer_models,
                self.layer_optimizers,
                "train",
                epoch,
                is_training=True,
            )

            # Validation only on specified epochs
            if epoch % self.validation_interval == 0 or epoch == self.task_cfg.num_epochs - 1:
                logger.info(f"Running validation on epoch {epoch+1}")
                with torch.no_grad():
                    self._process_data_split(
                        self.dataloaders["test"],
                        self.layer_models,
                        self.layer_optimizers,
                        "test",
                        epoch,
                        is_training=False,
                    )
            else:
                logger.info(f"Skipping validation on epoch {epoch+1}")

            # Save checkpoints at the end of each epoch
            self.save_checkpoint_per_layer(epoch, self.layer_models)

            # Free memory after each epoch
            gc.collect()
            torch.cuda.empty_cache()

    def plot_combined_accuracy_per_layer(self, epoch):
        """Plot and save accuracy per layer plots for both train and test sets"""
        # Create figure with adjusted size ratio
        plt.figure(figsize=(15, 8))

        # Get layer names and accuracies for both sets
        layers = list(self.final_metrics['train'].keys())
        train_accuracies = [self.final_metrics['train'][layer]["accuracy"] for layer in layers]
        test_accuracies = [self.final_metrics['test'][layer]["accuracy"] for layer in layers]

        # Create x-axis positions
        x_pos = range(len(layers))

        # Set up the plot style
        plt.rcParams['font.family'] = 'Helvetica'
        plt.rcParams['font.size'] = 14

        # Create main axis with proper spacing
        ax = plt.gca()
        # Remove all spines (borders)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

        # Plot train accuracies
        plt.plot(
            x_pos,
            train_accuracies,
            marker='o',
            linestyle='-',
            linewidth=3,
            markersize=10,
            color='#FF7F50',  # Coral
            label='Train Accuracy',
            markerfacecolor='#FF7F50',
            markeredgecolor='#FF7F50',
        )

        # Plot test accuracies
        plt.plot(
            x_pos,
            test_accuracies,
            marker='o',
            linestyle='-',
            linewidth=3,
            markersize=10,
            color='#038080',  # Teal
            label='Test Accuracy',
            markerfacecolor='#038080',
            markeredgecolor='#038080',
        )

        # Plot random baseline
        plt.axhline(y=self.random_baselines['train'], color='red', linestyle='--', linewidth=3, alpha=0.5, label='Random Baseline')

        # Customize plot
        plt.title('Accuracy per Layer', fontsize=28, pad=20)

        plt.xlabel('Layer', fontsize=28)
        plt.ylabel('Accuracy', fontsize=28)

        # Add grid with light gray color and in background
        plt.grid(True, alpha=0.2, color='gray', linestyle='-', which='both')
        ax.set_axisbelow(True)

        # Set axis limits
        plt.ylim(0.5, 0.85)

        # Show only every 5th tick on x-axis
        tick_positions = list(range(0, len(layers), 5))
        tick_labels = [str(i) for i in tick_positions]
        plt.xticks(tick_positions, tick_labels, fontsize=14)
        plt.yticks(fontsize=14)

        # Move legend outside top right
        plt.legend(fontsize=16, frameon=False, bbox_to_anchor=(1.02, 1.0), loc='upper left')  # Position outside top right

        # Adjust layout to prevent label cutoff and accommodate legend
        plt.tight_layout()

        # Save plots
        plots_dir = Path(self.cfg.run_dir) / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Save as PNG with extra space for legend
        png_path = plots_dir / f"accuracy_per_layer_combined_epoch_{epoch}.png"
        plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')

        # Save as PDF
        pdf_path = plots_dir / f"accuracy_per_layer_combined_epoch_{epoch}.pdf"
        plt.savefig(pdf_path, bbox_inches='tight', facecolor='white', edgecolor='none')

        plt.close()

        logger.info(f"Saved combined accuracy plots at epoch {epoch}:")
        logger.info(f"- PNG: {png_path}")
        logger.info(f"- PDF: {pdf_path}")


def main(cfg: DictConfig, aim_run: Run):
    probe_trainer = ProbeTrainer(cfg, aim_run)
    probe_trainer.initialize()
    probe_trainer.run()
