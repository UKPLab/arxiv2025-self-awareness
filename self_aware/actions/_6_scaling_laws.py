import json
import logging
import gc
import pickle
from tqdm import tqdm
from pathlib import Path

import numpy as np
from collections import Counter
import torch
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import roc_auc_score, accuracy_score
from aim import Run
from hydra.utils import instantiate, get_class
from omegaconf import DictConfig
from urartu.common.action import Action
from urartu.common.dataset import Dataset
from urartu.common.device import Device
from self_aware.utils.utils import set_random_seeds, get_required_init_args, get_loc_from_hook_name
from self_aware.utils.trainer import Trainer
from self_aware.actions._5_prob_trainer import SparseProbe
import matplotlib.pyplot as plt


# Configure logging and set random seeds for reproducibility
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ProbScalingBehaviorTrainer(Action, Trainer):
    def __init__(self, cfg: DictConfig, aim_run: Run) -> None:
        Action.__init__(self, cfg, aim_run)
        Trainer.__init__(self, cfg, aim_run)

        # Add storage for combined metrics
        self.all_combined_metrics = {}

    def initialize(self):
        Trainer.initialize()

    def initialize_model_step(self, step):
        # not sure if pythia revision=step0 is trully randomly initialized, thus ->
        if step == -1:
            from transformer_lens import HookedTransformer, HookedTransformerConfig

            self.model = instantiate(
                self.task_cfg.model.type,
                self.task_cfg.model.name,
                device=Device.get_device(),
                n_devices=torch.cuda.device_count(),
                revision=f"step{0}",
            )
            hooked_model_cfg = HookedTransformerConfig(
                n_layers=self.model.cfg.n_layers,
                d_model=self.model.cfg.d_model,
                n_ctx=self.model.cfg.n_ctx,
                d_head=self.model.cfg.d_head,
                act_fn=self.model.cfg.act_fn,
                tokenizer_name=self.task_cfg.model.name,
                device=Device.get_device(),
                n_devices=torch.cuda.device_count(),
            )
            self.model = HookedTransformer(hooked_model_cfg)
        else:
            self.model = instantiate(
                self.task_cfg.model.type,
                self.task_cfg.model.name,
                device=Device.get_device(),
                n_devices=torch.cuda.device_count(),
                revision=f"step{step}",
            )
        # Add these debug lines
        for name, param in self.model.named_parameters():
            if torch.isnan(param).any():
                logger.warning(f"NaN values found in model parameter {name} at step {step}")
                logger.warning(f"Parameter stats - min: {param.min()}, max: {param.max()}, mean: {param.mean()}")

    def _track_metrics(self, all_metrics, subset, epoch, checkpoint_step):
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
        self.aim_run.track(random_baseline, name='accuracy', epoch=epoch, step=checkpoint_step, context=baseline_context)

        for layer_hook_name in self.hook_names:
            # Get the original hook index for proper visualization
            if self.layer_subsample_factor > 1:
                # Get all hook names to find original index
                all_hook_names = [hook_name for hook_name in self.model.hook_dict.keys() if "resid" in hook_name]
                all_hook_names = [hook_name for hook_name in all_hook_names if "post" in hook_name or "pre" in hook_name]
                original_hook_idx = all_hook_names.index(layer_hook_name)
            else:
                original_hook_idx = self.hook_names.index(layer_hook_name)

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

            # Filter out NaN values in predictions and labels for accuracy
            valid_acc_mask = ~np.isnan(all_preds) & ~np.isnan(all_labs)
            if np.any(valid_acc_mask):
                valid_preds = all_preds[valid_acc_mask]
                valid_labs_acc = all_labs[valid_acc_mask]
                accuracy = accuracy_score(valid_labs_acc, valid_preds)
            else:
                accuracy = float('nan')  # Or 0.0, depending on your preference
                logger.warning(f"No valid predictions for accuracy calculation at layer {layer_hook_name}.")

            # Handle NaN values in probabilities before calculating ROC AUC
            valid_mask = ~np.isnan(all_probs)
            if np.any(valid_mask):
                valid_probs = all_probs[valid_mask]
                valid_labs = all_labs[valid_mask]
                if len(valid_probs) > 0 and len(np.unique(valid_labs)) > 1:
                    auc_roc = roc_auc_score(valid_labs, valid_probs)
                else:
                    auc_roc = 0.5  # Default value when we can't calculate ROC AUC
            else:
                auc_roc = 0.5  # Default value when all predictions are NaN

            # Log warning if we had to handle NaN values
            if not np.all(valid_acc_mask):
                logger.warning(
                    f"Found NaN values in predictions/labels for accuracy at layer {layer_hook_name}. "
                    f"Using {np.sum(valid_acc_mask)}/{len(all_preds)} valid entries."
                )
            if not np.all(valid_mask):
                logger.warning(
                    f"Found NaN values in probabilities for ROC AUC at layer {layer_hook_name}. "
                    f"Using {np.sum(valid_mask)}/{len(all_probs)} valid predictions."
                )

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
                    }
                last_epoch_metrics[layer_hook_name]["accuracy"].append(accuracy)
                last_epoch_metrics[layer_hook_name]["auc_roc"].append(auc_roc)
                last_epoch_metrics[layer_hook_name]["loss"].append(mean_loss)

            # Track metrics by epoch as usual
            self.aim_run.track(accuracy, name="accuracy", epoch=epoch, step=checkpoint_step, context=context)
            self.aim_run.track(auc_roc, name="auc_roc", epoch=epoch, step=checkpoint_step, context=context)
            self.aim_run.track(mean_loss, name="loss", epoch=epoch, step=checkpoint_step, context=context)

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
            # Calculate final metrics for this checkpoint step
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
                "checkpoint_step": checkpoint_step,
            }

            # Store metrics for both train and test/validation subsets
            if checkpoint_step not in self.all_combined_metrics:
                self.all_combined_metrics[checkpoint_step] = {}
            self.all_combined_metrics[checkpoint_step][subset] = combined_metrics

    def _process_data_split(self, dataloader, layer_models, layer_optimizers, subset, epoch, checkpoint_step, is_training=True):
        all_metrics = Trainer._process_data_split(dataloader, layer_models, layer_optimizers, subset, epoch, is_training)

        # Track metrics only if this is the last epoch or it's a validation run
        is_last_epoch = epoch == self.task_cfg.num_epochs - 1
        if is_last_epoch or not is_training:
            self._track_metrics(all_metrics, subset, epoch, checkpoint_step)

        # Clear cache at the end of processing
        self.feature_cache = {}
        torch.cuda.empty_cache()

        return all_metrics

    def run(self):
        if self.task_cfg.model.skip_steps:
            steps = list(range(self.task_cfg.model.step_first, self.task_cfg.model.step_last, self.task_cfg.model.step * 5))
        else:
            steps = list(range(self.task_cfg.model.step_first, self.task_cfg.model.step_last, self.task_cfg.model.step))
        # steps = [-1] + steps
        if steps[-1] != self.task_cfg.model.step_last:
            steps.append(self.task_cfg.model.step_last)
        steps_range = steps

        # Initialize dictionaries to store accuracies for each layer at each checkpoint
        layer_accuracies_train = {layer_name: [] for layer_name in self.hook_names}
        layer_accuracies_test = {layer_name: [] for layer_name in self.hook_names}

        for checkpoint_step in tqdm(steps_range, desc="Processing checkpoint steps"):
            # Reset everything for this checkpoint step
            logger.info(f"Starting training for checkpoint step {checkpoint_step}")
            torch.cuda.empty_cache()
            self.feature_cache = {}
            self.scaler = GradScaler()
            self.initialize_model_step(checkpoint_step)
            self.layer_models, self.layer_optimizers = self._initialize_layer_models()

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
                    probs = torch.clamp(probs, min=1e-7, max=1 - 1e-7)

                    # Log initial prediction statistics
                    mean_prob = probs.mean().item()
                    std_prob = probs.std().item()
                    positive_rate = (probs > 0.5).float().mean().item()

                    logger.info(f"Layer {layer_hook_name} - Initial: mean_prob={mean_prob:.4f}, std={std_prob:.4f}, pos_rate={positive_rate:.4f}")

                    # Clear memory
                    if self.memory_efficient_mode:
                        del x, logits, probs
                        torch.cuda.empty_cache()

            # Training for all epochs, only validate on the final one
            for epoch in range(self.task_cfg.num_epochs):
                # Training
                train_metrics = self._process_data_split(
                    self.dataloaders["train"], self.layer_models, self.layer_optimizers, "train", epoch, checkpoint_step, is_training=True
                )

                # Only validate on the final epoch
                if epoch == self.task_cfg.num_epochs - 1:
                    logger.info(f"Running final validation for checkpoint step {checkpoint_step}")
                    with torch.no_grad():
                        test_metrics = self._process_data_split(
                            self.dataloaders["test"], self.layer_models, self.layer_optimizers, "test", epoch, checkpoint_step, is_training=False
                        )

                    # Store accuracies for each layer
                    for layer_name in self.hook_names:
                        train_acc = accuracy_score(
                            np.concatenate(train_metrics[layer_name]["labels"]), np.concatenate(train_metrics[layer_name]["preds"])
                        )
                        test_acc = accuracy_score(
                            np.concatenate(test_metrics[layer_name]["labels"]), np.concatenate(test_metrics[layer_name]["preds"])
                        )

                        layer_accuracies_train[layer_name].append(train_acc)
                        layer_accuracies_test[layer_name].append(test_acc)

                # Save checkpoint only on the final epoch
                if epoch == self.task_cfg.num_epochs - 1:
                    self.save_checkpoint_per_layer(epoch, self.layer_models, checkpoint_step)

                # Free memory after each epoch
                gc.collect()
                torch.cuda.empty_cache()

            # Log completion of this checkpoint step
            logger.info(f"Completed training for checkpoint step {checkpoint_step}")

            # Force cleanup at the end of each checkpoint step
            del self.layer_models
            del self.layer_optimizers
            gc.collect()
            torch.cuda.empty_cache()

        # After processing all checkpoints, create the combined plot
        self.plot_combined_accuracy_per_checkpoint(steps_range, layer_accuracies_train, layer_accuracies_test)

        # After the checkpoint processing loop, save all metrics to JSON
        metrics_path = Path(self.cfg.run_dir) / "final_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(self.all_combined_metrics, f, indent=4)
        logger.info(f"Final metrics saved to {metrics_path}")

    def plot_combined_accuracy_per_checkpoint(self, checkpoint_steps, layer_accuracies_train, layer_accuracies_test):
        """
        Plot and save accuracy per checkpoint plots for both train and test sets,
        with thicker lines/markers and larger axis fonts.
        """
        # Two rows × one column
        fig, (ax_test, ax_train) = plt.subplots(2, 1, figsize=(20, 16), sharex=True)

        # Global style
        plt.rcParams['font.family'] = 'Helvetica'
        plt.rcParams['font.size'] = 25  # base font size for tick labels

        # Color maps
        num_layers = len(layer_accuracies_train)
        train_colors = plt.cm.Oranges(np.linspace(0.3, 0.9, num_layers))
        test_colors = plt.cm.GnBu(np.linspace(0.3, 0.9, num_layers))

        x_pos = range(len(checkpoint_steps))

        # — Top: Test —
        for (layer_name, accs), color in zip(layer_accuracies_test.items(), test_colors):
            ax_test.plot(x_pos, accs, marker='o', linestyle='-', linewidth=3, markersize=7, color=color, alpha=0.7)

        # — Bottom: Train —
        for (layer_name, accs), color in zip(layer_accuracies_train.items(), train_colors):
            ax_train.plot(x_pos, accs, marker='o', linestyle='-', linewidth=3, markersize=7, color=color, alpha=0.7)

        # Customize both axes
        for ax, title, baseline in [
            (ax_test, 'Test Accuracy per Layer', self.random_baselines['test']),
            (ax_train, 'Train Accuracy per Layer', self.random_baselines['train']),
        ]:
            # Remove spines
            for s in ['top', 'right', 'left', 'bottom']:
                ax.spines[s].set_visible(False)
            # Baseline
            ax.axhline(y=baseline, color='red', linestyle='--', linewidth=3, alpha=0.5)
            # Title
            ax.set_title(title, fontsize=34, pad=20)
            # Axis labels
            ax.set_ylabel('Accuracy', fontsize=30)
            # Grid
            ax.grid(True, alpha=0.2, color='gray', linestyle='-', which='both')
            ax.set_axisbelow(True)
            # Tick params
            ax.tick_params(axis='both', labelsize=16)
            # Remove any legend
            if ax.get_legend():
                ax.get_legend().remove()

        # Shared X on bottom
        ax_train.set_xlabel('Training Steps', fontsize=30)
        ax_train.set_xticks(x_pos)
        ax_train.set_xticklabels([str(s) for s in checkpoint_steps], rotation=45)

        # Layout & save
        plt.tight_layout()
        plots_dir = Path(self.cfg.run_dir) / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        png_path = plots_dir / "accuracy_per_checkpoint_all_layers.png"
        pdf_path = plots_dir / "accuracy_per_checkpoint_all_layers.pdf"
        plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
        plt.close()

        logger.info("Saved all-layer accuracy plots:")
        logger.info(f"- PNG: {png_path}")
        logger.info(f"- PDF: {pdf_path}")


def main(cfg: DictConfig, aim_run: Run):
    probe_scaling_behavior_trainer = ProbScalingBehaviorTrainer(cfg, aim_run)
    probe_scaling_behavior_trainer.initialize()
    probe_scaling_behavior_trainer.run()
