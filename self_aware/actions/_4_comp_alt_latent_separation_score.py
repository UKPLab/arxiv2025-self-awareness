import asyncio
import json
import logging
import os
from pathlib import Path
from collections import defaultdict

import plotly.graph_objects as go
import torch
from aim import Figure, Run
from hydra.utils import instantiate
from omegaconf import DictConfig
from sae_lens.toolkit.pretrained_saes_directory import get_pretrained_saes_directory
from tqdm import tqdm
from urartu.common.action import Action
from urartu.common.dataset import Dataset
from urartu.common.device import Device
from self_aware.utils.utils import set_random_seeds, sort_layer_ids, get_loc_from_hook_name



os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class CompareAlternativeLatentSeparationScore(Action):
    def __init__(self, cfg: DictConfig, aim_run: Run) -> None:
        super().__init__(cfg, aim_run)

        torch.set_float32_matmul_precision("high")

    def initialize(self):
        set_random_seeds(self.action_cfg.seed)

        self.model = instantiate(
            self.task_cfg.model.type,
            self.task_cfg.model.name,
            device=Device.get_device(),
            n_devices=torch.cuda.device_count(),
        )
        self.dataset = Dataset.get_dataset(self.task_cfg.dataset)
        self.dataloader = self.dataset.get_dataloader(self.task_cfg.dataset.dataloader, self.model.tokenizer, return_attrs=True)["train"]

        with open(Path(self.task_cfg.dataset.data_files).joinpath(".counts.json"), "r") as f:
            self.dataset_counts = json.load(f)

        self.aim_run["dataset_counts"] = self.dataset_counts
        
        # Initialize the checkpoint paths for each model type
        # self.sparse_prob_checkpoint_dir = Path(self.task_cfg.sparse_prob.ckpt_dir)
        self.regressor_prob_checkpoint_dir = Path(self.task_cfg.regressor_prob.ckpt_dir)
        # self.nn_prob_checkpoint_dir = Path(self.task_cfg.nn_prob.ckpt_dir)
        
        # Check that all checkpoint directories exist
        for dir_path, dir_name in [
            # (self.sparse_prob_checkpoint_dir, "sparse_prob"),
            (self.regressor_prob_checkpoint_dir, "regressor_prob"),
            # (self.nn_prob_checkpoint_dir, "nn_prob")
        ]:
            if not dir_path.exists():
                raise FileNotFoundError(f"{dir_name} checkpoint directory not found: {dir_path}")
        
        # Cache for epoch information by layer
        self.checkpoints_by_layer = {
            # 'sparse': self._map_checkpoint_by_layer(self.sparse_prob_checkpoint_dir),
            'regressor': self._map_checkpoint_by_layer(self.regressor_prob_checkpoint_dir),
            # 'nn': self._map_checkpoint_by_layer(self.nn_prob_checkpoint_dir)
        }
        
        # logger.info(f"Found checkpoints for {len(self.checkpoints_by_layer['sparse'])} sparse prob layers")
        logger.info(f"Found checkpoints for {len(self.checkpoints_by_layer['regressor'])} regressor prob layers")
        # logger.info(f"Found checkpoints for {len(self.checkpoints_by_layer['nn'])} nn prob layers")
        
        # Define the hook types we'll be processing
        self.hook_types = ["sae_acts", "regressor_prob_acts"]
        # self.hook_types = ["sae_acts", "sparse_prob_acts", "regressor_prob_acts", "nn_prob_acts"]

    def _map_checkpoint_by_layer(self, checkpoint_dir):
        """Map checkpoints by layer and sort by epoch."""
        checkpoints_by_layer = defaultdict(list)
        
        # Get all checkpoint files
        checkpoint_files = list(checkpoint_dir.glob("*.pt"))
        if not checkpoint_files:
            logger.warning(f"No checkpoint files found in {checkpoint_dir}")
            return checkpoints_by_layer
            
        for cp_file in checkpoint_files:
            # Parse filename to get epoch and layer
            # Format: checkpoint_EPOCH_LAYER.pt
            filename = cp_file.name
            epoch_num, layer_name = filename.split('_', 1) # '0_blocks.11.hook_resid_pre.pt'
            layer_name = layer_name.rsplit('.',  1)[0]  # Remove .pt extension

            try:
                # Extract epoch number and layer name
                checkpoints_by_layer[layer_name].append((epoch_num, cp_file))
            except (ValueError, IndexError):
                logger.warning(f"Could not parse checkpoint filename: {filename}")
                    
        # Sort checkpoints by epoch (descending order)
        for layer_name in checkpoints_by_layer:
            checkpoints_by_layer[layer_name].sort(key=lambda x: x[0], reverse=True)
            
        return checkpoints_by_layer

    def initialize_sae(self, sae_id):
        self.sae, self.cfg_dict, sparsity = instantiate(
            self.task_cfg.sae.type,
            release=self.task_cfg.sae.release,
            sae_id=sae_id,
            device=str(Device.get_device()),
        )
        self.sae.eval()  # prevents error if we're expecting a dead neuron mask for who grads

    def initialize_probs(self, hook_name, epoch=None):
        """Load probability models for a specific layer hook name."""
        device = Device.get_device()
        
        # Initialize models for this layer
        # self.sparse_prob = self._load_model_for_hook(
        #     self.task_cfg.sparse_prob, 
        #     hook_name, 
        #     self.checkpoints_by_layer['sparse'], 
        #     epoch=epoch, 
        #     device=device
        # )
        
        self.regressor_prob = self._load_model_for_hook(
            self.task_cfg.regressor_prob, 
            hook_name, 
            self.checkpoints_by_layer['regressor'], 
            epoch=epoch, 
            device=device
        )
        
        # self.nn_prob = self._load_model_for_hook(
        #     self.task_cfg.nn_prob, 
        #     hook_name, 
        #     self.checkpoints_by_layer['nn'], 
        #     epoch=epoch, 
        #     device=device
        # )

    def _load_model_for_hook(self, cls, hook_name, checkpoints_by_layer, epoch=None, device=None):
        """Load a single model for a specific hook."""
        if hook_name not in checkpoints_by_layer:
            logger.warning(f"No checkpoints found for layer {hook_name}")
            return None
            
        checkpoints = checkpoints_by_layer[hook_name]
        
        # Find the checkpoint to load
        checkpoint_to_load = None
        if epoch is not None:
            # Find checkpoint matching the requested epoch
            for ep, cp_file in checkpoints:
                if ep == epoch:
                    checkpoint_to_load = cp_file
                    break
            if checkpoint_to_load is None:
                logger.warning(f"No checkpoint found for layer {hook_name} at epoch {epoch}")
                return None
        else:
            # Load the latest epoch
            _, checkpoint_to_load = checkpoints[0]
        
        # Load the model
        checkpoint = torch.load(checkpoint_to_load, map_location=device)
        prob_model = instantiate(cls.type, **checkpoint["init_cfgs"]).to(device)
        prob_model.load_state_dict(checkpoint["model_state_dict"])
        prob_model.eval()
        
        logger.info(f"Loaded {cls.type._target_.split('.')[-1]} model for layer {hook_name} from epoch {checkpoint['epoch']}")
        return prob_model

    async def run(self):
        torch.manual_seed(self.action_cfg.seed)
        torch.cuda.manual_seed(self.action_cfg.seed)
        torch.cuda.manual_seed_all(self.action_cfg.seed)  # For multi-GPU setups, if applicable

        sae_directory = get_pretrained_saes_directory()
        sae_map = sae_directory[self.task_cfg.sae.release].saes_map

        logger.info(f"Initial SAE map contains {len(sae_map)} keys")

        if len(sae_map) == 0:
            raise ValueError(f"SAE map is empty for release: {self.task_cfg.sae.release}. Check your configuration.")

        if self.task_cfg.sae.get("width"):
            filtered_map = {k: v for k, v in sae_map.items() if self.task_cfg.sae.width in k}
            logger.info(f"After width filtering, SAE map contains {len(filtered_map)} keys")

            if len(filtered_map) > 0:
                sae_map = filtered_map
            else:
                logger.warning(f"Width filter '{self.task_cfg.sae.width}' removed all SAE layers! Using original map.")

        # Apply layer skipping if configured
        if getattr(self.task_cfg.sae, "skip_layers", False):
            keys_list = sort_layer_ids(list(sae_map.keys()))  # Sort before skipping
            if len(keys_list) > 0:  # Make sure we have keys before applying skipping
                filtered_map = {k: v for k, v in sae_map.items() if k in keys_list[::2]}
                logger.info(f"After skip_layers filtering, SAE map contains {len(filtered_map)} keys")

                if len(filtered_map) > 0:
                    sae_map = filtered_map
                else:
                    logger.warning("Layer skipping removed all SAE layers! Using original map.")

        # Sort the SAE map keys numerically
        sorted_sae_keys = sort_layer_ids(list(sae_map.keys()))
        logger.info(f"Sorted SAE layers: {sorted_sae_keys}")

        # Map SAE IDs to indices for visualization (use sorted order)
        self.sae_map_to_id = {k: i for i, k in enumerate(sorted_sae_keys)}
        self.aim_run["sae_map"] = self.sae_map_to_id

        # Get all hook names first
        # SAEs are resid_post hooks
        hook_names = [hook_name for hook_name in self.model.hook_dict.keys() if "resid" in hook_name and "post" in hook_name]
        self.aim_run["raw_hook_names"] = hook_names

        sae_layers = {int(key.split('/')[0].split('_')[1]) for key in self.sae_map_to_id.keys()}  # extracts the number after "layer_"
        pruned_hook_names = [name for name in hook_names if int(name.split('.')[1]) in sae_layers]

        self.hook_names = pruned_hook_names
        self.aim_run["hook_names"] = self.hook_names

        run_dir = Path(self.cfg.run_dir).joinpath("content")
        run_dir.mkdir(exist_ok=True)

        # Initialize results structure for multiple hooks
        self.results = {hook_type: {} for hook_type in self.hook_types}

        for sae_id, hook_id in tqdm(zip(sae_map, self.hook_names), desc="Processing layers", leave=True, dynamic_ncols=True):
            self.initialize_sae(sae_id)
            self.initialize_probs(hook_id)

            # Initialize storage for each hook type
            for hook_type in self.hook_types:
                if sae_id not in self.results[hook_type]:
                    self.results[hook_type][sae_id] = {}

            # Initialize storage for latents by hook type
            known_latents = {hook_type: {} for hook_type in self.hook_types}
            forgotten_latents = {hook_type: {} for hook_type in self.hook_types}

            with torch.inference_mode():
                for batch in tqdm(
                    self.dataloader,
                    desc=f"[Layer {hook_id}] Processing batches",
                    leave=False,
                    dynamic_ncols=True,
                ):  
                    # Get features at the entity's last token position for each layer
                    sole_template_tokenized = self.model.tokenizer(batch["template"])["input_ids"]
                    if self.task_cfg.cls_token_index == "entity_last":
                        sole_entities_type_name_tokenized = self.model.tokenizer([f"{entity_type} {entity_name}" for entity_type, entity_name in zip(batch["entity_type"], batch["entity_name"])])["input_ids"]
                        last_indices_tensor = torch.tensor([len(sole_template) - 1 - list(reversed(sole_template)).index(sole_entity[-1]) for sole_template, sole_entity in zip(sole_template_tokenized, sole_entities_type_name_tokenized)])
                    elif isinstance(self.task_cfg.cls_token_index, int):
                        last_indices_tensor = torch.tensor([len(sole_template) + self.task_cfg.cls_token_index for sole_template in sole_template_tokenized])
                    else:
                        raise NotImplementedError(f"cls_token_index: {self.task_cfg.cls_token_index} is not defined")

                    _, cache = self.model.run_with_cache(batch["input_ids"], prepend_bos=True, names_filter=lambda name: name in set(self.hook_names))

                    cache_tensor = cache[self.sae.cfg.hook_name]
                    if cache_tensor.device != self.sae.b_dec.device:
                        cache_tensor = cache_tensor.to(self.sae.b_dec.device)
                    
                    # SAE processing
                    sae_acts = self.sae.encode(cache_tensor)
                    # Probes processing
                    # sparse_prob_acts = self.sparse_prob.encode(cache_tensor)
                    regressor_prob_acts = self.regressor_prob.encode(cache_tensor)
                    # nn_prob_acts = self.nn_prob.encode(cache_tensor)

                    hook_acts = {
                        "sae_acts": sae_acts,
                        # "sparse_prob_acts": sparse_prob_acts,
                        "regressor_prob_acts": regressor_prob_acts,
                        # "nn_prob_acts": nn_prob_acts,
                    }

                    # Process each hook type separately
                    for hook_type, acts in hook_acts.items():
                        entity_embeddings = acts[torch.arange(acts.shape[0], device=acts.device), last_indices_tensor.to(acts.device), :,]  # [B, E*]

                        for i, sample_type in enumerate(batch["type"]):
                            entity_type = batch["entity_type"][i]
                            if sample_type == "known":
                                if entity_type not in known_latents[hook_type]:
                                    known_latents[hook_type][entity_type] = []
                                known_latents[hook_type][entity_type].append(entity_embeddings[i, :].cpu())
                            elif sample_type == "forgotten":
                                if entity_type not in forgotten_latents[hook_type]:
                                    forgotten_latents[hook_type][entity_type] = []
                                forgotten_latents[hook_type][entity_type].append(entity_embeddings[i, :].cpu())

                    del cache
                    del _
                    torch.cuda.empty_cache()

            # Process results for each hook type
            for hook_type in self.hook_types:
                # Stack latents for each entity type
                known_hook_latents = {entity_type: torch.stack(latents) if latents else torch.tensor([]) for entity_type, latents in known_latents[hook_type].items()}

                forgotten_hook_latents = {entity_type: torch.stack(latents) if latents else torch.tensor([]) for entity_type, latents in forgotten_latents[hook_type].items()}
                
                entity_types = list(known_hook_latents.keys())
                for entity_type in tqdm(entity_types, desc=f"[{hook_type}] Processing {len(entity_types)} entity types", leave=False, dynamic_ncols=True):
                    # Calculate metrics for this entity type and hook
                    numerator = (known_hook_latents[entity_type] > 0).sum(dim=0)
                    denominator = known_hook_latents[entity_type].shape[0]
                    f_l_known = numerator / denominator  # [E*]

                    numerator = (forgotten_hook_latents[entity_type] > 0).sum(dim=0)
                    denominator = forgotten_hook_latents[entity_type].shape[0]
                    f_l_forgotten = numerator / denominator

                    s_l_known = f_l_known - f_l_forgotten
                    s_l_forgotten = f_l_forgotten - f_l_known

                    # regressor probe has scalare value
                    top_k = 1 if f_l_known.shape == torch.Size([1]) else self.task_cfg.top_k
                    top_k_f_l_known_values, top_k_f_l_known_indices = torch.topk(f_l_known, k=top_k)
                    top_k_f_l_forgotten_values, top_k_f_l_forgotten_indices = torch.topk(f_l_forgotten, k=top_k)
                    top_k_s_l_known_values, top_k_s_l_known_indices = torch.topk(s_l_known, k=top_k)
                    top_k_s_l_forgotten_values, top_k_s_l_forgotten_indices = torch.topk(s_l_forgotten, k=top_k)

                    self.results[hook_type][sae_id][entity_type] = {
                        "f_l_known": f_l_known,
                        "f_l_forgotten": f_l_forgotten,
                        "s_l_known": s_l_known,
                        "s_l_forgotten": s_l_forgotten,
                        "top_k": {
                            "top_k_f_l_known": {
                                "values": top_k_f_l_known_values.tolist(),
                                "indices": top_k_f_l_known_indices.tolist(),
                            },
                            "top_k_f_l_forgotten": {
                                "values": top_k_f_l_forgotten_values.tolist(),
                                "indices": top_k_f_l_forgotten_indices.tolist(),
                            },
                            "top_k_s_l_known": {
                                "values": top_k_s_l_known_values.tolist(),
                                "indices": top_k_s_l_known_indices.tolist(),
                            },
                            "top_k_s_l_forgotten": {
                                "values": top_k_s_l_forgotten_values.tolist(),
                                "indices": top_k_s_l_forgotten_indices.tolist(),
                            },
                        },
                    }

                    # Track metrics for each hook type
                    non_zero_perc = torch.nonzero(s_l_known).shape[0] / s_l_known.shape[0]
                    std_value = torch.std(s_l_known).item()
                    avg_abs_value = torch.mean(torch.abs(s_l_known)).item()

                    self.aim_run.track(
                        name=f"s_l_non_zero_perc",
                        value=non_zero_perc,
                        context={"sample_type": "known", "entity_type": entity_type, "hook_type": hook_type},
                        step=self.sae_map_to_id[sae_id],
                    )
                    self.aim_run.track(
                        name=f"s_l_std_value",
                        value=std_value,
                        context={"sample_type": "known", "entity_type": entity_type, "hook_type": hook_type},
                        step=self.sae_map_to_id[sae_id],
                    )
                    self.aim_run.track(
                        name=f"s_l_avg_abs_value",
                        value=avg_abs_value,
                        context={"sample_type": "known", "entity_type": entity_type, "hook_type": hook_type},
                        step=self.sae_map_to_id[sae_id],
                    )

                    non_zero_perc = torch.nonzero(s_l_forgotten).shape[0] / s_l_forgotten.shape[0]
                    std_value = torch.std(s_l_forgotten).item()
                    avg_abs_value = torch.mean(torch.abs(s_l_forgotten)).item()

                    self.aim_run.track(
                        name=f"s_l_non_zero_perc",
                        value=non_zero_perc,
                        context={"sample_type": "forgotten", "entity_type": entity_type, "hook_type": hook_type},
                        step=self.sae_map_to_id[sae_id],
                    )
                    self.aim_run.track(
                        name=f"s_l_std_value",
                        value=std_value,
                        context={"sample_type": "forgotten", "entity_type": entity_type, "hook_type": hook_type},
                        step=self.sae_map_to_id[sae_id],
                    )
                    self.aim_run.track(
                        name=f"s_l_avg_abs_value",
                        value=avg_abs_value,
                        context={"sample_type": "forgotten", "entity_type": entity_type, "hook_type": hook_type},
                        step=self.sae_map_to_id[sae_id],
                    )

            # Plot results after processing each layer
            self.plot()

    def plot(self):
        run_dir = Path(self.cfg.run_dir).joinpath("content")
        run_dir.mkdir(exist_ok=True)
        
        # Define colors for each entity type
        colors = {
            "player": "#A7BCD9",
            "song": "#838383",
            "city": "#B5D2A4",
            "movie": "#BB9A93",
        }

        # Create separate plots for each hook type
        for hook_type in self.hook_types:
            data = self.results[hook_type]
            
            # Skip if no data for this hook type
            if not data:
                logger.warning(f"No data available for hook type: {hook_type}")
                continue
                
            # Separate plots for known and forgotten
            for category in tqdm(["known", "forgotten"], desc=f"[{hook_type}] Creating plots for categories", dynamic_ncols=True):
                fig = go.Figure()
                maxmin_values = []  # To store MaxMin values for each SAE ID
                sae_ids = []

                # Get first non-empty entity data
                first_sae_id = next(iter(data))
                if not data[first_sae_id]:
                    logger.warning(f"No entity data available for hook type: {hook_type}")
                    continue
                    
                for entity_type in tqdm(data[first_sae_id], desc=f"[{hook_type}:{category}] Processing entity types", leave=False, dynamic_ncols=True):
                    mean_values = []
                    std_devs = []

                    for sae_id in tqdm(data, desc=f"[{hook_type}:{category}:{entity_type}] Gathering layer data", leave=False, dynamic_ncols=True):
                        if entity_type not in data[sae_id]:
                            continue
                            
                        top_k = 1 if data[sae_id][entity_type][f"s_l_{category}"].shape == torch.Size([1]) else self.task_cfg.top_k
                        vals, _ = torch.topk(
                            data[sae_id][entity_type][f"s_l_{category}"],
                            top_k,
                        )
                        mean_values.append(vals.mean().item())
                        std_devs.append(vals.std().item())
                        # sae_ids.append(sae_id)
                        sae_ids.append(sae_id.split("/")[0].split("_")[-1])

                    fig.add_trace(
                        go.Scatter(
                            x=sae_ids,
                            y=mean_values,
                            mode="lines+markers",
                            name=f"{entity_type}",
                            marker=dict(size=18, opacity=1.0),
                            line=dict(width=5, color=colors.get(entity_type, "#000000")),  # Default to black if not found
                            error_y=dict(
                                type="data",
                                array=std_devs,
                                visible=True,
                                thickness=2,
                            ),
                        )
                    )

                # Calculate MaxMin for each SAE ID
                for sae_id in tqdm(data, desc=f"[{hook_type}:{category}] Calculating MaxMin values", leave=False, dynamic_ncols=True):
                    if not data[sae_id]:
                        continue
                    maxmin_value = min(torch.max(data[sae_id][entity_type][f"s_l_{category}"]).item() for entity_type in data[sae_id])
                    maxmin_values.append(maxmin_value)

                # Add MaxMin line to the plot
                fig.add_trace(
                    go.Scatter(
                        x=sae_ids,
                        y=maxmin_values,
                        mode="lines",
                        name="MaxMin",
                        line=dict(width=5, color="#FF9998", dash="dash"),  # Red dashed line
                    )
                )

                fig.update_layout(
                    title=dict(
                        text=f"Top 5 {category.capitalize()} Separation Scores Latents",
                        x=0.5,
                        font=dict(size=36),  # larger title
                    ),
                    xaxis_title="Layer",
                    yaxis_title="Score",
                    xaxis=dict(title_font=dict(size=30), tickfont=dict(size=25)),
                    yaxis=dict(title_font=dict(size=30), tickfont=dict(size=25)),
                    legend=dict(font=dict(size=26)),
                    template="plotly_white",
                    width=1100,
                    height=700,
                    plot_bgcolor="white",
                    paper_bgcolor="white",
                )

                self.aim_run.track(
                    name=f"top_k_latent_separation_scores",
                    value=Figure(fig),
                    context={"sample_type": category, "hook_type": hook_type},
                )
                
                # Save the plot as both PNG and PDF formats for better quality options
                fig_name = f"{self.action_cfg.seed}_{hook_type}_{category}_latent_separation_score_{self.task_cfg.model.name.split('/')[-1]}"
                logger.info(f"Saving plot: {fig_name}")
                fig.write_image(
                    run_dir.joinpath(f"{fig_name}.png"),
                    scale=2,
                )
                fig.write_image(
                    run_dir.joinpath(f"{fig_name}.pdf"),
                )


def main(cfg: DictConfig, aim_run: Run):
    latent_separation_score = CompareAlternativeLatentSeparationScore(cfg, aim_run)
    latent_separation_score.initialize()
    asyncio.run(latent_separation_score.run())
