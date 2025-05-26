import asyncio
import json
import logging
import os
from pathlib import Path

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
from self_aware.utils import set_random_seeds, sort_layer_ids


os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class LatentSeparationScore(Action):
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

    def initialize_sae(self, sae_id):
        self.sae, self.cfg_dict, sparsity = instantiate(
            self.task_cfg.sae.type,
            release=self.task_cfg.sae.release,
            sae_id=sae_id,
            device=str(Device.get_device()),
        )
        self.sae.eval()  # prevents error if we're expecting a dead neuron mask for who grads

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

        run_dir = Path(self.cfg.run_dir).joinpath("content")
        run_dir.mkdir(exist_ok=True)

        self.results = {}

        for sae_id in tqdm(sae_map, desc="Over layers", leave=True, dynamic_ncols=True):
            self.initialize_sae(sae_id)
            self.results[sae_id] = {}

            known_latents = {}
            forgotten_latents = {}

            with torch.inference_mode():

                for batch in tqdm(
                    self.dataloader,
                    desc=f"Processing batches for SAE {sae_id}",
                    leave=False,
                    dynamic_ncols=True,
                ):
                    _, cache = self.model.run_with_cache(batch["input_ids"], prepend_bos=True, names_filter=lambda name: name in set(self.sae_map_to_id.keys()))

                    cache_tensor = cache[self.sae.cfg.hook_name]
                    if cache_tensor.device != self.sae.b_dec.device:
                        cache_tensor = cache_tensor.to(self.sae.b_dec.device)
                    sae_acts = self.sae.encode(cache_tensor)
                    # sae_out = self.sae.decode(sae_acts)

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
                        last_indices_tensor = torch.tensor(
                            [len(sole_template) + self.task_cfg.cls_token_index for sole_template in sole_template_tokenized]
                        )
                    else:
                        raise NotImplementedError(f"cls_token_index: {self.task_cfg.cls_token_index} is not defined")

                    # Create a tensor of batch indices.
                    # Select for each sample the activation at its corresponding last index.
                    entity_embeddings = sae_acts[
                        torch.arange(sae_acts.shape[0], device=sae_acts.device),
                        last_indices_tensor.to(sae_acts.device),
                        :,
                    ]  # [B, E*]

                    for i, sample_type in enumerate(batch["type"]):
                        entity_type = batch["entity_type"][i]
                        if sample_type == "known":
                            if entity_type not in known_latents:
                                known_latents[entity_type] = []
                            # known_latents[entity_type].append(ch[i].cpu())
                            known_latents[entity_type].append(entity_embeddings[i, :].cpu())
                        elif sample_type == "forgotten":
                            if entity_type not in forgotten_latents:
                                forgotten_latents[entity_type] = []
                            # forgotten_latents[entity_type].append(ch[i].cpu())
                            forgotten_latents[entity_type].append(entity_embeddings[i, :].cpu())

                    del cache
                    del _
                    torch.cuda.empty_cache()

            known_latents = {entity_type: torch.stack(latents) if latents else torch.tensor([]) for entity_type, latents in known_latents.items()}

            forgotten_latents = {
                entity_type: torch.stack(latents) if latents else torch.tensor([]) for entity_type, latents in forgotten_latents.items()
            }

            # f_l_known = {} # [E*] # contains the number of non-zero values of each activation
            # f_l_forgotten = {} # [E*] # contains the number of non-zero values of each activation

            entity_types = list(known_latents.keys())
            for entity_type in tqdm(entity_types, desc=f"Entity types for SAE {sae_id}", leave=False, dynamic_ncols=True):
                # known_latents[entity_type] [B, E*] where B is varying number of elements in that category
                numerator = (known_latents[entity_type] > 0).sum(dim=0)
                denominator = known_latents[entity_type].shape[0]
                f_l_known = numerator / denominator  # [E*]

                numerator = (forgotten_latents[entity_type] > 0).sum(dim=0)
                denominator = forgotten_latents[entity_type].shape[0]
                f_l_forgotten = numerator / denominator

                s_l_known = f_l_known - f_l_forgotten
                s_l_forgotten = f_l_forgotten - f_l_known

                top_k_f_l_known_values, top_k_f_l_known_indices = torch.topk(f_l_known, k=self.task_cfg.top_k)
                top_k_f_l_forgotten_values, top_k_f_l_forgotten_indices = torch.topk(f_l_forgotten, k=self.task_cfg.top_k)
                top_k_s_l_known_values, top_k_s_l_known_indices = torch.topk(s_l_known, k=self.task_cfg.top_k)
                top_k_s_l_forgotten_values, top_k_s_l_forgotten_indices = torch.topk(s_l_forgotten, k=self.task_cfg.top_k)

                self.results[sae_id][entity_type] = {
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

                non_zero_perc = torch.nonzero(s_l_known).shape[0] / s_l_known.shape[0]
                std_value = torch.std(s_l_known).item()
                avg_abs_value = torch.mean(torch.abs(s_l_known)).item()

                self.aim_run.track(
                    name="s_l_non_zero_perc",
                    value=non_zero_perc,
                    context={"sample_type": "known", "entity_type": entity_type},
                    step=self.sae_map_to_id[sae_id],
                )
                self.aim_run.track(
                    name="s_l_std_value",
                    value=std_value,
                    context={"sample_type": "known", "entity_type": entity_type},
                    step=self.sae_map_to_id[sae_id],
                )
                self.aim_run.track(
                    name="s_l_avg_abs_value",
                    value=avg_abs_value,
                    context={"sample_type": "known", "entity_type": entity_type},
                    step=self.sae_map_to_id[sae_id],
                )

                non_zero_perc = torch.nonzero(s_l_forgotten).shape[0] / s_l_forgotten.shape[0]
                std_value = torch.std(s_l_forgotten).item()
                avg_abs_value = torch.mean(torch.abs(s_l_forgotten)).item()

                self.aim_run.track(
                    name="s_l_non_zero_perc",
                    value=non_zero_perc,
                    context={"sample_type": "forgotten", "entity_type": entity_type},
                    step=self.sae_map_to_id[sae_id],
                )
                self.aim_run.track(
                    name="s_l_std_value",
                    value=std_value,
                    context={"sample_type": "forgotten", "entity_type": entity_type},
                    step=self.sae_map_to_id[sae_id],
                )
                self.aim_run.track(
                    name="s_l_avg_abs_value",
                    value=avg_abs_value,
                    context={"sample_type": "forgotten", "entity_type": entity_type},
                    step=self.sae_map_to_id[sae_id],
                )

            # with open(run_dir.joinpath(f"{self.action_cfg.seed}.pkl"), "wb") as f:
            #     pickle.dump(self.results, f)

            self.plot()

    def plot(self):
        run_dir = Path(self.cfg.run_dir).joinpath("content")
        run_dir.mkdir(exist_ok=True)
        data = self.results
        # Define colors for each entity type
        colors = {
            "player": "#A7BCD9",
            "song": "#838383",
            "city": "#B5D2A4",
            "movie": "#BB9A93",
        }

        # Separate plots for known and forgotten
        for category in tqdm(["known", "forgotten"], desc="Plotting categories", dynamic_ncols=True):
            fig = go.Figure()
            maxmin_values = []  # To store MaxMin values for each SAE ID
            sae_ids = []

            for entity_type in tqdm(next(iter(data.values())), desc=f"Entity types in plot ({category})", leave=False, dynamic_ncols=True):
                mean_values = []
                std_devs = []

                for sae_id in tqdm(data, desc=f"SAE IDs in plot ({entity_type}, {category})", leave=False, dynamic_ncols=True):
                    vals, _ = torch.topk(
                        data[sae_id][entity_type][f"s_l_{category}"],
                        self.task_cfg.top_k,
                    )
                    mean_values.append(vals.mean().item())
                    std_devs.append(vals.std().item())
                    sae_ids.append(sae_id)

                fig.add_trace(
                    go.Scatter(
                        x=sae_ids,
                        y=mean_values,
                        mode="lines+markers",
                        name=f"{entity_type}_s",
                        marker=dict(size=8, opacity=1.0),
                        line=dict(width=3, color=colors.get(entity_type, "#000000")),  # Default to black if not found
                        error_y=dict(
                            type="data",
                            array=std_devs,
                            visible=True,
                            thickness=2,
                        ),
                    )
                )

            # Calculate MaxMin for each SAE ID
            for sae_id in tqdm(data, desc=f"MaxMin for {category}", leave=False, dynamic_ncols=True):
                maxmin_value = min(torch.max(data[sae_id][entity_type][f"s_l_{category}"]).item() for entity_type in data[sae_id])
                maxmin_values.append(maxmin_value)

            # Add MaxMin line to the plot
            fig.add_trace(
                go.Scatter(
                    x=sae_ids,
                    y=maxmin_values,
                    mode="lines",
                    name="MaxMin",
                    line=dict(width=2, color="#FF9998", dash="dash"),  # Red dashed line
                )
            )

            # Get cls_token_index display text
            cls_token_text = "entity_last" if self.task_cfg.cls_token_index == "entity_last" else f"token_{self.task_cfg.cls_token_index}"

            fig.update_layout(
                title=dict(
                    text=f"Mean and Std Dev of Top-K Latent Separation Scores<br>for {category.capitalize()} Across SAE IDs<br><span style='font-size:14px'>Model: {self.task_cfg.model.name} | {self.dataset_counts['meta']['few_shot']}-shot regime | Entity mode: {self.dataset_counts['meta']['few_shot_entity_mode']} | CLS: {cls_token_text}</span>",
                    x=0.5,
                    font=dict(size=20),
                ),
                xaxis_title="SAE IDs",
                yaxis_title="Values",
                template="plotly_white",
                width=1200,
                height=800,
                margin=dict(t=150),  # Increased top margin for the multi-line title
                plot_bgcolor="white",
                paper_bgcolor="white",
            )

            self.aim_run.track(
                name="top_k_latent_separation_scores",
                value=Figure(fig),
                context={"sample_type": category},
            )
            # Save the plot as a high-resolution PNG file
            fig.write_image(
                run_dir.joinpath(f"{self.action_cfg.seed}_{category}_latent_separation_score.png"),
                scale=2,
            )


def main(cfg: DictConfig, aim_run: Run):
    latent_separation_score = LatentSeparationScore(cfg, aim_run)
    latent_separation_score.initialize()
    asyncio.run(latent_separation_score.run())
