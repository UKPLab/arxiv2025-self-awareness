import json
import logging
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from aim import Run
from omegaconf import DictConfig
from tqdm import tqdm
from urartu.common.action import ActionDataset
from urartu.common.model import Model
from self_aware.utils.utils import set_random_seeds

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class SampleConstructor(ActionDataset):
    def __init__(self, cfg: DictConfig, aim_run: Run) -> None:
        super().__init__(cfg, aim_run)
        logger.info("Initializing SampleConstructor")
        torch.set_float32_matmul_precision("high")

    def initialize(self):
        logger.info("Setting random seeds and initializing model")
        set_random_seeds(self.action_cfg.seed)
        self.model = Model.get_model(self.task_cfg.model)

        self.raw_dataset = {val: {} for val in self.task_cfg.dataset.entity_types}
        dataset_path = Path(self.task_cfg.dataset.path)

        if not dataset_path.exists() or not dataset_path.is_dir():
            raise LookupError(f"Directory: {dataset_path} is not found")

        logger.info(f"Loading dataset from {dataset_path}")
        dataset_files = [path for path in dataset_path.glob("*.jsonl") if path.stem in self.task_cfg.dataset.entity_types]
        for dataset_file in dataset_files:
            data = []
            with open(dataset_file, "r") as f:
                for line in f:
                    data.append(json.loads(line))
            self.raw_dataset[dataset_file.stem] = data
            logger.info(f"Loaded {len(data)} samples from {dataset_file.stem}")

        self.run_dir = Path(self.cfg.run_dir)
        self.dataset = {val: [] for val in self.task_cfg.dataset.entity_types}
        logger.info(f"Run directory set to {self.run_dir}")

    def _is_date(self, obj):
        if obj and isinstance(obj, str) and len(obj) >= 10:
            return obj[4] == "-" and obj[7] == "-"
        return False

    def template_constructor(self):
        entries_dir = self.run_dir.joinpath(self.task_cfg.dataset.hash)
        os.makedirs(entries_dir, exist_ok=True)
        logger.info(f"Creating templates in {entries_dir}")

        for entity_type in tqdm(self.task_cfg.dataset.entity_types):
            logger.info(f"Processing entity type: {entity_type}")
            initial_count = len(self.dataset[entity_type])
            
            for sample in tqdm(self.raw_dataset[entity_type]):
                entity_constrs = self.task_cfg.dataset.entity_constr.get(entity_type, [])
                for entity_constr in entity_constrs:
                    entity_name = sample.get(entity_constr["entity_name"], {}).get("value")
                    template = entity_constr["template"]
                    relation = entity_constr["relation"]
                    attribute = sample.get(entity_constr["attribute"], {}).get("value")
                    
                    if self._is_date(attribute):
                        try:
                            date_object = datetime.strptime(attribute, "%Y-%m-%dT%H:%M:%SZ")
                            attribute = date_object.strftime("%Y")
                        except ValueError:
                            pass

                    # hard-coded special case for weird city names e.g. Q1016113
                    if entity_type == "city" and sum(char.isdigit() for char in entity_name) > len(entity_name) / 2:
                        continue

                    if not entity_name or not attribute:
                        continue
                    # eliminate duplicates
                    if (entity_name, relation, attribute) in [(s["entity_name"], s["relation"], s["attribute"]) for s in self.dataset[entity_type]]:
                        continue

                    if self.task_cfg.dataset.quotes:
                        template = template.replace("<entity_type>", entity_type).replace("<entity_name>", f"'{entity_name}'")
                    else:
                        template = template.replace("<entity_type>", entity_type).replace("<entity_name>", entity_name)

                    if self.task_cfg.dataset.questionable:
                        if self.task_cfg.dataset.quotes:
                            question = entity_constr["question"].replace("<entity_type>", entity_type).replace("<entity_name>", f"'{entity_name}'")
                        else:
                            question = entity_constr["question"].replace("<entity_type>", entity_type).replace("<entity_name>", entity_name)
                        template = f"{question} {template}"
                    if "prepend_text" in self.task_cfg:
                        template = f"{self.task_cfg.prepend_text} {template}"
                    template_full = f"{template} {attribute}"

                    self.dataset[entity_type].append(
                        {
                            "template": template,
                            "template_full": template_full,
                            "entity_type": entity_type,
                            "entity_name": entity_name,
                            "relation": relation,
                            "attribute": attribute,
                        }
                    )

            final_count = len(self.dataset[entity_type])
            logger.info(f"Created {final_count - initial_count} new templates for {entity_type}")

            with open(entries_dir.joinpath(f"{entity_type}.jsonl"), "w", encoding="utf-8") as f:
                for item in self.dataset[entity_type]:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            logger.info(f"Saved {final_count} templates to {entries_dir.joinpath(f'{entity_type}.jsonl')}")

    def run(self):
        entries_dir = self.run_dir.joinpath(self.task_cfg.dataset.hash, f"{self.task_cfg.dataset.hash}_type")
        os.makedirs(entries_dir, exist_ok=True)
        logger.info(f"Starting model evaluation in {entries_dir}")

        # Add entity type level counts
        counts = {entity_type: {"known": 0, "forgotten": 0, "relations": {}} for entity_type in self.task_cfg.dataset.entity_types}

        for entity_type in tqdm(self.task_cfg.dataset.entity_types, desc="Process entities"):
            logger.info(f"Evaluating entity type: {entity_type}")
            for sample in tqdm(self.dataset[entity_type], desc=f"Process samples of `{entity_type}`"):
                generate_cfg = self.task_cfg.model.get("generate")

                label_tokenized = self.model.tokenizer(sample["attribute"], return_tensors="pt")
                label_seq = label_tokenized["input_ids"].squeeze(0)  # shape: [1, T]
                generate_cfg["max_new_tokens"] = len(label_seq)

                output, logits = self.model.generate(sample["template"], generate_cfg=generate_cfg)
                logits = logits.squeeze(dim=1)  # shape: [T, V]

                _, top_k_indices = torch.topk(logits, self.task_cfg.model.get("k"), dim=-1)  # shape: [T, k], [T, k]
                _, bottom_k_indices = torch.topk(
                    -logits,
                    int(self.task_cfg.model.get("l") * self.model.tokenizer.vocab_size),
                    dim=-1,
                )  # shape: [T, k], [T, k]

                # ratio of label tokens in prediction: number of tokens in label_seq that are present in top_k_indices
                # if the token at label_seq[i] is in k tokens in top_k_indices[i]
                min_length = min(len(label_seq), logits.size(0))
                top_k_ratio = np.sum([1 for i in range(min_length) if label_seq[i] in top_k_indices[i]]) / len(label_seq)
                bottom_k_ratio = np.sum([1 for i in range(min_length) if label_seq[i] in bottom_k_indices[i]]) / len(label_seq)

                relation = sample["relation"]

                # Initialize relation counts if not present
                if relation not in counts[entity_type]["relations"]:
                    counts[entity_type]["relations"][relation] = {"known": 0, "forgotten": 0}

                if top_k_ratio > bottom_k_ratio:
                    sample["type"] = "known"
                    counts[entity_type]["known"] += 1
                    counts[entity_type]["relations"][relation]["known"] += 1
                else:
                    sample["type"] = "forgotten"
                    counts[entity_type]["forgotten"] += 1
                    counts[entity_type]["relations"][relation]["forgotten"] += 1

                sample["top_k_ratio"] = format(top_k_ratio, ".2f")
                sample["bottom_k_ratio"] = format(bottom_k_ratio, ".2f")
                sample["most_prob_prediction"] = output.replace(sample["template"], "")

            with open(entries_dir.joinpath(f"{entity_type}.jsonl"), "w", encoding="utf-8") as f:
                for item in self.dataset[entity_type]:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
            logger.info(f"Entity type {entity_type} results - Known: {counts[entity_type]['known']}, Forgotten: {counts[entity_type]['forgotten']}")

        counts["meta"] = {
            "k": self.task_cfg.model.get("k"),
            "l_ass": self.task_cfg.model.get("l") * self.model.tokenizer.vocab_size,
            "l": self.task_cfg.model.get("l"),
        }
        with open(entries_dir.joinpath(".counts.json"), "w", encoding="utf-8") as f:
            json.dump(counts, f)
        logger.info(f"Saved counts to {entries_dir.joinpath('.counts.json')}")

        self.aim_run["dataset_meta"] = counts
        logger.info("Completed model evaluation and saved results")


def main(cfg: DictConfig, aim_run: Run):
    logger.info("Starting SampleConstructor main process")
    sample_constructor = SampleConstructor(cfg, aim_run)
    sample_constructor.initialize()
    sample_constructor.template_constructor()
    sample_constructor.run()
    logger.info("SampleConstructor process completed successfully")
