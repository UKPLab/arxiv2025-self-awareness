from dataclasses import dataclass
from typing import Dict, Any, Optional
from datasets import Dataset as HFDataset
import pandas as pd
import random
from collections import defaultdict

import logging


logger = logging.getLogger(__name__)


@dataclass
class DatasetContext:
    """Data class to hold the context needed for perturbation"""

    dataset: Any  # The actual dataset
    task_cfg: Dict[str, Any]  # Configuration
    dataset_counts: Optional[Dict] = None  # Optional dataset counts


class ContextPerturbatorFactory:
    @staticmethod
    def create_perturbator(context: DatasetContext) -> Optional['ContextPerturbator']:
        """Factory method to create a perturbator if needed"""
        if any(key in context.task_cfg for key in ["quotation", "questioning", "prepend_text", "few_shot"]):
            return ContextPerturbator(context)
        return None


class ContextPerturbator:
    def __init__(self, context: DatasetContext) -> None:
        self.context = context

        if "quotation" in self.context.task_cfg:
            self.apply_quotation()

        if "questioning" in self.context.task_cfg:
            self.apply_questioning()

        if "prepend_text" in self.context.task_cfg:
            self.apply_random_text()

        if "few_shot" in self.context.task_cfg:
            self.apply_few_shot_templates()

    def apply_quotation(self):
        if not hasattr(self.context.task_cfg, "quotation"):
            logger.info("No apply_quotation specified in task_cfg. Skipping adding quotations.")
            return

        if "test" in self.context.task_cfg.modif_subset:
            logger.info(f"Prepending text to all test samples: '{self.context.task_cfg.quotation}'")
            # Create a copy of the test dataset to modify
            test_data = self.context.dataset.dataset["test"].to_pandas()

            # Prepend the specified text to each template
            for idx in range(len(test_data)):
                if self.context.task_cfg.quotation == "single":
                    test_data.loc[idx, "template"] = test_data.loc[idx, "template"].replace(
                        test_data.loc[idx, "entity_name"], f"'{test_data.loc[idx, 'entity_name']}'"
                    )
                    test_data.loc[idx, "template_full"] = test_data.loc[idx, "template_full"].replace(
                        test_data.loc[idx, "entity_name"], f"'{test_data.loc[idx, 'entity_name']}'"
                    )
                elif self.context.task_cfg.quotation == "double":
                    test_data.loc[idx, "template"] = test_data.loc[idx, "template"].replace(
                        test_data.loc[idx, "entity_name"], f'"{test_data.loc[idx, "entity_name"]}"'
                    )
                    test_data.loc[idx, "template_full"] = test_data.loc[idx, "template_full"].replace(
                        test_data.loc[idx, "entity_name"], f'"{test_data.loc[idx, "entity_name"]}"'
                    )

            self.context.dataset.dataset["test"] = HFDataset.from_pandas(test_data)
            logger.info(f"Successfully prepended text to {len(test_data)} test samples")

        if "train" in self.context.task_cfg.modif_subset:
            logger.info(f"Prepending text to all train samples: '{self.context.task_cfg.quotation}'")
            # Create a copy of the test dataset to modify
            train_data = self.context.dataset.dataset["train"].to_pandas()

            # Prepend the specified text to each template
            for idx in range(len(train_data)):
                train_data.loc[idx, "template"] = f"{self.context.task_cfg.quotation} {train_data.loc[idx, 'template']}"
                train_data.loc[idx, "template_full"] = f"{self.context.task_cfg.quotation} {train_data.loc[idx, 'template_full']}"

            self.context.dataset.dataset["train"] = HFDataset.from_pandas(train_data)
            logger.info(f"Successfully prepended text to {len(train_data)} test samples")

    def apply_questioning(self):
        if not hasattr(self.context.task_cfg, "questioning"):
            logger.info("No questioning template path specified in task_cfg. Skipping adding questioning.")
            return

        import yaml

        if "test" in self.context.task_cfg.modif_subset:
            logger.info(f"Prepending text to all test samples: '{self.context.task_cfg.questioning}'")
            # Create a copy of the test dataset to modify
            test_data = self.context.dataset.dataset["test"].to_pandas()

            with open(self.context.task_cfg.questioning, "r") as f:
                data = yaml.safe_load(f)
            entity_constr = data['entity_constr']

            # Prepend the specified text to each template
            for idx in range(len(test_data)):
                question = next(
                    item["question"]
                    for item in entity_constr[test_data.loc[idx, "entity_type"]]
                    if item["relation"] == test_data.loc[idx, "relation"]
                )
                question = question.replace("<entity_type>", test_data.loc[idx, "entity_type"]).replace(
                    "<entity_name>", test_data.loc[idx, "entity_name"]
                )
                test_data.loc[idx, "template"] = f"{question} {test_data.loc[idx, 'template']}"
                test_data.loc[idx, "template_full"] = f"{question} {test_data.loc[idx, 'template_full']}"

            self.context.dataset.dataset["test"] = HFDataset.from_pandas(test_data)
            logger.info(f"Successfully prepended text to {len(test_data)} test samples")

        if "train" in self.context.task_cfg.modif_subset:
            logger.info(f"Prepending text to all train samples: '{self.context.task_cfg.questioning}'")
            # Create a copy of the test dataset to modify
            train_data = self.context.dataset.dataset["train"].to_pandas()

            # Prepend the specified text to each template
            for idx in range(len(train_data)):
                train_data.loc[idx, "template"] = f"{self.context.task_cfg.questioning} {train_data.loc[idx, 'template']}"
                train_data.loc[idx, "template_full"] = f"{self.context.task_cfg.questioning} {train_data.loc[idx, 'template_full']}"

            self.context.dataset.dataset["train"] = HFDataset.from_pandas(train_data)
            logger.info(f"Successfully prepended text to {len(train_data)} test samples")

    def apply_random_text(self):
        """
        Prepend the specified text from self.context.task_cfg.prepend_text to all samples in the train|/test dataset
        """
        if not hasattr(self.context.task_cfg, "prepend_text"):
            logger.info("No prepend_text specified in task_cfg. Skipping prepending.")
            return

        if "test" in self.context.task_cfg.modif_subset:
            logger.info(f"Prepending text to all test samples: '{self.context.task_cfg.prepend_text}'")
            # Create a copy of the test dataset to modify
            test_data = self.context.dataset.dataset["test"].to_pandas()

            # Prepend the specified text to each template
            for idx in range(len(test_data)):
                test_data.loc[idx, "template"] = f"{self.context.task_cfg.prepend_text} {test_data.loc[idx, 'template']}"
                test_data.loc[idx, "template_full"] = f"{self.context.task_cfg.prepend_text} {test_data.loc[idx, 'template_full']}"

            self.context.dataset.dataset["test"] = HFDataset.from_pandas(test_data)
            logger.info(f"Successfully prepended text to {len(test_data)} test samples")

        if "train" in self.context.task_cfg.modif_subset:
            logger.info(f"Prepending text to all train samples: '{self.context.task_cfg.prepend_text}'")
            # Create a copy of the test dataset to modify
            train_data = self.context.dataset.dataset["train"].to_pandas()

            # Prepend the specified text to each template
            for idx in range(len(train_data)):
                train_data.loc[idx, "template"] = f"{self.context.task_cfg.prepend_text} {train_data.loc[idx, 'template']}"
                train_data.loc[idx, "template_full"] = f"{self.context.task_cfg.prepend_text} {train_data.loc[idx, 'template_full']}"

            self.context.dataset.dataset["train"] = HFDataset.from_pandas(train_data)
            logger.info(f"Successfully prepended text to {len(train_data)} test samples")

    def apply_few_shot_templates(self):
        """
        Apply few-shot templates to the test dataset similar to SampleConstructor's
        few_shot_template_constructor method.
        """
        if not hasattr(self.context.task_cfg, "few_shot") or self.context.task_cfg.few_shot <= 0:
            logger.info("Few-shot is disabled or not configured. Skipping template construction.")
            return

        # Import HFDataset here to ensure it's available
        from datasets import Dataset as HFDataset

        if "test" in self.context.task_cfg.modif_subset:
            logger.info(f"Applying {self.context.task_cfg.few_shot}-shot templates to the test dataset")

            # Get entity_types from the dataset counts if available
            entity_types = []
            if hasattr(self.context, "dataset_counts"):
                entity_types = [et for et in self.context.dataset_counts.keys() if et != "meta"]

            if not entity_types and "entity_type" in self.context.dataset.dataset["test"].column_names:
                # Get unique entity types from the dataset if not available from counts
                entity_types = list(set(self.context.dataset.dataset["test"]["entity_type"]))

            if not entity_types:
                logger.warning("No entity types found. Cannot apply few-shot templates.")
                return

            # Create a temporary copy of the test dataset to work with
            test_dataset = self.context.dataset.dataset["test"].to_pandas()

            # Apply the appropriate few-shot template construction based on the config
            if "few_shot_entity_mode" in self.context.task_cfg and "few_shot_relation_mode" in self.context.task_cfg:
                logger.error("Please provide either `few_shot_entity_mode` or `few_shot_relation_mode`, not both.")
                return

            # Create a new dataset to store the few-shot templates
            few_shot_data = []

            if "few_shot_entity_mode" in self.context.task_cfg:
                if self.context.task_cfg.few_shot_entity_mode == "unique":
                    logger.info("Applying unique entity mode few-shot templates")
                    self._few_shot_entity_unique_template(test_dataset, few_shot_data)
                elif self.context.task_cfg.few_shot_entity_mode == "only":
                    logger.info("Applying entity-only mode few-shot templates")
                    self._few_shot_entity_only_template(test_dataset, few_shot_data)
                else:
                    logger.error(f"Mode `{self.context.task_cfg.few_shot_entity_mode}` for `few_shot_entity_mode` is not implemented.")
                    return
            elif "few_shot_relation_mode" in self.context.task_cfg:
                if self.context.task_cfg.few_shot_relation_mode == "unique":
                    logger.info("Applying unique relation mode few-shot templates")
                    self._few_shot_relation_unique_template(test_dataset, few_shot_data)
                elif self.context.task_cfg.few_shot_relation_mode == "only":
                    logger.info("Applying relation-only mode few-shot templates")
                    self._few_shot_relation_only_template(test_dataset, few_shot_data)
                else:
                    logger.error(f"Mode `{self.context.task_cfg.few_shot_relation_mode}` for `few_shot_relation_mode` is not implemented.")
                    return

            if few_shot_data:
                # Replace the test dataset with the few-shot version
                self.context.dataset.dataset["test"] = HFDataset.from_pandas(pd.DataFrame(few_shot_data))
                logger.info(f"Applied few-shot templates. New test dataset size: {len(self.context.dataset.dataset['test'])}")
            else:
                logger.warning("No few-shot templates were created. Using original test dataset.")

        if "train" in self.context.task_cfg.modif_subset:
            logger.info(f"Applying {self.context.task_cfg.few_shot}-shot templates to the train dataset")

            # Get entity_types from the dataset counts if available
            entity_types = []
            if hasattr(self.context, "dataset_counts"):
                entity_types = [et for et in self.context.dataset_counts.keys() if et != "meta"]

            if not entity_types and "entity_type" in self.context.dataset.dataset["train"].column_names:
                # Get unique entity types from the dataset if not available from counts
                entity_types = list(set(self.context.dataset.dataset["train"]["entity_type"]))

            if not entity_types:
                logger.warning("No entity types found. Cannot apply few-shot templates.")
                return

            # Create a temporary copy of the train dataset to work with
            train_dataset = self.context.dataset.dataset["train"].to_pandas()

            # Apply the appropriate few-shot template construction based on the config
            if "few_shot_entity_mode" in self.context.task_cfg and "few_shot_relation_mode" in self.context.task_cfg:
                logger.error("Please provide either `few_shot_entity_mode` or `few_shot_relation_mode`, not both.")
                return

            # Create a new dataset to store the few-shot templates
            few_shot_data = []

            if "few_shot_entity_mode" in self.context.task_cfg:
                if self.context.task_cfg.few_shot_entity_mode == "unique":
                    logger.info("Applying unique entity mode few-shot templates")
                    self._few_shot_entity_unique_template(train_dataset, few_shot_data)
                elif self.context.task_cfg.few_shot_entity_mode == "only":
                    logger.info("Applying entity-only mode few-shot templates")
                    self._few_shot_entity_only_template(train_dataset, few_shot_data)
                else:
                    logger.error(f"Mode `{self.context.task_cfg.few_shot_entity_mode}` for `few_shot_entity_mode` is not implemented.")
                    return
            elif "few_shot_relation_mode" in self.context.task_cfg:
                if self.context.task_cfg.few_shot_relation_mode == "unique":
                    logger.info("Applying unique relation mode few-shot templates")
                    self._few_shot_relation_unique_template(train_dataset, few_shot_data)
                elif self.context.task_cfg.few_shot_relation_mode == "only":
                    logger.info("Applying relation-only mode few-shot templates")
                    self._few_shot_relation_only_template(train_dataset, few_shot_data)
                else:
                    logger.error(f"Mode `{self.context.task_cfg.few_shot_relation_mode}` for `few_shot_relation_mode` is not implemented.")
                    return

            if few_shot_data:
                # Replace the train dataset with the few-shot version
                self.context.dataset.dataset["train"] = HFDataset.from_pandas(pd.DataFrame(few_shot_data))
                logger.info(f"Applied few-shot templates. New train dataset size: {len(self.context.dataset.dataset['train'])}")
            else:
                logger.warning("No few-shot templates were created. Using original train dataset.")

    def _few_shot_entity_unique_template(self, test_dataset, few_shot_data):
        """
        Apply unique entity few-shot templates.
        No duplicated entity_names in the shot context.
        """
        # Group by entity_type and process each group
        for entity_type, group in test_dataset.groupby("entity_type"):
            # Work with a copy to avoid modifying the original
            remaining_samples = group.to_dict('records')
            random.shuffle(remaining_samples)

            shots = []
            while remaining_samples:
                current_shot = []
                seen_entity_names = set()
                i = 0

                # Get few_shot + 1 samples (context + target)
                while i < len(remaining_samples) and len(current_shot) < self.context.task_cfg.few_shot + 1:
                    sample = remaining_samples[i]
                    if sample["entity_name"] not in seen_entity_names:
                        current_shot.append(sample)
                        seen_entity_names.add(sample["entity_name"])
                        remaining_samples.pop(i)
                    else:
                        i += 1

                if len(current_shot) == self.context.task_cfg.few_shot + 1:  # Only add complete shots
                    shots.append(current_shot)

            # Process each shot to create the few-shot templates
            for shot in shots:
                # Use first few_shot samples as context, last one as target
                context = shot[:-1]
                target = shot[-1]

                template = ". ".join([s["template_full"] for s in context]) + ". " + target["template"]
                template_full = ". ".join([s["template_full"] for s in context]) + ". " + target["template_full"]

                # Create a new row with the few-shot template
                new_row = target.copy()
                new_row["template"] = template
                new_row["template_full"] = template_full
                new_row["context"] = {
                    "templates": [s["template"] for s in context],
                    "template_fulls": [s["template_full"] for s in context],
                    "entity_names": [s["entity_name"] for s in context],
                    "relations": [s["relation"] for s in context],
                    "attributes": [s["attribute"] for s in context],
                }

                few_shot_data.append(new_row)

    def _few_shot_entity_only_template(self, test_dataset, few_shot_data):
        """
        Apply entity-only few-shot templates.
        Group samples by entity_name.
        """
        # Group by entity_type and process each group
        for entity_type, group in test_dataset.groupby("entity_type"):
            # Group samples by entity_name
            entity_groups = defaultdict(list)
            for _, sample in group.iterrows():
                entity_groups[sample["entity_name"]].append(sample.to_dict())

            # Create batches for each entity
            for entity_name, samples in entity_groups.items():
                while len(samples) >= self.context.task_cfg.few_shot + 1:  # Need few_shot samples for context + 1 for prediction
                    batch = []
                    target = None

                    # Select few_shot samples for context
                    for _ in range(self.context.task_cfg.few_shot):
                        idx = random.randrange(len(samples))
                        batch.append(samples.pop(idx))

                    # Select one more sample as the target
                    idx = random.randrange(len(samples))
                    target = samples.pop(idx)

                    template = ". ".join([s["template_full"] for s in batch]) + ". " + target["template"]
                    template_full = ". ".join([s["template_full"] for s in batch]) + ". " + target["template_full"]

                    # Create a new row with the few-shot template
                    new_row = target.copy()
                    new_row["template"] = template
                    new_row["template_full"] = template_full
                    new_row["context"] = {
                        "templates": [s["template"] for s in batch],
                        "template_fulls": [s["template_full"] for s in batch],
                        "entity_names": [s["entity_name"] for s in batch],
                        "relations": [s["relation"] for s in batch],
                        "attributes": [s["attribute"] for s in batch],
                    }

                    few_shot_data.append(new_row)

    def _few_shot_relation_unique_template(self, test_dataset, few_shot_data):
        """
        Apply unique relation few-shot templates.
        No duplicated relations in the shot context.
        """

        # Group by entity_type and process each group
        for entity_type, group in test_dataset.groupby("entity_type"):
            # Work with a copy to avoid modifying the original
            remaining_samples = group.to_dict('records')
            random.shuffle(remaining_samples)

            # Group samples by relation first
            relation_groups = defaultdict(list)
            for sample in remaining_samples:
                relation_groups[sample["relation"]].append(sample)

            shots = []
            while len(relation_groups) >= 2:  # Need at least 2 relations (1 for context, 1 for target)
                current_shot = []
                available_relations = list(relation_groups.keys())

                # If we don't have enough relations for a full shot, break
                if len(available_relations) < min(self.context.task_cfg.few_shot + 1, 2):
                    break

                # Randomly select relations and samples
                random.shuffle(available_relations)
                for relation in available_relations[: self.context.task_cfg.few_shot + 1]:
                    # Get a random sample for this relation
                    samples = relation_groups[relation]
                    idx = random.randrange(len(samples))
                    current_shot.append(samples.pop(idx))

                    # Remove relation group if empty
                    if not samples:
                        del relation_groups[relation]

                    # Break if we have enough samples
                    if len(current_shot) == self.context.task_cfg.few_shot + 1:
                        break

                # Add shot if we have at least 2 samples
                if len(current_shot) >= 2:
                    shots.append(current_shot)

            # Process each shot to create the few-shot templates
            for shot in shots:
                context = shot[:-1]
                target = shot[-1]

                template = ". ".join([s["template_full"] for s in context]) + ". " + target["template"]
                template_full = ". ".join([s["template_full"] for s in context]) + ". " + target["template_full"]

                # Create a new row with the few-shot template
                new_row = target.copy()
                new_row["template"] = template
                new_row["template_full"] = template_full
                new_row["context"] = {
                    "templates": [s["template"] for s in context],
                    "template_fulls": [s["template_full"] for s in context],
                    "entity_names": [s["entity_name"] for s in context],
                    "relations": [s["relation"] for s in context],
                    "attributes": [s["attribute"] for s in context],
                }

                few_shot_data.append(new_row)

    def _few_shot_relation_only_template(self, test_dataset, few_shot_data):
        """
        Apply relation-only few-shot templates.
        Group samples by relation.
        """

        # Group by entity_type and process each group
        for entity_type, group in test_dataset.groupby("entity_type"):
            # Group samples by relation
            relation_groups = defaultdict(list)
            for _, sample in group.iterrows():
                relation_groups[sample["relation"]].append(sample.to_dict())

            # Create batches for each relation that has enough samples
            for relation, samples in relation_groups.items():
                # Only process relations that have at least 2 samples (1 for context, 1 for target)
                while len(samples) >= 2:
                    batch = []
                    target = None

                    # Select samples for context (up to few_shot or available samples - 1)
                    context_size = min(self.context.task_cfg.few_shot, len(samples) - 1)
                    for _ in range(context_size):
                        idx = random.randrange(len(samples))
                        batch.append(samples.pop(idx))

                    # Select one more sample as the target
                    idx = random.randrange(len(samples))
                    target = samples.pop(idx)

                    template = ". ".join([s["template_full"] for s in batch]) + ". " + target["template"]
                    template_full = ". ".join([s["template_full"] for s in batch]) + ". " + target["template_full"]

                    # Create a new row with the few-shot template
                    new_row = target.copy()
                    new_row["template"] = template
                    new_row["template_full"] = template_full
                    new_row["context"] = {
                        "templates": [s["template"] for s in batch],
                        "template_fulls": [s["template_full"] for s in batch],
                        "entity_names": [s["entity_name"] for s in batch],
                        "relations": [s["relation"] for s in batch],
                        "attributes": [s["attribute"] for s in batch],
                    }

                    few_shot_data.append(new_row)
