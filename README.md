<!--- BADGES: START --->

[![arXiv](https://img.shields.io/badge/arXiv-red?style=flat-square&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2505.21399)
[![GitHub - License](https://img.shields.io/github/license/UKPLab/arxiv2025-self-awareness)](https://opensource.org/licenses/Apache-2.0)


<!--- BADGES: END --->



# Factual Self-Awareness in Language Models: Representation, Robustness, and Scaling

This repository contains the code for reproducing the experiments from the paper "Factual Self-Awareness in Language Models: Representation, Robustness, and Scaling". The codebase is built on top of [Urartu](https://github.com/tamohannes/urartu), an open-source NLP framework that offers high-level wrappers for effortless experiment management, enhanced reproducibility, and flexible configuration. We recommend familiarizing yourself with Urartu's structure and capabilities before diving into this codebase.

## More about Self-Awareness 🤔

Factual incorrectness in generated content is one of the primary concerns in ubiquitous deployment of large language models (LLMs). Prior findings suggest LLMs can (sometimes) detect factual incorrectness in their generated content (i.e., fact-checking post-generation). In this work, we provide evidence supporting the presence of LLMs' 'internal compass' that dictate the correctness of factual recall at the time of generation.

We demonstrate that for a given subject entity and a relation, LLMs internally encode linear features in the Transformer's residual stream that dictate whether it will be able to recall the correct attribute (that forms a valid entity-relation-attribute triplet). This self-awareness signal is robust to minor formatting variations. 

We investigate the effects of context perturbation via different example selection strategies. Scaling experiments across model sizes and training dynamics highlight that self-awareness emerges rapidly during training and peaks in intermediate layers. These findings uncover intrinsic self-monitoring capabilities within LLMs, contributing to their interpretability and reliability.

![teaser](https://github.com/user-attachments/assets/7417edf1-aafd-4b7e-b1e2-500c4fa3c2aa)

The Self-Awareness codebase is built upon the [Urartu framework](https://github.com/tamohannes/urartu) (v3). For detailed insights into its structure, please refer to the [Getting Started Guide](https://github.com/tamohannes/urartu/blob/main/starter_template_setup.md).

## Installation 🚀

Getting started with self-awareness is simple! Just follow these steps:

```bash
pip install -e .
```

## Experiment Pipeline 🔬

The experiments are organized in a sequential pipeline of actions, where each action builds upon the outputs of previous actions. Here's the step-by-step process:

1. **Data Collection** (`_1_data_scraper`):
   - Scrapes and prepares the initial dataset
   - Output is stored in `.runs/[current_date]/`

2. **Sample Construction** (`_2_sample_constructor`):
   - Uses the output from step 1 to construct known and forgotten samples
   - Requires the output directory from step 1 as input

3. **Latent Separation Analysis** (`_3_latent_separation_score`):
   - Analyzes latent space separation between known and forgotten samples
   - Uses the output from step 2

4. **Comparative Analysis** (`_4_comp_alt_latent_separation_score`):
   - Calculates latent separation scores for both Sparse Autoencoders (SAEs) and probing methods
   - Compares the effectiveness of different approaches in separating known and forgotten samples
   - Uses the output from step 2

5. **Probe Training** (`_5_prob_trainer`):
   - Trains probe (e.g. linear) models on the constructed samples on each layer of the model
   - Uses the output from step 2

6. **Scaling Laws** (`_6_scaling_laws`):
   - Analyzes scaling properties across training checkpoints of a model
   - Uses the output from step 2

## Running Experiments 🧪

Each action can be run using the `urartu` command with appropriate configuration. Here's an example command structure:

```bash
urartu action_config=<action_name> aim=aim slurm=<slurm_config> +action_config.task.model.api_token=YOUR_API_TOKEN
```

Where:
- `<action_name>` is one of: `_1_data_scraper`, `_2_sample_constructor`, `_3_latent_separation_score`, `_4_comp_alt_latent_separation_score`, `_5_prob_trainer`, or `_6_scaling_laws`
- `<slurm_config>` can be `slurm` for cluster execution or `no_slurm` for local execution

### Example Commands

1. Run data collection:
```bash
urartu action_config=_1_data_scraper aim=aim slurm=no_slurm +action_config.task.model.api_token=YOUR_API_TOKEN
```

2. Run sample construction:
```bash
urartu action_config=_2_sample_constructor aim=aim slurm=no_slurm +action_config.task.model.api_token=YOUR_API_TOKEN
```

3. Run subsequent experiments:
```bash
urartu action_config=_6_scaling_laws aim=aim slurm=no_slurm +action_config.task.model.api_token=YOUR_API_TOKEN
```

> **Note**: The input paths for each action can be specified either:
> - In the corresponding YAML config file (recommended for reproducibility)
> - Via command line arguments using the `+action_config.task.input_dir` parameter

## Configuration ⚙️

This project uses [Hydra](https://hydra.cc/) for configuration management. Each action has a corresponding configuration file in the `configs/action_config` directory with the same name as the action (e.g., `_1_data_scraper.yaml`). These configs can be modified to adjust experiment parameters.

> **Note**: The configuration files contain TODO comments indicating where the output paths from previous actions should be specified. When running subsequent actions in the pipeline, make sure to update these paths with the actual output directories from the previous steps.

Hydra allows for flexible configuration management through:
- YAML configuration files
- Command-line overrides using the `+` prefix
- Configuration composition
- Environment variable interpolation

## Output Structure 📁

All experiment outputs are stored in the `.runs` directory, organized by date. Each run creates a new directory with the format `.runs/YYYY-MM-DD/`. The output from each action should be used as input for subsequent actions in the pipeline.


## Cite 📚

If you use this code in your research, please cite our paper:

```
@misc{tamoyan2025factualselfawarenesslanguagemodels,
      title={Factual Self-Awareness in Language Models: Representation, Robustness, and Scaling}, 
      author={Hovhannes Tamoyan and Subhabrata Dutta and Iryna Gurevych},
      year={2025},
      eprint={2505.21399},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.21399}, 
}
```

## Contacts

[Hovhannes Tamoyan](mailto:hovhannes.tamoyan@tu-darmstadt.de)

Please feel free to contact us if you have any questions or need to report any issues.


## Links 🔗

[UKP Lab Homepage](https://www.ukp.tu-darmstadt.de/) | [TU Darmstadt Website](https://www.tu-darmstadt.de/index.en.jsp)

## Disclaimer ⚠️

> This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication.
