# GitHub - facebookresearch/jepa-wms: Code, data and weights for the paper **What drives success in physical planning with Joint-Embedding Predictive World Models?** · GitHub

**URL:** https://github.com/facebookresearch/jepa-wms

---

Skip to content
Navigation Menu
Platform
Solutions
Resources
Open Source
Enterprise
Pricing
Sign in
Sign up
facebookresearch
/
jepa-wms
Public
Notifications
Fork 50
 Star 449
Code
Issues
2
Pull requests
2
Discussions
Actions
Projects
Security and quality
Insights
main
3 Branches
0 Tags
Code
Folders and files
Name	Last commit message	Last commit date

Latest commit
Basile-Terv
Merge pull request #28 from facebookresearch/rebuttal-exps
13cf1d9
 · 
History
9 Commits


.github
	
Initial open source release
	


app
	
Add dset_fraction, distributed improvements, data-scaling configs
	


assets
	
Initial open source release
	


configs
	
Add dset_fraction, distributed improvements, data-scaling configs
	


evals
	
Add dset_fraction, distributed improvements, data-scaling configs
	


src
	
Add dset_fraction, distributed improvements, data-scaling configs
	


tests
	
Initial open source release
	


.flake8
	
Initial open source release
	


.gitignore
	
remove use_config_folder, add decoders in README.md
	


.pre-commit-config.yaml
	
Initial open source release
	


CODE_OF_CONDUCT.md
	
Initial open source release
	


CONTRIBUTING.md
	
Initial open source release
	


LICENSE
	
Initial open source release
	


README.md
	
Add models on HuggingFace Hub
	


THIRD-PARTY-LICENSES.md
	
Initial open source release
	


hubconf.py
	
Add models on HuggingFace Hub
	


pyproject.toml
	
Add models on HuggingFace Hub
	


setup_macros.py
	
Initial open source release
	
Repository files navigation
README
Code of conduct
Contributing
License
Security

🌍 JEPA-WMs

What Drives Success in Physical Planning with
Joint-Embedding Predictive World Models?

   



Meta AI Research, FAIR

Basile Terver, Tsung-Yen Yang, Jean Ponce, Adrien Bardes, Yann LeCun

PyTorch implementation, data and pretrained models for JEPA-WMs.

🎯 Pretrained Models

We provide pretrained JEPA-WMs, as well as DINO-WM and V-JEPA-2-AC(fixed) baseline models for various environments.

Download options: Models are available on 🤗 Hugging Face Hub (recommended) or via direct download from fbaipublicfiles.

JEPA-WM Models
Environment	Resolution	Encoder	Pred. Depth	Weights
DROID & RoboCasa	256×256	DINOv3 ViT-L/16	12	🤗 HF / direct
Metaworld	224×224	DINOv2 ViT-S/14	6	🤗 HF / direct
Push-T	224×224	DINOv2 ViT-S/14	6	🤗 HF / direct
PointMaze	224×224	DINOv2 ViT-S/14	6	🤗 HF / direct
Wall	224×224	DINOv2 ViT-S/14	6	🤗 HF / direct
DINO-WM Baseline Models
Environment	Resolution	Encoder	Pred. Depth	Weights
DROID & RoboCasa	224×224	DINOv2 ViT-S/14	6	🤗 HF / direct
Metaworld	224×224	DINOv2 ViT-S/14	6	🤗 HF / direct
Push-T	224×224	DINOv2 ViT-S/14	6	🤗 HF / direct
PointMaze	224×224	DINOv2 ViT-S/14	6	🤗 HF / direct
Wall	224×224	DINOv2 ViT-S/14	6	🤗 HF / direct
V-JEPA-2-AC(fixed) Baseline Model
Environment	Resolution	Encoder	Pred. Depth	Weights
DROID & RoboCasa	256×256	V-JEPA-2 ViT-G/16	24	🤗 HF / direct
VM2M Decoder Heads (optional)

Decoder heads enable visualization and rollout decoding. They are not required for training world models or running planning evaluations.

Decoder	Encoder	Resolution	Weights
dinov2_vits_224 (05norm)	DINOv2 ViT-S/14	224×224	🤗 HF / direct
dinov2_vits_224_INet	DINOv2 ViT-S/14	224×224	🤗 HF / direct
dinov3_vitl_256_INet	DINOv3 ViT-L/16	256×256	🤗 HF / direct
vjepa2_vitg_256_INet	V-JEPA-2 ViT-G/16	256×256	🤗 HF / direct

Decoder assignment: DINO-WM uses dinov2_vits_224 (05norm), JEPA-WM uses INet variants (dinov2_vits_224_INet for sim envs, dinov3_vitl_256_INet for real-robot), VJ2AC uses vjepa2_vitg_256_INet.

🔌 Loading Models with PyTorch Hub
🤗 Loading Models with Hugging Face Hub
🚀 Getting Started
Installation

We use conda for system dependencies (FFmpeg) and uv for fast Python package management.

# 1. Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Create conda environment with FFmpeg
conda create -n jepa-wms python=3.10 ffmpeg=7 -c conda-forge -y
conda activate jepa-wms

# 3. Clone and install
git clone git@github.com:facebookresearch/jepa-wms.git
cd jepa-wms
uv pip install -e .
# Optional: Install dev dependencies
uv pip install -e ".[dev]"

# 4. Verify installation
python -c "import torchcodec; print('✓ torchcodec works')"
⚙️ Configuration

Set these environment variables in your ~/.bashrc or ~/.zshrc:

export JEPAWM_DSET=/path/to/your/datasets
export JEPAWM_LOGS=/desired_path/to/your/train_logs_and_planning_eval_logs
export JEPAWM_HOME=/path/to/your/workspace # dir where you cloned this repo
export JEPAWM_CKPT=/desired_path/to/your/saved_checkpoints # Optional
export JEPAWM_OSSCKPT=/path/to/your/pretrained_opensource_encoders  # Optional

Note on config paths: In training configs (configs/vjepa_wm/), the folder field (using ${JEPAWM_LOGS}) stores train / validation logs and planning eval outputs, while checkpoint_folder (using ${JEPAWM_CKPT}) stores saved model checkpoints. If checkpoint_folder is omitted, it defaults to folder.

Then run:

source ~/.bashrc && cd $JEPAWM_HOME/jepa-wms && python setup_macros.py && conda activate jepa-wms
📁 Repository structure under JEPAWM_HOME
🧠 Pretrained Encoders
🤖 MuJoCo 2.1 for PointMaze
🏠 RoboCasa install (optional)
📦 Downloading Data

All datasets are available on 🤗 HuggingFace: facebook/jepa-wms

# Download all datasets
python src/scripts/download_data.py

# Download specific dataset(s)
python src/scripts/download_data.py --dataset pusht pointmaze wall

# List available datasets
python src/scripts/download_data.py --list
Dataset	Description
pusht	Push-T environment trajectories*
pointmaze	PointMaze navigation trajectories*
wall	Wall environment trajectories*
metaworld	42 Metaworld tasks (100 episodes each)
robocasa	RoboCasa kitchen manipulation
franka	Franka robot trajectories

* The pusht, pointmaze, and wall datasets are sourced from the DINO-WM project without modification. We re-host them on our HuggingFace repository for convenience.

🤖 DROID dataset (optional)
📂 Dataset directory structure
💡 Common Concepts
🐛 The --debug Flag

Use --debug with app.main or evals.main to run in single-process mode on the current node:

python -m app.main --fname <config.yaml> --debug

This is useful for:

Interactive debugging with pdb breakpoints
Single-GPU runs without distributed overhead

⚠️ Don't confuse with meta.quick_debug in config files, which reduces dataset size and iterations for quick sanity checks.

🔄 Automatic Evaluation During Training

The training script automatically launches planning evaluations every meta.eval_freq epochs:

Config generation: Merges your training settings with eval templates from configs/online_plan_evals/
Job submission: Launches eval jobs for each generated config

The evals.separate option controls how evals are executed:

Value	Behavior
true (default)	Submit as separate SLURM jobs via sbatch
false	Run evals on rank 0 of the training job
🏋️ Training
Quick Start

Distributed training (from login node):

python -m app.main_distributed --fname configs/vjepa_wm/<env>_sweep/<model>.yaml --account <account> --qos <qos> --time <time>

Single-GPU training (interactive session):

python -m app.main --fname configs/vjepa_wm/<env>_sweep/<model>.yaml --debug
📋 Paper Configs
🎨 Training Decoder Heads (optional)
📊 Evaluation
⚙️ Manual Eval Config Generation

Eval configs are auto-generated during training. You can also manually generate or write eval configs to run evaluations independently:

Set meta.plan_only_eval_mode: true in your training config
Set evals.dump_eval_configs: true in your training config
Run: python -m app.main --fname <config.yaml> --debug

The dump directory is automatically derived from evals.eval_cfg_paths (e.g., configs/online_plan_evals/mz/... → configs/dump_online_evals/mz/).

▶️ Running Evaluations

Once you have a valid eval config, run evaluations using:

# Single GPU
python -m evals.main --fname <config.yaml> --debug

# Distributed
python -m evals.main_distributed --fname <config.yaml> --account <account> --qos lowest --time 120

# Grid evaluation (sweep over hyperparameters or epoch checkpoints)
python -m evals.simu_env_planning.run_eval_grid --env <env> --config <config.yaml>

📓 Visualization: app/plan_common/notebooks/logs_planning_joint.ipynb

Full documentation: evals/simu_env_planning/README.md

📈 Reproducing Paper Design Choice Plots
🔮 Unroll Decode Evaluation
📁 Code Structure
.
├── app                              # training loops
│   ├── vjepa_wm                     #   train world model / heads
│   ├── plan_common                  #   shared planning components
│   │   ├── datasets                 #   environment-specific datasets
│   │   ├── models                   #   world model architectures
│   │   └── plot                     #   plotting utilities
│   ├── main_distributed.py          #   entrypoint for sbatch on slurm
│   └── main.py                      #   entrypoint for local run
├── configs                          # config files
│   ├── dump_online_evals            #   generated eval cfgs from train loop
│   ├── evals                        #   pre-generated full eval cfgs
│   ├── online_plan_evals            #   eval cfg templates to fill with train cfg
│   ├── vjepa_wm                     #   train configs
├── evals                            # evaluations
│   ├── simu_env_planning            #   planning evaluation
│   ├── main_distributed.py          #   entrypoint for distributed evals
│   └── main.py                      #   entrypoint for local evals
├── src                              # the package
│   ├── datasets                     #   VM2M datasets, loaders (optional)
│   ├── models                       #   V-JEPA1/2 model definitions
│   ├── masks                        #   masking utilities (optional)
│   └── utils                        #   shared utilities
├── tests                            # unit tests for some modules


🔧 Troubleshooting
🖥️ SLURM Configuration (HPC Users)
🖥️ MuJoCo Rendering
🚀 Distributed jobs
🔄 Updating uv.lock
🐍 numba/numpy issues
📄 License

This project is licensed under CC-BY-NC 4.0. See THIRD-PARTY-LICENSES.md for third-party components.

📚 Citing JEPA-WMs

If you find this repository useful, please consider giving a ⭐ and citing:

@misc{terver2025drivessuccessphysicalplanning,
      title={What Drives Success in Physical Planning with Joint-Embedding Predictive World Models?},
      author={Basile Terver and Tsung-Yen Yang and Jean Ponce and Adrien Bardes and Yann LeCun},
      year={2025},
      eprint={2512.24497},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2512.24497},
}
About

Code, data and weights for the paper **What drives success in physical planning with Joint-Embedding Predictive World Models?**

Resources
Readme
License
Code of conduct
Code of conduct
Contributing
Contributing
Security policy
Security policy
Activity
Custom properties
Stars
449 stars
Watchers
9 watching
Forks
50 forks
Report repository
Releases
No releases published
Contributors
1
 (1)
Basile-TervBasile Terver
Languages
Python
99.1%
Jupyter Notebook
0.9%
Footer
© 2026 GitHub, Inc.
Footer navigation
Terms
Privacy
Security
Status
Community
Docs
Contact
Manage cookies
Do not share my personal information
 