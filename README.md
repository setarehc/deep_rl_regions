# Linear Regions in Deep Reinforcement Learning

Training, evaluation and visualization of how deep reinforcement learning policies divide the state space from [Understanding the Evolution of Linear Regions in Deep Reinforcement Learning](https://https://www.cs.ubc.ca/~setarehc/projects/understanding_policies_webpage/page.html).

# Installation
This codebase was tested with Python3.7, Pytorch1.12.1 and CUDA10.2 (when required).
Clone the repository and navigate to the root directory of the repository:
```
cd deep_rl_regions
```
Install the required packages using:
```
pip install -r requirements.txt
```
Training logs are created using [Weights & Biases](https://wandb.ai/) (wandb) and evaluation metrics are also logged to wandb. Sign up on wandb using desired `USERNAME`, and create a project named `deep_rl_regions`. The chosen username and project name will be later used when running some of the scripts.


If you prefer not to use wandb, some effort is altering the scripts.
# Training

```
python train.py --device=cpu --env=HalfCheetah-v2 --total_timesteps=100000 --nsteps=1000 --eval_every=1 --save_every=1 --policy_dims=c16,16 --value_dims=c32,32 --save_dir=checkpoints --entity=USERNAME --project_name=deep_rl_regions
```
Environment can simply be changed to any Mujoco environment.

# Evaluation
To evaluate a single training run with wandb run id of `RUNID`, run:
```
python evaluate.py --run_id=RUNID --entity=USERNAME --project_name=deep_rl_regions
```
To evaluate all runs on a wandb training sweep with `SWEEPID`, run:
```
python evaluate.py --sweep_id=SWEEPID --entity=USERNAME --project_name=deep_rl_regions
```

### Pre Evaluation
To compute metrics over random lines and random trajectories as described in section 5.2 of the paper, run the following script before running `evaluate.py`:
```
python randomized_metric_helper.py --sweep_id=SWEEPID
``` 
then run `evaluate.py` with options `--evaluate_random_trajectories` and `--evaluate_random_lines`.


# Visualization
To visualize lineare regions in high-dimensional state spaces, we define a 2-dimensional plane using 3 randomly sampled points in the state space, and project the linear regions onto this plane and visualize the projected linear regions. To do this, run:
```
python visualize_hd.py --env=HalfCheetah-v2 --policy_path=path_to_saved_policy_checkpoint --stats_path=path_to_saved_stats_checkpoint --save_path=path_to_saved_plot_dir
```
To visualize 2-dimensional state spaces, run:
```
python visualize_hd.py --env=envs:Car1DEnv-v1 --policy_path=path_to_saved_policy_checkpoint --stats_path=path_to_saved_stats_checkpoint --save_path=path_to_saved_plot_dir
```


# Reference
```
@article{cohan2022understanding,
  title={Understanding the Evolution of Linear Regions in Deep Reinforcement Learning},
  author={Cohan, Setareh and Kim, Nam Hee and Rolnick, David and van de Panne, Michiel},
  journal={Advances in Neural Information Processing Systems},
  volume={35},
  pages={10891--10903},
  year={2022}
}
```

