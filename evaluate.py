import wandb
import os
import pickle
import math
import numpy as np
from pathlib import Path
from argparse import ArgumentParser
from compute_metrics import evaluate_analytic


def main(args):
    # Load training runs from wandb
    api = wandb.Api()
    if args.run_id is not None and args.sweep_id is None: # single run evaluation
        runs = api.runs(path=f"{args.entity}/{args.project_name}", filters={"config.run_name":args.run_id}) #, "tags":"train"
    elif args.sweep_id is not None and args.run_id is None:
        sweep = api.sweep(f"{args.entity}/{args.project_name}/sweeps/{args.sweep_id}")
        runs = sweep.runs
    else:
        raise ValueError("ID of training run/runs required.")
    runs = sorted(runs, key=lambda x: x.id)
    
    # SLURM multi processing handler if one is used
    if "SLURM_ARRAY_TASK_ID" in os.environ:
        N = len(runs)
        count = int(os.environ["SLURM_ARRAY_TASK_COUNT"])
        array_idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
        min_idx = int(os.environ["SLURM_ARRAY_TASK_MIN"])
        chunk_length = int(math.ceil(N/count))
        start_idx = (array_idx - min_idx) * chunk_length
        end_idx = start_idx + chunk_length
        if start_idx > N:
            return
        if end_idx >= N:
            end_idx = N
        runs = runs[start_idx: end_idx]

    for run in runs:
        env_name = run.config["env"]
        checkpoints_dir = run.config["save_dir"]
        policy_dims = run.config["policy_dims"]
        run_folder_path = Path(checkpoints_dir) / f"car1d_{run.id}"
        
        id_name = args.sweep_id if args.sweep_id is not None else args.run_id
        random_evaluation_dir = Path(args.randomized_data_dir) / id_name

        metrics_dict, summary_dict = evaluate_analytic(env_name=env_name, policy_dims=policy_dims, run_folder_path=run_folder_path,
                                                       stochastic_policy=args.add_stochastic,
                                                       random_evaluation_dir=random_evaluation_dir, evaluate_random_trajectories=args.evaluate_random_trajectories, 
                                                       evaluate_random_lines=args.evaluate_random_lines, multi_processing=args.multi_processing)
        
        if args.log_locally:
            # Log computed metrics locally
            save_path = Path("results") / id_name / f"car1d_{run.id}"
            os.makedirs(save_path, exist_ok=True)

            # log history
            history_path = save_path / "history_fixed.pkl"
            for epoch, epoch_metrics_dict in enumerate(metrics_dict.values()):
                epoch_metrics_dict["epoch"] = epoch
            pickle.dump(metrics_dict, open(history_path, "wb"))
            
            # log summary
            summary_path = save_path / "summary.pkl"
            pickle.dump(summary_dict, open(summary_path, "wb"))

        if not args.no_wandb_log:
            # Log evaluation metrics on wandb
            # initialize a wandb evaluation run corresponding to the training run evaluated
            eval_run = wandb.init(project=args.project_name, tags=["SUBMIT"], entity=args.entity, config=run.config, reinit=True, settings=wandb.Settings(start_method="thread"))
            wandb.config.update({"sweep_id": args.sweep_id})

            # log metrics to wandb
            for epoch, epoch_metrics_dict in enumerate(metrics_dict.values()):
                epoch_metrics_dict["epoch"] = epoch
                # epoch_metrics_dict["cumulative_rewards"] = cumulative_rewards[epoch] # TODO must have eval_every=1 otherwise, some will be NaN
                wandb.log(epoch_metrics_dict)

            # write summary in wandb
            eval_run.summary.update(summary_dict)
            wandb.finish()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--entity", help="Weights & Biases entity", type=str)
    parser.add_argument("--project_name", help="Weights & Biases project name", required=True, type=str)
    parser.add_argument("--sweep_id", help="To evaluate a set of runs in a sweep, input sweep_id", type=str, default=None)
    parser.add_argument("--run_id", help="For single run evaluation, use run_id instead of sweep_id", type=str, default=None)
    parser.add_argument("--randomized_data_dir", type=str, default="randomized_data")
    parser.add_argument("--add_stochastic", help="Include results from the stochastic policy as well", action="store_true") # default is false; only use deterministic policy
    parser.add_argument("--evaluate_random_trajectories", help="Evaluate over random trajectories", action="store_true") # default is false - must first run randomized_metric_helper.py if want to use this
    parser.add_argument("--evaluate_random_lines", help="Evaluate over random lines", action="store_true") # default is false - must first run randomized_metric_helper.py if want to use this
    parser.add_argument("--log_locally", help="Save computed metrics locally", action="store_true") # default is false
    parser.add_argument("--no_wandb_log", help="Do not log computed metrics to wandb", action="store_true") # default is false; logs to wandb
    parser.add_argument("--multi_processing", help="Run line segment computation in parallel", action="store_true") # default is false; no multiprocessing
    args = parser.parse_args()
    main(args)
