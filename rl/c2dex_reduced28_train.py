"""Bounded single-GPU paired fine-tuning; original branch PPO and rewards."""
import argparse
import json
from pathlib import Path
import os


def main():
    from isaaclab.app import AppLauncher
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--updates',type=int,default=200);p.add_argument('--num-envs',type=int,default=512)
    p.add_argument('--seed',type=int,default=42)
    AppLauncher.add_app_launcher_args(p);a=p.parse_args()
    if a.updates<=0 or a.num_envs<=0:raise ValueError('Positive bounded budget required')
    a.output.mkdir(parents=True,exist_ok=False)
    app=AppLauncher(a).app
    try:
        import gymnasium as gym
        import torch
        import isaaclab_tasks
        import regrind.tasks
        from isaaclab_tasks.utils import load_cfg_from_registry
        from isaaclab.utils.io import dump_yaml
        from rl.c2dex_reduced28_runtime import configure,runner_for
        task='Regrind-Tron2-Revo3-Screwdriver-PrecomputedIK-v0'
        cfg=configure(load_cfg_from_registry(task,'env_cfg_entry_point'),a.inputs,a.output,a.num_envs,a.seed)
        # Keep the branch's RSI phase distribution for the paired adaptation.
        cfg.commands.motion.rsi_enabled=True
        env=gym.make(task,cfg=cfg)
        runner,wrapped=runner_for(env,task,a.seed,a.output)
        runner.load(str(a.checkpoint),load_optimizer=True,map_location='cuda:0')
        initial_iteration=int(runner.current_learning_iteration)
        env.unwrapped.common_step_counter=initial_iteration*24
        runner.save_interval=a.updates
        dump_yaml(str(a.output/'env.yaml'),cfg)
        start=__import__('time').monotonic()
        runner.learn(num_learning_iterations=a.updates,init_at_random_ep_len=False)
        output=a.output/'final.pt';runner.save(str(output))
        (a.output/'training_summary.json').write_text(json.dumps(dict(completed=True,
            initial_checkpoint=str(a.checkpoint),initial_iteration=initial_iteration,
            final_iteration=int(runner.current_learning_iteration),additional_updates=a.updates,
            environments=a.num_envs,rollout_steps=24,transitions=a.updates*a.num_envs*24,
            checkpoint=str(output),wall_seconds=__import__('time').monotonic()-start,
            CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),
            scope='bounded paired ZIP-only nominal/no-DR fine-tuning, not original full training reproduction'),indent=2))
        wrapped.close()
    finally:app.close()


if __name__=='__main__':main()
