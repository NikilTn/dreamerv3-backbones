import atexit
import pathlib
import sys
import time
import warnings

import hydra
import torch
from omegaconf import OmegaConf

import tools
from buffer import Buffer
from dreamer import Dreamer
from envs import make_envs
from trainer import OnlineTrainer

warnings.filterwarnings("ignore")
sys.path.append(str(pathlib.Path(__file__).parent))
# torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

# Backbones with a parallel `forward_parallel` path get `recurrent_posterior=False`
# by default so CategoricalRSSM.observe dispatches to the parallel scan. CLI
# overrides (e.g. `model.rssm.recurrent_posterior=true`) still win.
_PARALLEL_BACKBONES = {"transformer", "storm", "mamba2"}


def _default_recurrent_posterior(backbone: str) -> bool:
    return str(backbone).lower() not in _PARALLEL_BACKBONES


OmegaConf.register_new_resolver("default_recurrent_posterior", _default_recurrent_posterior, replace=True)


@hydra.main(version_base=None, config_path="configs", config_name="configs")
def main(config):
    tools.set_seed_everywhere(config.seed)
    if config.deterministic_run:
        tools.enable_deterministic_run()
    logdir = pathlib.Path(config.logdir).expanduser()
    logdir.mkdir(parents=True, exist_ok=True)
    start_time = time.time()

    # Mirror stdout/stderr to a file under logdir while keeping console output.
    console_f = tools.setup_console_log(logdir, filename="console.log")
    atexit.register(lambda: console_f.close())

    print("Logdir", logdir)
    tools.save_resolved_config(logdir, config)
    tools.update_run_metadata(
        logdir,
        status="running",
        start_time_unix=start_time,
        seed=int(config.seed),
        device=str(config.device),
        env_name=str(config.env.task),
        model_backbone=str(config.model.backbone),
        rep_loss=str(config.model.rep_loss),
        cpc_enabled=bool(config.model.cpc.enabled),
        sampling_strategy=str(config.buffer.sampling.strategy),
        subexp=(
            "cpc_dfs"
            if bool(config.model.cpc.enabled) and str(config.buffer.sampling.strategy) == "dfs"
            else "cpc"
            if bool(config.model.cpc.enabled)
            else "dfs"
            if str(config.buffer.sampling.strategy) == "dfs"
            else "none"
        ),
    )

    logger = tools.Logger(logdir, wandb_config=config.get("wandb"))
    # save config
    logger.log_hydra_config(config)

    replay_buffer = Buffer(config.buffer)

    print("Create envs.")
    train_envs, eval_envs, obs_space, act_space = make_envs(config.env)

    print("Simulate agent.")
    agent = Dreamer(
        config.model,
        obs_space,
        act_space,
    ).to(config.device)
    try:
        policy_trainer = OnlineTrainer(config.trainer, replay_buffer, logger, logdir, train_envs, eval_envs)
        policy_trainer.begin(agent)

        items_to_save = {
            "agent_state_dict": agent.state_dict(),
            "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
        }
        torch.save(items_to_save, logdir / "latest.pt")
        end_time = time.time()
        tools.update_run_metadata(
            logdir,
            status="completed",
            end_time_unix=end_time,
            elapsed_seconds=end_time - start_time,
            trainer_steps=float(config.trainer.steps),
        )
    except Exception:
        end_time = time.time()
        tools.update_run_metadata(
            logdir,
            status="failed",
            end_time_unix=end_time,
            elapsed_seconds=end_time - start_time,
            trainer_steps=float(config.trainer.steps),
        )
        raise


if __name__ == "__main__":
    main()
