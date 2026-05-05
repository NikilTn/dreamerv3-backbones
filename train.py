import atexit
import pathlib
import sys
import time
import warnings

import hydra
import torch
from omegaconf import open_dict

import tools
from buffer import Buffer
from dreamer import Dreamer
from envs import make_envs
from trainer import OnlineTrainer

warnings.filterwarnings("ignore")
sys.path.append(str(pathlib.Path(__file__).parent))
# torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")


def load_resume_checkpoint(agent, checkpoint_path, device, *, load_optim=True, strict=True):
    checkpoint_path = pathlib.Path(checkpoint_path).expanduser()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Resume checkpoint does not exist: {checkpoint_path}")
    checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    if "agent_state_dict" not in checkpoint:
        raise KeyError(f"Resume checkpoint is missing 'agent_state_dict': {checkpoint_path}")
    agent.load_state_dict(checkpoint["agent_state_dict"], strict=bool(strict))
    if load_optim:
        optim_state = checkpoint.get("optims_state_dict")
        if optim_state is None:
            raise KeyError(f"Resume checkpoint is missing 'optims_state_dict': {checkpoint_path}")
        tools.recursively_load_optim_state_dict(agent, optim_state)
    return int(checkpoint.get("trainer_step", 0))


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

    logger = tools.Logger(logdir)
    # save config
    logger.log_hydra_config(config)

    replay_buffer = Buffer(config.buffer)

    print("Create envs.")
    train_envs, eval_envs, obs_space, act_space = make_envs(config.env)

    print("Simulate agent.")
    with open_dict(config.model):
        config.model.burn_in = int(getattr(config.trainer, "burn_in", 0))
    agent = Dreamer(
        config.model,
        obs_space,
        act_space,
    ).to(config.device)
    resume_step = 0
    if config.resume_checkpoint:
        resume_step = load_resume_checkpoint(
            agent,
            config.resume_checkpoint,
            config.device,
            load_optim=bool(config.resume_load_optim),
            strict=bool(config.resume_strict),
        )
        tools.update_run_metadata(
            logdir,
            resumed_from=str(pathlib.Path(config.resume_checkpoint).expanduser()),
            resume_step=int(resume_step),
            resume_load_optim=bool(config.resume_load_optim),
            resume_replay_buffer=False,
        )
        print(
            f"Resumed model from {config.resume_checkpoint} at trainer_step={resume_step}. "
            "Replay buffer was not restored and will be refilled from fresh env interaction.",
            flush=True,
        )
    try:
        policy_trainer = OnlineTrainer(config.trainer, replay_buffer, logger, logdir, train_envs, eval_envs)
        policy_trainer.begin(agent, initial_step=resume_step if resume_step > 0 else None)
        policy_trainer.save_checkpoint(agent, int(config.trainer.steps), final=True)
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
