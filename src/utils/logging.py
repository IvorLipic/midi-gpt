import wandb

def init_wandb(config, resume=None, id=None):
    if resume:
        wandb.init(
            project="midi-gpt",
            config=config,
            resume=resume,
            id=id
        )
    else:
        wandb.init(
            project="midi-gpt",
            config=config
        )

def log(metrics):
    wandb.log(metrics)