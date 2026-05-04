import wandb

def init_wandb(config):
    wandb.init(
        project="midi-gpt",
        config=config
    )

def log(metrics):
    wandb.log(metrics)