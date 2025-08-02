import os
import wandb
import logging
from omegaconf import DictConfig

USER_HOME = os.path.expanduser("~")
DEFAULT_CACHE_FOLDER = os.path.join(USER_HOME, '.cache/llm_eval/')


class WandbTask():
    
    def __init__(self, cfg: DictConfig, wandb_exp_name: str, wandb_project: str, wandb_save_dir: str=None, **kwargs):
        
        if cfg.metrics.enable_report:
            os.environ["WANDB_BASE_URL"] = cfg.metrics.wandb.wandb_base_url
            os.environ["WANDB_API_KEY"] = cfg.metrics.wandb.wandb_api_key
            os.environ["WANDB_ENTITY"] = cfg.metrics.wandb.wandb_entity
            os.environ["WANDB_API_TIMEOUT"] = cfg.metrics.wandb.wandb_api_timeout
            os.environ["WANDB_RUN_TIMEOUT"] = cfg.metrics.wandb.wandb_run_timeout
            os.environ["WANDB_INIT_TIMEOUT"] = cfg.metrics.wandb.wandb_init_timeout
            
            if not wandb_save_dir:
                wandb_save_dir = DEFAULT_CACHE_FOLDER
            if not wandb_exp_name or not wandb_exp_name:
                raise ValueError("Please specify the wandb experiment name!")
            
            wandb_kwargs = {
                    "dir": wandb_save_dir,
                    "name": wandb_exp_name,
                    "project": wandb_project,
                    "config": kwargs
                }

            os.makedirs(wandb_kwargs['dir'], exist_ok=True)
            
            try:
                self.wandb_writer = wandb.init(**wandb_kwargs)
            except Exception as e:
                logging.error(f"Failed to initialize wandb: {str(e)}")
                self.wandb_writer = None
        else:
            self.wandb_writer = None


    def __del__(self):
        if self.wandb_writer:
            self.wandb_writer.finish()
            
    
    def get_wandb_writer(self):
        return self.wandb_writer
    
    
    def finish_wandb_writer(self):
        if self.wandb_writer:
            self.wandb_writer.finish()

