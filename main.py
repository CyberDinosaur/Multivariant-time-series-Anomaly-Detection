import hydra
import omegaconf
from omegaconf import DictConfig


@hydra.main(config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    from utils.misc import seed_everything, get_device
    from model_pipeline import pipeline
    
    # Firstly we need to set the seeds of all relevant items
    if cfg['seed'] is not None:
        seed = seed_everything(cfg['seed'])
        
    # By the way, get the device we want to use (which might be declined if targeted device is occupied)
    device = get_device(cfg['device'])
     
    # Then we run the pipeline with cfg as its hyper-parameters on the device
    model = pipeline(cfg, device, seed)
 

if __name__ == '__main__':
    main()