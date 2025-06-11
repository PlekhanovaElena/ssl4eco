import warnings
# Hack to avoid pesky FutureWarnings from pl-bolts
warnings.simplefilter("ignore", category=FutureWarning)

import os
import logging
import sys

# Add directory of the current script to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hydra
from omegaconf import DictConfig, OmegaConf

from downstream_tasks.utils.root import init_root
init_root()

from linear_probing import run_linear_probing
from knn_probing import run_knn_probing

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="default")
def main(cfg : DictConfig) -> None:
    log.info("\n" + OmegaConf.to_yaml(cfg))
    log.info("Saving to: " + cfg.SAVE_NAME)
    log.info(f"Output directory: {hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}")

    try:
        os.mkdir(cfg.SAVE_NAME[:-len(cfg.SAVE_NAME.split("/")[-1])])
    except:
        pass
    try:
        os.mkdir(cfg.SAVE_NAME)
    except:
        pass
    if cfg.DATASET == "clef":
        try:
            # Need another folder for CLEF so we don't overrride
            os.mkdir(cfg.SAVE_NAME + "/blind")
        except:
            pass
    if "biomassters" in cfg.DATASET:
        try:
            # Need another folder for CLEF so we don't overrride
            os.mkdir(cfg.SAVE_NAME + "/test")
        except:
            pass

    if cfg.PROBE == "knn":
        run_knn_probing(cfg, log)
    elif cfg.PROBE == "linear":
        run_linear_probing(cfg, log)
    else:
        raise NotImplementedError(f"Unknown PROBE mode: {cfg.PROBE}")

if __name__ == "__main__":  
    main()