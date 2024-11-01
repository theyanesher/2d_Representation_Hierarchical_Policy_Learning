import hydra
from omegaconf import DictConfig, OmegaConf

OmegaConf.register_new_resolver("eval", eval, replace=True)

# @hydra.main(version_base=None, config_path="conf", config_name="train_low_level_200_objects")
# def my_app(cfg : DictConfig) -> None:
#     # print(OmegaConf.to_yaml(cfg))
#     # cfg = OmegaConf.load(cfg)
#     # print(cfg.task.dataset.augmentation_rot)
#     print(cfg.task.dataset.num_load_episodes)

def compare_dicts(d1, d2, path=''):
    """Recursively compare two dictionaries and print differences."""
    for key in d1:
        if key not in d2:
            print(f"Key {path + '.' + key} is missing in the second dictionary.")
        elif isinstance(d1[key], dict) and isinstance(d2[key], dict):
            compare_dicts(d1[key], d2[key], path + '.' + key)
        elif d1[key] != d2[key]:
            print(f"Difference at {path + '.' + key}: {d1[key]} != {d2[key]}")


@hydra.main(version_base=None, config_path="conf", config_name="train_low_level_200_objects")
def compare_configs(config1):
    # Load configurations
    # config1 = OmegaConf.load('train_low_level_200_objects.yaml')
    config2 = OmegaConf.load('conf/config.yaml')
    # print(OmegaConf.to_yaml(config1))
    # print(config1.task.dataset.num_load_episodes)

    # Compare configurations
    diff = OmegaConf.to_container(config1, resolve=True) != OmegaConf.to_container(config2, resolve=True)
    
    if diff:
        print("Configurations differ.")
        # print("Config1:")
        # print(OmegaConf.to_yaml(config1))
        # print("Config2:")
        # print(OmegaConf.to_yaml(config2))
        # print(diff)
        config1 = OmegaConf.to_container(config1, resolve=True)
        config2 = OmegaConf.to_container(config2, resolve=True)
        compare_dicts(config1, config2)
        # print(config1)
    else:
        print("Configurations are identical.")


if __name__ == "__main__":
    compare_configs()
    # my_app()