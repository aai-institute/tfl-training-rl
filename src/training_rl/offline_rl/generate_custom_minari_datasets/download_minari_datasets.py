from training_rl.offline_rl.load_env_variables import load_env_variables
load_env_variables()

import minari
from training_rl.offline_rl.generate_custom_minari_datasets.generate_minari_dataset_grid_envs import \
    MinariDatasetConfig


def download_minari_dataset(data_set_name: str):
    """
    Download a Minari dataset from the cloud and save some meta info locally
    :param data_set_name: one of minari datasets, for instance "relocate-cloned-v1"
    :return:
    """

    minari.download_dataset(data_set_name)

    data = minari.load_dataset(data_set_name)

    minari_config = MinariDatasetConfig(
        env_name="See: https://minari.farama.org/main/",
        data_set_name=data_set_name,
        num_steps=data.total_steps,
    )
    minari_config.save_to_file()


if __name__ == "__main__":
    DATASET_NAME ="pen-human-v2"

    download_minari_dataset(DATASET_NAME)