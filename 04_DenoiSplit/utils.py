from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from careamics.lvae_training.dataset import DataSplitType
from careamics.lvae_training.dataset.utils.data_utils import get_datasplit_tuples
from microsplit_reproducibility.datasets.custom_dataset_2D import load_one_file


def load_data(
    datadir: str | Path,
    structures: list[Literal["Nuclei", "Microtubules", "NucMembranes", "Centromeres"]],
) -> NDArray:
    """Load data of the specified structures from the specified directory."""
    data_path = Path(datadir)
    
    # pick only directories that match the structures
    channel_dirs = sorted(
        p for p in data_path.iterdir() if p.is_dir() and p.name in structures
    )

    channels_data: list[NDArray] = []
    for channel_dir in channel_dirs:
        image_files = sorted(f for f in channel_dir.iterdir() if f.is_file())
        channel_images = [load_one_file(image_path) for image_path in image_files]

        channel_stack = np.concatenate(
            channel_images, axis=0
        )  # FIXME: this line works iff images have
        # a singleton channel dimension. Specify in the notebook or change with `torch.stack`??
        channels_data.append(channel_stack)

    final_data = np.stack(channels_data, axis=-1)
    return final_data


def get_train_val_data(
    datadir,
    datasplit_type: DataSplitType,
    val_fraction=None,
    test_fraction=None,
) -> NDArray:
    """Split the data into train, validation, and test sets."""
    data = load_data(datadir)
    train_idx, val_idx, test_idx = get_datasplit_tuples(
        val_fraction, test_fraction, len(data)
    )
    
    if datasplit_type == DataSplitType.All:
        data = data.astype(np.float64)
    elif datasplit_type == DataSplitType.Train:
        data = data[train_idx].astype(np.float64)
    elif datasplit_type == DataSplitType.Val:
        data = data[val_idx].astype(np.float64)
    elif datasplit_type == DataSplitType.Test:
        data = data[test_idx].astype(np.float64)
    else:
        raise Exception("invalid datasplit")

    return data