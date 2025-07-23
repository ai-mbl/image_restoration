from typing import Any
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from careamics.lvae_training.dataset import DataSplitType
from careamics.lvae_training.dataset.utils.data_utils import get_datasplit_tuples
from microsplit_reproducibility.datasets.custom_dataset_2D import load_one_file
from microsplit_reproducibility.utils.paper_metrics import avg_range_inv_psnr, compute_SE, _get_list_of_images_from_gt_pred
from microssim import MicroMS3IM, MicroSSIM
from numpy.typing import NDArray
from skimage.metrics import structural_similarity
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure


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
    data_config: Any,
    datadir: str | Path,
    datasplit_type: DataSplitType,
    val_fraction: float,
    test_fraction: float,
    structures: list[Literal["Nuclei", "Microtubules", "NucMembranes", "Centromeres"]],
    **kwargs: Any,
) -> NDArray:
    """Split the data into train, validation, and test sets."""
    data = load_data(datadir, structures)
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


def compute_metrics(highres_data: list[torch.Tensor], pred_unnorm: list[torch.Tensor], verbose=True):
    """
    last dimension is the channel dimension
    """
    psnr_list = []
    microssim_list = []
    ms3im_list = []
    ssim_list = []
    msssim_list = []
    for ch_idx in range(highres_data[0].shape[-1]):
        # list of gt and prediction images. This handles both 2D and 3D data. 
        # This also handles when individual images are lists.
        gt_ch, pred_ch = _get_list_of_images_from_gt_pred(
            highres_data, pred_unnorm, ch_idx
        )
        
        # PSNR
        psnr_list.append(avg_range_inv_psnr(gt_ch, pred_ch))

        # MicroSSIM
        microssim_obj = MicroSSIM()
        microssim_obj.fit(gt_ch, pred_ch)
        mssim_scores = [
            microssim_obj.score(gt_ch[i], pred_ch[i]) for i in range(len(gt_ch))
        ]
        microssim_list.append((np.mean(mssim_scores), compute_SE(mssim_scores)))

        # MicroS3IM
        m3sim_obj = MicroMS3IM()
        m3sim_obj.fit(gt_ch, pred_ch)
        ms3im_scores = [
            m3sim_obj.score(gt_ch[i], pred_ch[i]) for i in range(len(gt_ch))
        ]
        ms3im_list.append((np.mean(ms3im_scores), compute_SE(ms3im_scores)))
        
        # SSIM
        ssim = [
            structural_similarity(
                gt_ch[i], pred_ch[i], data_range=gt_ch[i].max() - gt_ch[i].min()
            )
            for i in range(len(gt_ch))
        ]
        ssim_list.append((np.mean(ssim), compute_SE(ssim)))
        
        # MSSSIM
        ms_ssim = []
        for i in range(len(gt_ch)):
            ms_ssim_obj = MultiScaleStructuralSimilarityIndexMeasure(
                data_range=gt_ch[i].max() - gt_ch[i].min()
            )
            ms_ssim.append(
                ms_ssim_obj(
                    torch.Tensor(pred_ch[i][None, None]),
                    torch.Tensor(gt_ch[i][None, None]),
                ).item()
            )
        msssim_list.append((np.mean(ms_ssim), compute_SE(ms_ssim)))
    
    if verbose:

        def ssim_str(ssim_tmp):
            return f"{np.round(ssim_tmp[0], 3):.3f}+-{np.round(ssim_tmp[1], 3):.3f}"

        def psnr_str(psnr_tmp):
            return f"{np.round(psnr_tmp[0], 2)}+-{np.round(psnr_tmp[1], 3)}"

        print(
            "PSNR on Highres", "\t".join([psnr_str(psnr_tmp) for psnr_tmp in psnr_list])
        )
        print(
            "MicroSSIM on Highres",
            "\t".join([ssim_str(ssim) for ssim in microssim_list]),
        )
        print(
            "MicroS3IM on Highres", "\t".join([ssim_str(ssim) for ssim in ms3im_list])
        )
        print("SSIM on Highres", "\t".join([ssim_str(ssim) for ssim in ssim_list]))
        print("MSSSIM on Highres", "\t".join([ssim_str(ssim) for ssim in msssim_list]))

    return {
        "rangeinvpsnr": psnr_list,
        "microssim": microssim_list,
        "ms3im": ms3im_list,
        "ssim": ssim_list,
        "msssim": msssim_list,
    }