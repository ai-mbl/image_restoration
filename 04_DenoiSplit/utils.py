from typing import Any
from pathlib import Path
from typing import Literal, Union

import numpy as np
import torch
from careamics.lvae_training.dataset import DataSplitType
from careamics.lvae_training.dataset.utils.data_utils import get_datasplit_tuples
from microsplit_reproducibility.datasets.custom_dataset_2D import load_one_file
from microsplit_reproducibility.utils.paper_metrics import avg_range_inv_psnr, compute_SE, _get_list_of_images_from_gt_pred
from microssim import MicroMS3IM, MicroSSIM
from numpy.typing import NDArray
from skimage.measure import pearson_corr_coeff
from skimage.metrics import structural_similarity
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


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


def _normalize_for_lpips(imgs: list[NDArray]) -> NDArray:
    """Normalize the given image in [0, 1] for LPIPS.
    
    Parameters
    ----------
    img : NDArray
        A list of multi-channels images to normalize, each one of shape (C, Z, Y, X).
    
    Returns
    -------
    NDArray
        The normalized image.
    """
    # TODO: use training dset stats for normalization (?)
    ax_idxs = tuple(range(1, imgs[0].ndim))
    min_ = np.min([img.min(axis=ax_idxs) for img in imgs])
    max_ = np.max([img.max(axis=ax_idxs) for img in imgs])
    min_ = np.asarray(min_).reshape(-1, *np.ones_like(ax_idxs, dtype=int))
    max_ = np.asarray(max_).reshape(-1, *np.ones_like(ax_idxs, dtype=int))
    return np.array([(img - min_) / (max_ - min_) for img in imgs])


def lpips(
    prediction: Union[np.ndarray, torch.Tensor], 
    target: Union[np.ndarray, torch.Tensor]
) -> float:
    """Compute the Learned Perceptual Image Patch Similarity (LPIPS) over images.
    
    If inputs are 3D, LPIPS is averaged over the Z-stack.
    
    NOTES:
    - LPIPS can use different networks. Here we use the SqueezeNet model.
    - The inputs are expected to be normalized in the range [0, 1].
    - We use the mean reduction, i.e., the LPIPS value is averaged over the batch.

    Parameters
    ----------
    prediction : Union[np.ndarray, torch.Tensor]
        Array of predicted images, shape is (N, C, [Z], Y, X).
    target : Union[np.ndarray, torch.Tensor]
        Array of ground truth images, shape is (N, C, [Z], Y, X).

    Returns
    -------
    float
        LPIPS value over the batch.
    """
    assert prediction.shape == target.shape, "Prediction and target shapes must match."
    assert prediction.max() <= 1 and prediction.min() >= 0, (
        "Prediction must be normalized in [0, 1]."
    )
    assert target.max() <= 1 and target.min() >= 0, (
        "Target must be normalized in [0, 1]."
    )
    
    # check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # compute LPIPS
    lpips = LearnedPerceptualImagePatchSimilarity(
        net_type='squeeze', reduction='mean', normalize=True
    ).to(device)
    
    if len(prediction.shape) == 5: # 3D input
        # iterate over Z-stack
        return np.mean([
            lpips(
                torch.tensor(prediction[:, :, i], device=device, dtype=torch.float32),
                torch.tensor(target[:, :, i], device=device, dtype=torch.float32)
            ).item()
            for i in range(prediction.shape[2])
        ])
    else:
        return lpips(
            torch.tensor(prediction, device=device, dtype=torch.float32),
            torch.tensor(target, device=device, dtype=torch.float32)
        ).item()


def ssim_str(ssim_tmp):
    return f"{np.round(ssim_tmp[0], 3):.3f} ± {np.round(ssim_tmp[1], 3):.3f}"

def psnr_str(psnr_tmp):
    return f"{np.round(psnr_tmp[0], 2)} ± {np.round(psnr_tmp[1], 3)}"

def compute_metrics(
    highres_data,
    pred_unnorm,
    metrics,
):
    """
    last dimension is the channel dimension
    """
    mse_list = []
    psnr_list = []
    pearson_list = []
    microssim_list = []
    ms3im_list = []
    ssim_list = []
    msssim_list = []
    lpips_list = []
    for ch_idx in range(highres_data[0].shape[-1]):
        # list of gt and prediction images. This handles both 2D and 3D data. 
        # This also handles when individual images are lists.
        gt_ch, pred_ch = _get_list_of_images_from_gt_pred(
            highres_data, pred_unnorm, ch_idx
        )
        
        # PSNR
        if "PSNR" in metrics:
            psnr_list.append(avg_range_inv_psnr(gt_ch, pred_ch))
            print(
                "PSNR:", "\t".join([psnr_str(psnr_tmp) for psnr_tmp in psnr_list])
            )

        # MicroSSIM
        if "MicroSSIM" in metrics:
            microssim_obj = MicroSSIM()
            microssim_obj.fit(gt_ch, pred_ch)
            mssim_scores = [
                microssim_obj.score(gt_ch[i], pred_ch[i]) for i in range(len(gt_ch))
            ]
            microssim_list.append((np.mean(mssim_scores), compute_SE(mssim_scores)))
            print(
                "MicroSSIM:",
                "\t".join([ssim_str(ssim) for ssim in microssim_list]),
            )

        # MicroS3IM
        if "MicroS3IM" in metrics:
            m3sim_obj = MicroMS3IM()
            m3sim_obj.fit(gt_ch, pred_ch)
            ms3im_scores = [
                m3sim_obj.score(gt_ch[i], pred_ch[i]) for i in range(len(gt_ch))
            ]
            ms3im_list.append((np.mean(ms3im_scores), compute_SE(ms3im_scores)))
            print(
                "MicroS3IM:", "\t".join([ssim_str(ssim) for ssim in ms3im_list])
            )
        
        # SSIM
        if "SSIM" in metrics:
            ssim = [
                structural_similarity(
                    gt_ch[i], pred_ch[i], data_range=gt_ch[i].max() - gt_ch[i].min()
                )
                for i in range(len(gt_ch))
            ]
            ssim_list.append((np.mean(ssim), compute_SE(ssim)))
            print("SSIM:", "\t".join([ssim_str(ssim) for ssim in ssim_list]))

        # MSSSIM
        if "MSSSIM" in metrics:
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
            print("MSSSIM:", "\t".join([ssim_str(ssim) for ssim in msssim_list]))

        # Pearson's Correlation Coefficient
        if "Pearson" in metrics:
            pearson_scores = [
                pearson_corr_coeff(gt_ch[i].flatten(), pred_ch[i].flatten())
                for i in range(len(gt_ch))
            ]
            pearson_list.append((np.mean(pearson_scores), compute_SE(pearson_scores)))
            print(
                "Pearson:",
                "\t".join([ssim_str(ssim) for ssim in pearson_list]),
            )
            
        # LPIPS
        if "LPIPS" in metrics:
            lpips_scores = []
            for i in range(len(gt_ch)):
                # inputs are expected to be RGB + have batch dimension
                curr_target = np.repeat(
                    gt_ch[i][None, ...], repeats=3, axis=0
                )
                curr_pred = np.repeat(
                    pred_ch[i][None, ...], repeats=3, axis=0
                )
                curr_target = _normalize_for_lpips([curr_target])
                curr_pred = _normalize_for_lpips([curr_pred])
                lpips_scores.append(
                    lpips(
                        prediction=curr_pred,
                        target=curr_target
                    )
                )
            lpips_list.append((np.mean(lpips_scores), compute_SE(lpips_scores)))
            print(
                "LPIPS:",
                "\t".join([ssim_str(ssim) for ssim in lpips_list]),
            )

    return {
        "rangeinvpsnr": psnr_list,
        "microssim": microssim_list,
        "ms3im": ms3im_list,
        "ssim": ssim_list,
        "msssim": msssim_list,
    }