from evaluation.quantitive_metrics.psnr import psnr
from evaluation.quantitive_metrics.ssmi import ssmi
from evaluation.quantitive_metrics.sharp_diff import sharp_diff


def create_metrics():
    return [psnr, ssmi, sharp_diff]