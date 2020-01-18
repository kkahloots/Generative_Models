from evaluation.quantitive_metrics.psnr import psnr_metric
from evaluation.quantitive_metrics.ssmi import ssmi_metric
from evaluation.quantitive_metrics.sharp_diff import sharp_diff_metric


def compute_metrics(inputs):
    metrics = {}
    metrics['psnr'] = psnr_metric(inputs)
    metrics['ssmi'] = ssmi_metric(inputs)
    metrics['sharp_diff'] = sharp_diff_metric(inputs)

    return metrics