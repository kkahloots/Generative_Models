from evaluation.quantitive_metrics.psnr import psnr
from evaluation.quantitive_metrics.ssmi import ssmi
from evaluation.quantitive_metrics.sharp_difference import sharpdiff


def create_metrics():
    return {'x_logits': [psnr, ssmi, sharpdiff]}