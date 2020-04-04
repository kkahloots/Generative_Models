from evaluation.quantitive_metrics.peak_signal_to_noise_ratio import psnr
from evaluation.quantitive_metrics.structural_similarity import ssim_multiscale
from evaluation.quantitive_metrics.sharp_difference import sharpdiff


def create_metrics():
    return {'x_logits': [psnr, ssim_multiscale, sharpdiff]}