from evaluation.quantitive_metrics.peak_signal_to_noise_ratio import prepare_psnr
from evaluation.quantitive_metrics.structural_similarity import prepare_ssim_multiscale
from evaluation.quantitive_metrics.sharp_difference import prepare_sharpdiff
from statistical.ae_losses import prepare_mean_absolute_error, prepare_mean_squared_error

def create_metrics(inputs_flat_shape):
    return {'x_logits': [prepare_psnr(inputs_flat_shape), prepare_ssim_multiscale(inputs_flat_shape), prepare_sharpdiff(inputs_flat_shape), prepare_mean_absolute_error(inputs_flat_shape), prepare_mean_squared_error(inputs_flat_shape)]}