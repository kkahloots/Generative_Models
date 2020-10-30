import tensorflow as tf
import tensorflow_probability as tfp
from evaluation.shared import log10

from evaluation.quantitive_metrics.structural_similarity import prepare_ssim_multiscale
def prepare_ell_struct(outputs_flat_shape):
    ssim_fn = prepare_ssim_multiscale(outputs_flat_shape)
    def ell_struct_loss(x_true, x_logits):
        x_logits = tf.reshape(x_logits, tf.TensorShape(outputs_flat_shape))
        x_true = tf.reshape(x_true, tf.TensorShape(outputs_flat_shape))

        dist = tfp.distributions.Bernoulli(
            probs=tf.clip_by_value(x_true, 1e-6, 1 - 1e-6))
        loss_lower_bound = dist.entropy()

        ell = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logits, labels=x_true)

        ellwlb = tf.reduce_sum(ell-loss_lower_bound, axis=[1, 2, 3])
        ell_noise = -log10(ellwlb)
        ssim = -ssim_fn(x_true, tf.sigmoid(x_logits))
        total_variation = -tf.reduce_sum(tf.image.total_variation(tf.sigmoid(x_logits)), name='total_variation')

        return  ellwlb + ell_noise + ssim + total_variation
    return ell_struct_loss

def frechet_sigma_trace(real_activations, generated_activations):
    """Classifier distance for evaluating a generative model.
    This methods computes the Frechet classifier distance from activations of
    real images and generated images. This can be used independently of the
    frechet_classifier_distance() method, especially in the case of using large
    batches during evaluation where we would like precompute all of the
    activations before computing the classifier distance.
    This technique is described in detail in https://arxiv.org/abs/1706.08500.
    Given two Gaussian distribution with means m and m_w and covariance matrices
    C and C_w, this function calculates
                  |m - m_w|^2 + Tr(C + C_w - 2(C * C_w)^(1/2))
    which captures how different the distributions of real images and generated
    images (or more accurately, their visual features) are. Note that unlike the
    Inception score, this is a true distance and utilizes information about real
    world images.
    Note that when computed using sample means and sample covariance matrices,
    Frechet distance is biased. It is more biased for small sample sizes. (e.g.
    even if the two distributions are the same, for a small sample size, the
    expected Frechet distance is large). It is important to use the same
    sample size to compute frechet classifier distance when comparing two
    generative models.
    Args:
      real_activations: 2D Tensor containing activations of real data. Shape is
        [batch_size, activation_size].
      generated_activations: 2D Tensor containing activations of generated data.
        Shape is [batch_size, activation_size].
    Returns:
     The Frechet Inception distance. A floating-point scalar of the same type
     as the output of the activations.
    """

    # Compute mean and covariance matrices of activations.
    m = tf.reduce_mean(real_activations, 0)
    m_w = tf.reduce_mean(generated_activations, 0)
    num_examples_real = tf.cast(tf.shape(real_activations)[0], dtype='float32')
    num_examples_generated = tf.cast(tf.shape(generated_activations)[0], dtype='float32')

    # sigma = (1 / (n - 1)) * (X - mu) (X - mu)^T
    real_centered = real_activations - m
    sigma = tf.matmul(
        real_centered, real_centered, transpose_a=True) / (
                    num_examples_real - 1)

    gen_centered = generated_activations - m_w
    sigma_w = tf.matmul(
        gen_centered, gen_centered, transpose_a=True) / (
                      num_examples_generated - 1)

    # Find the Tr(sqrt(sigma sigma_w)) component of FID
    sqrt_trace_component = trace_sqrt_product(sigma, sigma_w)

    # Compute the two components of FID.

    # First the covariance component.
    # Here, note that trace(A + B) = trace(A) + trace(B)
    trace = tf.linalg.trace(sigma + sigma_w+1e-6) - 2.0 * sqrt_trace_component

    return trace


def trace_sqrt_product(sigma, sigma_v):
    """Find the trace of the positive sqrt of product of covariance matrices.
    '_symmetric_matrix_square_root' only works for symmetric matrices, so we
    cannot just take _symmetric_matrix_square_root(sigma * sigma_v).
    ('sigma' and 'sigma_v' are symmetric, but their product is not necessarily).
    Let sigma = A A so A = sqrt(sigma), and sigma_v = B B.
    We want to find trace(sqrt(sigma sigma_v)) = trace(sqrt(A A B B))
    Note the following properties:
    (i) forall M1, M2: eigenvalues(M1 M2) = eigenvalues(M2 M1)
     => eigenvalues(A A B B) = eigenvalues (A B B A)
    (ii) if M1 = sqrt(M2), then eigenvalues(M1) = sqrt(eigenvalues(M2))
     => eigenvalues(sqrt(sigma sigma_v)) = sqrt(eigenvalues(A B B A))
    (iii) forall M: trace(M) = sum(eigenvalues(M))
     => trace(sqrt(sigma sigma_v)) = sum(eigenvalues(sqrt(sigma sigma_v)))
                                   = sum(sqrt(eigenvalues(A B B A)))
                                   = sum(eigenvalues(sqrt(A B B A)))
                                   = trace(sqrt(A B B A))
                                   = trace(sqrt(A sigma_v A))
    A = sqrt(sigma). Both sigma and A sigma_v A are symmetric, so we **can**
    use the _symmetric_matrix_square_root function to find the roots of these
    matrices.
    Args:
    sigma: a square, symmetric, real, positive semi-definite covariance matrix
    sigma_v: same as sigma
    Returns:
    The trace of the positive square root of sigma*sigma_v
    """

    # Note sqrt_sigma is called "A" in the proof above
    sqrt_sigma = _symmetric_matrix_square_root(sigma)

    # This is sqrt(A sigma_v A) above
    sqrt_a_sigmav_a = tf.matmul(sqrt_sigma, tf.matmul(sigma_v, sqrt_sigma))

    return tf.linalg.trace(_symmetric_matrix_square_root(sqrt_a_sigmav_a))


def _symmetric_matrix_square_root(mat, eps=1e-10):
    """Compute square root of a symmetric matrix.
    Note that this is different from an elementwise square root. We want to
    compute M' where M' = sqrt(mat) such that M' * M' = mat.
    Also note that this method **only** works for symmetric matrices.
    Args:
    mat: Matrix to take the square root of.
    eps: Small epsilon such that any element less than eps will not be square
      rooted to guard against numerical instability.
    Returns:
    Matrix square root of mat.
    """
    # Unlike numpy, tensorflow's return order is (s, u, v)
    s, u, v = tf.linalg.svd(mat)
    # sqrt is unstable around 0, just use 0 in such case
    si = tf.where(tf.less(s, eps), s, tf.sqrt(s))
    # Note that the v returned by Tensorflow is v = V
    # (when referencing the equation A = U S V^T)
    # This is unlike Numpy which returns v = V^T
    return tf.matmul(tf.matmul(u, tf.linalg.diag(si)), v, transpose_b=True)

# Default values obtained by Wang et al.
_MSSSIM_WEIGHTS = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)

def contrast_structure_fn(img1,
                       img2,
                       max_val,
                       outputs_flat_shape,
                       power_factors=_MSSSIM_WEIGHTS,
                       filter_size=11,
                       filter_sigma=1.5,
                       k1=0.01,
                       k2=0.03):
    """Computes the MS-SSIM between img1 and img2.
    This function assumes that `img1` and `img2` are image batches, i.e. the last
    three dimensions are [height, width, channels].
    Note: The true SSIM is only defined on grayscale.  This function does not
    perform any colorspace transform.  (If input is already YUV, then it will
    compute YUV SSIM average.)
    Original paper: Wang, Zhou, Eero P. Simoncelli, and Alan C. Bovik. "Multiscale
    structural similarity for image quality assessment." Signals, Systems and
    Computers, 2004.
    Arguments:
    img1: First image batch.
    img2: Second image batch. Must have the same rank as img1.
    max_val: The dynamic range of the images (i.e., the difference between the
      maximum the and minimum allowed values).
    power_factors: Iterable of weights for each of the scales. The number of
      scales used is the length of the list. Index 0 is the unscaled
      resolution's weight and each increasing scale corresponds to the image
      being downsampled by 2.  Defaults to (0.0448, 0.2856, 0.3001, 0.2363,
      0.1333), which are the values obtained in the 01 paper.
    filter_size: Default value 11 (size of gaussian filter).
    filter_sigma: Default value 1.5 (width of gaussian filter).
    k1: Default value 0.01
    k2: Default value 0.03 (SSIM is less sensitivity to K2 for lower values, so
      it would be better if we taken the values in range of 0< K2 <0.4).
    Returns:
    A tensor containing an MS-SSIM value for each image in batch.  The values
    are in range [0, 1].  Returns a tensor with shape:
    broadcast(img1.shape[:-3], img2.shape[:-3]).
    """

    # Convert to tensor if needed.
    img1 = tf.convert_to_tensor(img1, name='img1')
    img2 = tf.convert_to_tensor(img2, name='img2')
    # Shape checking.
    shape1, shape2 = outputs_flat_shape, outputs_flat_shape

    # Need to convert the images to float32.  Scale max_val accordingly so that
    # SSIM is computed correctly.
    max_val = tf.cast(max_val, dtype='float32')
    img1 = tf.cast(img1, dtype='float32')
    img2 = tf.cast(img2, dtype='float32')

    imgs = [img1, img2]
    shapes = [shape1, shape2]

    # img1 and img2 are assumed to be a (multi-dimensional) batch of
    # 3-dimensional images (height, width, channels). `heads` contain the batch
    # dimensions, and `tails` contain the image dimensions.
    heads = [s[:-3] for s in shapes]
    tails = [s[-3:] for s in shapes]

    divisor = [1, 2, 2, 1]
    divisor_tensor = tf.constant(divisor[1:], dtype="int32")

    def do_pad(images, remainder):
        padding = tf.expand_dims(remainder, -1)
        padding = tf.pad(padding, [[1, 0], [1, 0]])
        return [tf.pad(x, padding, mode='SYMMETRIC') for x in images]

    mcs = []
    for k in range(len(power_factors)):
        if k > 0:
            # Avg pool takes rank 4 tensors. Flatten leading dimensions.
            flat_imgs = [
                tf.reshape(x, tf.concat([[-1], t], 0))
                for x, t in zip(imgs, tails)
            ]

            remainder = tails[0] % divisor_tensor
            need_padding = tf.reduce_any(tf.not_equal(remainder, 0))
            # pylint: disable=cell-var-from-loop
            padded = tf.cond(need_padding,
                             lambda: do_pad(flat_imgs, remainder),
                             lambda: flat_imgs)
            # pylint: enable=cell-var-from-loop

            downscaled = [
                tf.nn.avg_pool(
                    x, ksize=divisor, strides=divisor, padding='VALID')
                for x in padded
            ]
            tails = [x[1:] for x in tf.shape_n(downscaled)]
            imgs = [
                tf.reshape(x, tf.concat([h, t], 0))
                for x, h, t in zip(downscaled, heads, tails)
            ]

        # Overwrite previous ssim value since we only need the last one.
        ssim_per_channel, cs = _ssim_per_channel(
            *imgs,
            max_val=max_val,
            filter_size=filter_size,
            filter_sigma=filter_sigma,
            k1=k1,
            k2=k2)
        mcs.append(tf.nn.relu(cs))

    # Remove the cs score for the last scale. In the MS-SSIM calculation,
    # we use the l(p) at the highest scale. l(p) * cs(p) is ssim(p).
    mcs.pop()  # Remove the cs score for the last scale.
    return tf.stack(mcs, axis=-1)

    # Take weighted geometric mean across the scale axis.
    #ms_ssim = tf.reduce_prod(tf.pow(mcs_and_ssim, power_factors), [-1])

    #return tf.reduce_mean(ms_ssim, [-1])  # Avg over color channels.

def _fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function."""
    size = tf.convert_to_tensor(size, dtype='int32')
    sigma = tf.convert_to_tensor(sigma)

    coords = tf.cast(tf.range(size), sigma.dtype)
    coords -= tf.cast(size - 1, sigma.dtype) / 2.0

    g = tf.square(coords)
    g *= -0.5 / tf.square(sigma)

    g = tf.reshape(g, shape=[1, -1]) + tf.reshape(g, shape=[-1, 1])
    g = tf.reshape(g, shape=[1, -1])  # For tf.nn.softmax().
    g = tf.nn.softmax(g)
    return tf.reshape(g, shape=[size, size, 1, 1])


def _ssim_per_channel(img1,
                      img2,
                      max_val=1.0,
                      filter_size=11,
                      filter_sigma=1.5,
                      k1=0.01,
                      k2=0.03):
    """Computes SSIM index between img1 and img2 per color channel.
    This function matches the standard SSIM implementation from:
    Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004). Image
    quality assessment: from error visibility to structural similarity. IEEE
    transactions on image processing.
    Details:
    - 11x11 Gaussian filter of width 1.5 is used.
    - k1 = 0.01, k2 = 0.03 as in the 01 paper.
    Args:
    img1: First image batch.
    img2: Second image batch.
    max_val: The dynamic range of the images (i.e., the difference between the
      maximum the and minimum allowed values).
    filter_size: Default value 11 (size of gaussian filter).
    filter_sigma: Default value 1.5 (width of gaussian filter).
    k1: Default value 0.01
    k2: Default value 0.03 (SSIM is less sensitivity to K2 for lower values, so
      it would be better if we taken the values in range of 0< K2 <0.4).
    Returns:
    A pair of tensors containing and channel-wise SSIM and contrast-structure
    values. The shape is [..., channels].
    """
    filter_size = tf.constant(filter_size, dtype='int32')
    filter_sigma = tf.constant(filter_sigma, dtype=img1.dtype)

    shape1, shape2 = tf.shape(img1), tf.shape(img2)

    # TODO(sjhwang): Try to cache kernels and compensation factor.
    kernel = _fspecial_gauss(filter_size, filter_sigma)
    kernel = tf.tile(kernel, multiples=[1, 1, shape1[-1], 1])

    # The correct compensation factor is `1.0 - tf.reduce_sum(tf.square(kernel))`,
    # but to match MATLAB implementation of MS-SSIM, we use 1.0 instead.
    compensation = 1.0

    # TODO(sjhwang): Try FFT.
    # TODO(sjhwang): Gaussian kernel is separable in space. Consider applying
    #   1-by-n and n-by-1 Gaussain filters instead of an n-by-n filter.
    def reducer(x):
        shape = tf.shape(x)
        x = tf.reshape(x, shape=tf.concat([[-1], shape[-3:]], 0))
        y = tf.nn.depthwise_conv2d(x, kernel, strides=[1, 1, 1, 1], padding='VALID')
        return tf.reshape(
            y, tf.concat([shape[:-3], tf.shape(y)[1:]], 0))

    luminance, cs = _ssim_helper(img1, img2, reducer, max_val, compensation, k1,
                                 k2)

    # Average over the second and the third from the last: height, width.
    axes = tf.constant([-3, -2], dtype="int32")
    ssim_val = tf.reduce_mean(luminance * cs, axes)
    cs = tf.reduce_mean(cs, axes)
    return ssim_val, cs


def _ssim_helper(x, y, reducer, max_val, compensation=1.0, k1=0.01, k2=0.03):
    """Helper function for computing SSIM.
      SSIM estimates covariances with weighted sums.  The default parameters
      use a biased estimate of the covariance_regularizer:
      Suppose `reducer` is a weighted sum, then the mean estimators are
        \mu_x = \sum_i w_i x_i,
        \mu_y = \sum_i w_i y_i,
      where w_i's are the weighted-sum weights, and covariance_regularizer estimator is
        cov_{xy} = \sum_i w_i (x_i - \mu_x) (y_i - \mu_y)
      with assumption \sum_i w_i = 1. This covariance_regularizer estimator is biased, since
        E[cov_{xy}] = (1 - \sum_i w_i ^ 2) Cov(X, Y).
      For SSIM measure with unbiased covariance_regularizer estimators, pass as `compensation`
      argument (1 - \sum_i w_i ^ 2).
      Arguments:
        x: First set of images.
        y: Second set of images.
        reducer: Function that computes 'local' averages from set of images. For
          non-convolutional version, this is usually tf.reduce_mean(x, [1, 2]), and
          for convolutional version, this is usually tf.nn.avg_pool2d or
          tf.nn.conv2d with weighted-sum kernel.
        max_val: The dynamic range (i.e., the difference between the maximum
          possible allowed value and the minimum allowed value).
        compensation: Compensation factor. See above.
        k1: Default value 0.01
        k2: Default value 0.03 (SSIM is less sensitivity to K2 for lower values, so
          it would be better if we taken the values in range of 0< K2 <0.4).
      Returns:
        A pair containing the luminance measure, and the contrast-structure measure.
    """

    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2

    # SSIM luminance measure is
    # (2 * mu_x * mu_y + c1) / (mu_x ** 2 + mu_y ** 2 + c1).
    mean0 = reducer(x)
    mean1 = reducer(y)
    num0 = mean0 * mean1 * 2.0
    den0 = tf.square(mean0) + tf.square(mean1)
    luminance = (num0 + c1) / (den0 + c1)

    # SSIM contrast-structure measure is
    #   (2 * cov_{xy} + c2) / (cov_{xx} + cov_{yy} + c2).
    # Note that `reducer` is a weighted sum with weight w_k, \sum_i w_i = 1, then
    #   cov_{xy} = \sum_i w_i (x_i - \mu_x) (y_i - \mu_y)
    #          = \sum_i w_i x_i y_i - (\sum_i w_i x_i) (\sum_j w_j y_j).
    num1 = reducer(x * y) * 2.0
    den1 = reducer(tf.square(x) + tf.square(y))
    c2 *= compensation
    cs = (num1 - num0 + c2) / (den1 - den0 + c2)

    # SSIM score is the product of the luminance and contrast-structure measures.
    return luminance, cs
