import tensorflow as tf
from utils.reporting.ploting import plot_and_save_generated
class SampleGeneration(tf.keras.callbacks.Callback):
    def __init__(
            self,
            filepath,
            gen_freq=5,
            save_img=False,
            random_latent=None,
            latent_shape=50,
            gray_plot=False,
            **kws
    ):
        self.gen_freq = gen_freq
        self.filepath = filepath
        self.save_img = save_img
        self.gray_plot = gray_plot

        if random_latent is None:
            self.random_latent = tf.random.normal(shape=[50, latent_shape])
        else:
            self.random_latent = random_latent

        tf.keras.callbacks.Callback.__init__(self, **kws)

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.gen_freq == 0:  # or save after some epoch, each k-th epoch etc.
            generated = self.model.generate_sample(model=self.model.get_variable,
                                                   inputs_shape=self.model.inputs_shape,
                                                   latent_shape=[50, self.model.latent_dim],
                                                   eps=self.random_latent)
            plot_and_save_generated(generated=generated, epoch=epoch, path=self.filepath, gray=self.gray_plot,
                                    save=self.save_img)
