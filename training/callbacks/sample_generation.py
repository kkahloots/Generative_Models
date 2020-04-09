import tensorflow as tf
from utils.reporting.ploting import plot_and_save_generated
class SampleGeneration(tf.keras.callbacks.Callback):
    def __init__(
            self,
            filepath,
            gen_freq=5,
            save_img=False,
            random_latents=None,
            latents_shape=50,
            gray_plot=False,
            **kws
    ):
        self.gen_freq = gen_freq
        self.filepath = filepath
        self.save_img = save_img
        self.gray_plot = gray_plot

        if random_latents is None:
            self.random_latents = tf.random.normal(shape=[50, latents_shape])
        else:
            self.random_latents = random_latents

        tf.keras.callbacks.Callback.__init__(self, **kws)

    def on_train_end(self, logs=None):
        self.generate(-999, logs)

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.gen_freq == 0:  # or save after some epoch, each k-th epoch etc.
            self.generate(epoch, logs)

    def generate(self, epoch, logs={}):
        generated = self.model.generate_sample(model=self.model.get_variable,
                                               input_shape=self.model.get_inputs_shape(),
                                               latents_shape=[50, self.model.latents_dim],
                                               eps=self.random_latents)
        plot_and_save_generated(generated=generated, epoch=epoch, path=self.filepath, gray=self.gray_plot,
                                save=self.save_img)
