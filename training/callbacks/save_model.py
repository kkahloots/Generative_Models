import tensorflow as tf
class ModelSaver(tf.keras.callbacks.Callback):
    def __init__(self, filepath, save_freq=5, **kws):
        self.save_freq = save_freq
        self.filepath = filepath
        tf.keras.callbacks.Callback.__init__(self, **kws)

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.save_freq == 0:  # or save after some epoch, each k-th epoch etc.
            self.model.save(self.filepath)