import tensorflow as tf
class ModelSaver(tf.keras.callbacks.ModelCheckpoint):
    def __init__(self, filepath):
        tf.keras.callbacks.ModelCheckpoint.__init__(self,
                                                    filepath= filepath,
                                                    monitor='loss',
                                                    verbose=1,
                                                    save_weights_only=False,
                                                    save_best_only=True
                                                    )
