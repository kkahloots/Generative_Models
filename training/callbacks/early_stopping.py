import tensorflow as tf
EarlyStopping = lambda: tf.keras.callbacks.EarlyStopping(
                                                monitor='loss',
                                                min_delta=1e-12,
                                                patience=50,
                                                verbose=1,
                                                restore_best_weights=False
                                            )