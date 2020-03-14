import tensorflow as tf
EarlyStopping = lambda: tf.keras.callbacks.EarlyStopping(
                                                monitor='val_loss',
                                                min_delta=1e-12,
                                                patience=5,
                                                verbose=1,
                                                restore_best_weights=True
                                            )