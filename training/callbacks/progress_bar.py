from keras_tqdm import TQDMNotebookCallback

class NotebookPrograssBar(TQDMNotebookCallback):
    def on_train_batch_begin(self, x, y=None):
        return None

    def on_train_batch_end(self, x, y=None):
        return None

    def on_test_batch_begin(self, x, y=None):
        return None

    def on_test_batch_end(self, x, y=None):
        return None

    def on_test_begin(self, x, y=None):
        return None

    def on_test_end(self, x, y=None):
        return None

