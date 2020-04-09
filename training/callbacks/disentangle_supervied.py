
import tensorflow as tf

from evaluation.supervised_metrics.disentangle_api import supervised_metrics
from utils.data_and_files.file_utils import log


class DisentanglementSuperviedMetrics(tf.keras.callbacks.Callback):
    def __init__(
            self,
            ground_truth_data,
            representation_fn,
            random_state,
            file_Name,
            num_train=1000,
            num_test=200,
            batch_size=32,
            continuous_factors=False,
            gt_freq=25,
            **kws
    ):
        self.gt_data = ground_truth_data
        self.representation_fn = representation_fn
        self.random_state = random_state
        self.file_Name = file_Name
        self.num_train = num_train
        self.num_test = num_test
        self.batch_size = batch_size
        self.continuous_factors = continuous_factors
        self.gt_freq = gt_freq
        tf.keras.callbacks.Callback.__init__(self, **kws)

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.gt_freq == 0:  # or save after some epoch, each k-th epoch etc.
            s_scores = supervised_metrics(
                ground_truth_data=self.gt_data,
                representation_fn=self.representation_fn,
                random_state=self.random_state,
                num_train=self.num_train,
                num_test=self.num_test,
                continuous_factors=self.continuous_factors,
                batch_size=self.batch_size
            )
            gt_metrics = {'Epoch': epoch, **s_scores}
            log(file_name=self.file_Name, message=dict(gt_metrics))
