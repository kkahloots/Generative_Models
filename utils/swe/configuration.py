from types import FunctionType
from utils.swe.codes import properties

class default_config:
    # Model Configuration
    model_name = ''

    # Graph Configuration
    lay_dim = 50
    num_lay = 4
    batch_size = 50
    latents_dim = 50
    validation_pct = 20
    input_shape = (28, 28, 1)
    isConv=False
    kernel_shape=3
    sampling_rate=2
    addBatchNorm=True
    addDropout=True

    # Training Configuration
    L2_alpha = 1e-5
    steps_per_epoch = 500
    validation_step = 100
    valid_file_formats = frozenset(['jpg', 'jpeg', 'JPG', 'JPEG', 'png', 'PNG'])
    max_num_example_per_class = 2 ** 27 - 1  # ~134M

    test_epoch = 5
    save_epoch = 5

    # example Configuration
    example_dim = None
    example_dir = None
    learning_rate = 1e-3

class Config:
    def __init__(self):
        keys = list()
        items = list()
        ddict = dict(default_config.__dict__)
        for key, item in ddict.items():
            if key in properties(default_config):
                keys.append(key)
                items.append(item)
        ddict =  dict(zip(keys, items))
        for k, v in ddict.items():
            setattr(self, k, v)

    def __getitem__(self, item):
        return self.__dict__[item]

    def __setitem__(self, item, value):
        self.__dict__[item] = value

    def update(self, new_vals):
        self.__dict__.update(new_vals)

    def keys(self):
        keys = list()
        for key, item in self.__dict__.items():
            if type(item) != FunctionType:
                keys.append(key)
        return keys

    def dict(self):
        keys = list()
        items = list()
        for key, item in self.__dict__.items():
            if type(item) != FunctionType:
                keys.append(key)
                items.append(item)
        return dict(zip(keys, items))


