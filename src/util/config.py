class Config(object):
    def __init__(self, epoch=500, batch=32, n_mfcc=20, sr=None, num_seg=5):
        self._epoch = epoch
        self._batch = batch
        self._n_mfcc = n_mfcc
        self._sr = sr
        self._num_seg = num_seg

    @property
    def num_seg(self):
        return self._num_seg

    @property
    def sr(self):
        return self._sr

    @property
    def n_mfcc(self):
        return self._n_mfcc