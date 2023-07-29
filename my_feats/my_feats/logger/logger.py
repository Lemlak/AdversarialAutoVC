import torch
from tensorboardX import SummaryWriter
from .plotting_utils import  plot_spectrogram_to_numpy, plot_onset_to_numpy


class Logger(SummaryWriter):
    def __init__(self, logdir, plot_name):
        super(Logger, self).__init__(logdir)
        self.plot_name = plot_name

    def log_training(self, iteration, **kwarg):
        self.add_scalars(self.plot_name, kwarg, iteration)
        
    def log_validation(self, iteration, **kwarg):
        for key in (kwarg.keys()):
            (type_, method_, data) = kwarg[key]
            
            if type_ is "audio":
                self.add_audio(
                f'{key}',
                data, iteration, sample_rate=method_)
            elif type_ == "scalars":
                self.add_scalars("validation.loss", data, iteration)
            elif type_ == "image":
                data = data
                self.add_image(
                f'{key}',
                method_(data),
                iteration, dataformats='HWC')