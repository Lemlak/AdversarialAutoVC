import torch
import numpy as np
import os
import omegaconf as OmegaConf

def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)
    
def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C

def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)

def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C

def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output
    
def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output


mel_basis = {}
hann_window = {}

def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    #if torch.min(y) < -1.:
    #        print('min value is ', torch.min(y))
    #if torch.max(y) > 1.:
    #    print('max value is ', torch.max(y))
    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis[str(fmax)+'_'+str(y.device)] = torch.from_numpy(mel).float().to(y.device)      
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)
        
    spec = torch.stft(y, n_fft=n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)], 
                      center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=False)

    spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))

    spec = torch.matmul(mel_basis[str(fmax)+'_'+str(y.device)], spec)
    spec = spectral_normalize_torch(spec)

    return spec



class HifiGAN():
    def __init__(self):
        super().__init__()
        path_config = os.path.join(os.path.dirname(os.path.abspath(__file__)), './configs/hifi-gan/UNIVERSAL_V1/config.json')
        path_ckpt = os.path.join(os.path.dirname(os.path.abspath(__file__)), './configs/hifi-gan/UNIVERSAL_V1/g_02500000')
        hifigan_config = OmegaConf.load(path_config)
        self.vocoder = hifigan_vocoder(hifigan_config)
        state_dict_g = torch.load(path_ckpt)
        self.vocoder.load_state_dict(state_dict_g['generator'])
        self.vocoder.eval()
         
        for param in self.vocoder.parameters():
            param.requires_grad = False
         
    def get_mel(self, x):
        return mel_spectrogram(x, 1024, 80, 22050, 256, 1024, 0, 8000)

            
    def vocode(self, mels):
        if len(mels.shape) == 2:
            mels = mels.unsqueeze(0)
        return self.vocoder(mels).squeeze()

    def extract(self, x):
        return self.get_mel(x)
        

