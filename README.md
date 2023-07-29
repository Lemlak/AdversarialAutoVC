# AdversarialAutoVC

This is code for training AutoVC model with adversarial classifier for better disentanglement and WGAN for better naturalness. 

Also, this model uses MelGAN vocoder instead of WaveNet.

Ideas are summed up in attached PDF.

## How to
First, generate mel-spectrogram features using `Audio2Mel` from `my_feats` package.

You can run training following example in `sg_script.sh`
