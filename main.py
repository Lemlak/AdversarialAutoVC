import torch
import torch.utils.data as torch_data
import torch.nn.functional
import nn_model
import yaml
from tqdm import tqdm
import os
import soundfile as sf
from resemblyzer import VoiceEncoder, preprocess_wav
import my_feats
from collections import deque
from safe_gpu import safe_gpu
import numpy as np
import matplotlib.pyplot as plt

import argparse
from safe_gpu import safe_gpu

# get GPU
pl___ = safe_gpu.GPUOwner()


class Trainer:
    def __init__(self, hyper,
                 train_set: torch_data.Dataset,
                 val_set: torch_data.Dataset = None,
                 checkpoint=None,
                 output='./',):
        self.output = output
        self.logger = my_feats.prepare_directories_and_logger(my_feats.Logger, output_directory=self.output, plot_name='AutoVC-adv')
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        print(f'Using {self.device}.')
        self.checkpoint = checkpoint
        self.resemblyzer = VoiceEncoder().to(self.device)

        self.best_val = 10000
        self.hyper = hyper

        self.train_set = train_set
        self.val_set = val_set

        self.n_speakers = len(self.val_set)
        assert self.n_speakers == len(self.val_set)

        self.vocoder = torch.hub.load('descriptinc/melgan-neurips', 'load_melgan')

        # networks
        self.generator = None
        self.adv_classifier = None
        # optimizers
        self.optimizer_g = None
        self.optimizer_c = None
        # iteration counter
        self.iteration = 0

        self.id_loss = getattr(torch.nn, self.hyper['id_loss'])()
        self.code_loss = getattr(torch.nn, self.hyper['code_loss'])()
        self.class_loss = getattr(torch.nn, self.hyper['class_loss'])()
        self.mse_loss = torch.nn.MSELoss()

        self.init_network()


    def init_network(self):
        self.generator = nn_model.Generator(dim_neck=self.hyper['dim_neck'],
                                            dim_emb=self.hyper['dim_emb'],
                                            dim_pre=self.hyper['dim_pre'],
                                            freq=self.hyper['freq'])
        self.critic = nn_model.Critic()                                    
        self.adv_classifier = nn_model.TDNN(2*self.hyper['dim_neck'], self.n_speakers, hidden=self.hyper['dim_hidden'])
        self.adv_classifier2 = nn_model.TDNN(2*self.hyper['dim_neck'], self.n_speakers, hidden=self.hyper['dim_hidden'])
        self.adv_classifier3 = nn_model.TDNN(2*self.hyper['dim_neck'], self.n_speakers, hidden=self.hyper['dim_hidden'])
        copy_weights(self.adv_classifier, self.adv_classifier2)
        copy_weights(self.adv_classifier, self.adv_classifier3)

        self.optimizer_g = torch.optim.Adam(self.generator.parameters(),
                                            self.hyper['g_lr'],
                                            (self.hyper['beta1'], self.hyper['beta2']))
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(),0.0001,
                                            (0.5, 0.9))
        self.optimizer_c = torch.optim.AdamW(self.adv_classifier.parameters(),
                                            self.hyper['g_lr'],
                                            (self.hyper['beta1'], self.hyper['beta2']))
        self.optimizer_c2 = torch.optim.AdamW(self.adv_classifier2.parameters(),
                                            self.hyper['g_lr'],
                                            (self.hyper['beta1'], self.hyper['beta2']))
        self.optimizer_c3 = torch.optim.AdamW(self.adv_classifier3.parameters(),
                                            self.hyper['g_lr'],
                                            (self.hyper['beta1'], self.hyper['beta2']))

        if self.checkpoint:
            self.load_checkpoint(self.checkpoint)

    def train(self):
        # initialize all
        print(f'Starting training from iteration {self.iteration}.')
        # get dataloaders
        train_loader = my_feats.DataLoader(dataset=self.train_set,
                                             batch_size=self.hyper['batch_size'],
                                             shuffle=True,
                                             num_workers=self.hyper['num_workers'],
                                             drop_last=True,
                                             collate_fn=my_feats.melaudiocollate)

        val_loader = torch_data.DataLoader(self.val_set)
        self.generator.train().to(self.device)
        self.critic.train().to(self.device)
        self.adv_classifier.train().to(self.device)
        self.adv_classifier2.train().to(self.device)
        self.adv_classifier3.train().to(self.device)

        optimizer_to(self.optimizer_g, self.device)
        optimizer_to(self.optimizer_critic, self.device)
        optimizer_to(self.optimizer_c, self.device)
        optimizer_to(self.optimizer_c2, self.device)
        optimizer_to(self.optimizer_c3, self.device)

        train_iter = iter(train_loader)
        # run training
        for i in tqdm(range(self.iteration, self.hyper['num_iters']), desc='AutoVC adversarial'):
            # ----------------------------------
            # ------------- train --------------
            # ----------------------------------

            # -----------------------
            # ------ CLASSIFIER------
            # -----------------------
            for _ in range(self.hyper['g_critic']):
                try:
                    batch = next(train_iter)
                except TypeError:
                    train_iter = iter(train_loader)
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    batch = next(train_iter)
                cuda_batch = [x.to(self.device) for x in batch]
                mel_real, src_emb, src_id = cuda_batch
                trim = mel_real.size(2) - mel_real.size(2) % self.hyper['freq']
                mel_real_t = mel_real[:, :, :trim]

                
                self.optimizer_g.zero_grad()
                self.optimizer_critic.zero_grad()
                self.optimizer_c.zero_grad()
                self.optimizer_c2.zero_grad()
                self.optimizer_c3.zero_grad()
                g_out, g_out_postnet, g_codes, my_codes = self.generator(mel_real_t, src_emb, src_emb)
                
                # updates for adversarial classifiers
                c_logit = self.adv_classifier(my_codes.transpose(1,2).detach())
                c_loss = self.class_loss(c_logit, src_id)

                c_loss.backward(retain_graph=True)
                self.optimizer_c.step()
                prob1 = self.get_prob(c_logit, src_id)
                self.logger.log_training(iteration=i+1, src_prob1=prob1)

                c_logit2 = self.adv_classifier2(my_codes.transpose(1,2).detach())
                c_loss2 = self.class_loss(c_logit2, src_id)

                c_loss2.backward(retain_graph=True)
                self.optimizer_c2.step()
                prob2 = self.get_prob(c_logit2, src_id)
                self.logger.log_training(iteration=i+1, src_prob2=prob2)  
                
                c_logit3 = self.adv_classifier3(my_codes.transpose(1,2).detach())
                c_loss3 = self.class_loss(c_logit3, src_id)

                c_loss3.backward(retain_graph=True)
                self.optimizer_c3.step()
                prob3 = self.get_prob(c_logit3, src_id)
                self.logger.log_training(iteration=i+1, src_prob3=prob3)  
                
                
                # GAN augmentations
                time_prob = np.random.rand()
                freq_prob = np.random.rand()
                noise_prob = np.random.rand()
                net_real_input = mel_real_t
                net_fake_input = g_out_postnet
                if time_prob < 0.15:
                    mask = torch.ones_like(mel_real_t)
                    t = 80
                    l = np.random.randint(t//4)
                    start = np.random.randint(0, t-l)
                    mask[:, start:start+l, :] = torch.zeros((mask.size(0), l, mask.size(2)))
                    net_real_input = net_real_input*mask
                    net_fake_input = net_fake_input*mask
                if freq_prob < 0.15:
                    mask = torch.ones_like(mel_real_t)
                    t = mel_real_t.size(2)
                    l = np.random.randint(t//4)
                    start = np.random.randint(0, t-l)
                    mask[:, :, start:start+l] = torch.zeros((mask.size(0), mask.size(1), l))
                    net_real_input = net_real_input*mask
                    net_fake_input = net_fake_input*mask
                if noise_prob < 0.1:
                    mask = torch.randn(mel_real_t.size(), device=self.device)*0.001
                    net_real_input = torch.log10(torch.clamp(torch.pow(10, net_real_input) + mask, min=1e-5))
                    net_fake_input = torch.log10(torch.clamp(torch.pow(10, net_fake_input) + mask, min=1e-5))                

                c_fake_out = self.critic(net_fake_input.detach()).squeeze()
                c_real_out = self.critic(net_real_input).squeeze()
                c_fake_loss = torch.mean(c_fake_out)
                c_real_loss = torch.mean(c_real_out)
                crit_loss = c_fake_loss - c_real_loss + calculate_gradient_penalty(self.critic, net_real_input.data, net_fake_input.data, self.device)
                self.logger.log_training(iteration=i+1, crit_loss=crit_loss)  
                crit_loss.backward()
                self.optimizer_critic.step()
                
                
            # -----------------------
            # ------ GENERATOR ------
            # -----------------------

            # cycle consistency loss L_cyc
            g_loss_id = self.id_loss(mel_real_t, g_out)
            g_loss_id_psnt = self.id_loss(mel_real_t, g_out_postnet)

            # Code semantic loss.
            g_code_reconst = self.generator(g_out_postnet, src_emb, None)
            g_loss_cd = self.code_loss(g_codes, g_code_reconst)

            # Adversarial loss
            g_logit = self.adv_classifier(my_codes.transpose(1,2))
            g_loss_adv = - prob2/(1-prob2) * self.class_loss(g_logit, src_id).clamp(-10,10)
            
            # Total loss
            g_loss = g_loss_id + 10*g_loss_id_psnt + self.hyper['lambda_id'] * g_loss_cd + g_loss_adv
            
            # GAN loss
            gc_fake_out = self.critic(net_fake_input).squeeze()
            g_fake_loss = - 0.1*torch.mean(gc_fake_out)
            g_loss += g_fake_loss

            # Backward and optimize.

            g_loss.backward()

            self.optimizer_g.step()

            if (i + 1) % self.hyper['log_step'] == 0:
                self.logger.log_training(iteration=i+1,
                                         g_loss=g_loss.item(),
                                         g_loss_id=g_loss_id.item(),
                                         g_loss_id_psnt=g_loss_id_psnt.item(),
                                         g_loss_cd=g_loss_cd.item(),
                                         g_loss_adv=g_loss_adv.item(),
                                         g_loss_gan=g_fake_loss.item())

            if (i+1) % self.hyper['test_step'] == 0 or i == 0:
                self.test_step(val_loader, i)

            if (i + 1) % self.hyper['model_save_step'] == 0 or i == 0:
                self.save_checkpoint(i+1)

    def get_prob(self, logit, ids):
        softmax_out = torch.nn.functional.softmax(logit.detach())
        spk_probs = softmax_out[np.arange(self.hyper['batch_size']), ids]
        prob = torch.exp(torch.mean(torch.log(spk_probs))).item()
        return prob
                
    def test_step(self, val_loader, i):
        self.generator.eval()
        with torch.no_grad():
            with tqdm(total=len(val_loader)) as progress_bar:
                val_loss = []
                for val_mel_real, val_audio_real, _ in val_loader:
                    val_mel_real = val_mel_real.to(self.device)
                    val_emb_real = val_audio_real.to(self.device)
                    trim = val_mel_real.size(2) - val_mel_real.size(2) % self.hyper['freq']
                    val_mel_real = val_mel_real[:, :, :trim]

                    _, val_mel_fake, _, _ = self.generator(val_mel_real, val_emb_real, val_emb_real)
                    val_loss.append(self.id_loss(val_mel_fake, val_mel_real).item())
                    progress_bar.update(1)  # update progress
        val_loss_mean = np.mean(val_loss)
        if val_loss_mean < self.best_val:
            print(f'Saving new best model val_loss={val_loss_mean} ({self.best_val})')
            self.best_val = val_loss_mean
            self.save_checkpoint('best_model')
        self.logger.log_validation(iteration=i + 1,
                                   accuracy=("scalars", None,
                                             {'val_loss': val_loss_mean}))

        test_mels_dir = '/mnt/matylda6/ibrukner/code/vctk_speakers/mels'
        test_mels = {}
        test_embs = {}
        for f, l in [("p225_001", 'unseen_F'),
                     ("p226_001", 'unseen_M'),
                     ("p233_001", 'seen_F'),
                     ("p270_001", 'seen_M')]:
            tmp = torch.from_numpy(np.load(os.path.join(test_mels_dir, f'{f}_mic1.npy'))).to(self.device).unsqueeze(0)
            pad = self.hyper['freq'] - tmp.size(2) % self.hyper['freq']
            tmp = torch.nn.functional.pad(tmp, (0, pad), value=1e-5)

            test_mels[f'{f}_{l}'] = tmp
            test_embs[f'{f}_{l}'] = torch.from_numpy(self.resemblyzer.embed_utterance(preprocess_wav(os.path.join(test_mels_dir, f'{f}_mic1.flac')))).to(self.device).unsqueeze(0)

        test_mel_fakes = {}
        test_converted = {}
        test_origs = {}
        for src in test_mels.keys():
            test_mel_fakes[f'{src}'] = {}
            test_converted[f'{src}'] = {}
            test_origs[f'{src}'] = self.vocoder.inverse(test_mels[f'{src}']).squeeze().detach().cpu().numpy()
            for tar in test_mels.keys():
                _, test_mel_fakes[f'{src}'][f'{tar}'], _, _ = self.generator(test_mels[src],
                                                                test_embs[src],
                                                                test_embs[tar])
                test_converted[f'{src}'][f'{tar}'] = \
                    self.vocoder.inverse(test_mel_fakes[f'{src}'][f'{tar}']).squeeze().detach().cpu().numpy()

        fig, ax = plt.subplots(nrows=4, ncols=5, figsize=(25, 16))
        for ii, src in enumerate(test_mels.keys()):
            # show spectrogram
            im = ax[ii, 0].imshow(test_mels[src].cpu().squeeze().detach().numpy(),
                                  aspect="auto", origin="lower", interpolation='none')
            plt.colorbar(im, ax=ax[ii, 0])
            ax[ii, 0].set_xlabel("Frames")
            ax[ii, 0].set_ylabel("Channels")
            ax[ii, 0].set_title(f'{src}')
            # add audio
            self.logger.add_audio(
                src,
                test_origs[src], global_step=i + 1, sample_rate=22050)
            for jj, tar in enumerate(test_mels.keys()):
                # show spectrogram
                im = ax[ii, jj + 1].imshow(test_mel_fakes[src][tar].cpu().squeeze().detach().numpy(),
                                           aspect="auto", origin="lower", interpolation='none')
                plt.colorbar(im, ax=ax[ii, jj + 1])
                ax[ii, jj + 1].set_xlabel("Frames")
                ax[ii, jj + 1].set_ylabel("Channels")
                ax[ii, jj + 1].set_title(f'{src}_{tar}')
                # add audio
                self.logger.add_audio(
                    f'{src}_{tar}',
                    test_converted[src][tar], global_step=i + 1, sample_rate=22050)
        fig.tight_layout()
        fig.canvas.draw()
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close()
        self.logger.add_image(
            "Spectrograms",
            data,
            i + 1, dataformats='HWC')

        self.logger.close()
        self.generator.train()

    def load_checkpoint(self, checkpoint_path):
        assert os.path.isfile(checkpoint_path)
        print("Loading checkpoint '{}'".format(checkpoint_path))
        checkpoint_dict = torch.load(checkpoint_path)
        self.generator.load_state_dict(checkpoint_dict['g_state_dict'])
        self.optimizer_g.load_state_dict(checkpoint_dict['optimizer_g'])
        optimizer_to(self.optimizer_g, self.device)
        
        self.critic.load_state_dict(checkpoint_dict['crit_state_dict'])
        self.optimizer_critic.load_state_dict(checkpoint_dict['optimizer_crit'])
        optimizer_to(self.optimizer_critic, self.device)
        
        self.adv_classifier.load_state_dict(checkpoint_dict['c_state_dict'])
        self.optimizer_c.load_state_dict(checkpoint_dict['optimizer_c'])
        optimizer_to(self.optimizer_c, self.device)
        
        self.adv_classifier2.load_state_dict(checkpoint_dict['c2_state_dict'])
        self.optimizer_c2.load_state_dict(checkpoint_dict['optimizer_c2'])
        optimizer_to(self.optimizer_c2, self.device)
        
        self.adv_classifier3.load_state_dict(checkpoint_dict['c3_state_dict'])
        self.optimizer_c3.load_state_dict(checkpoint_dict['optimizer_c3'])
        optimizer_to(self.optimizer_c3, self.device)
        
        self.iteration = checkpoint_dict['iteration']
        print("Loaded checkpoint '{}' from iteration {}".format(checkpoint_path, self.iteration))

    def save_checkpoint(self, iteration):
        filepath = os.path.join(self.output, f'{iteration}.ckpt')
        print("Saving model and optimizer state at iteration {} to {}".format(
            iteration, filepath))
        fp = '/' + '/'.join(filepath.split('/')[:-1])
        if os.path.isdir(fp) is False:
            os.mkdir(fp)
        torch.save({'iteration': iteration,
                    'g_state_dict': self.generator.state_dict(),
                    'crit_state_dict': self.critic.state_dict(),
                    'c_state_dict': self.adv_classifier.state_dict(),
                    'c2_state_dict': self.adv_classifier2.state_dict(),
                    'c3_state_dict': self.adv_classifier3.state_dict(),
                    'optimizer_g': self.optimizer_g.state_dict(),
                    'optimizer_crit': self.optimizer_critic.state_dict(),
                    'optimizer_c': self.optimizer_c.state_dict(),
                    'optimizer_c2': self.optimizer_c2.state_dict(),
                    'optimizer_c3': self.optimizer_c3.state_dict(),
                    }, filepath)


def optimizer_to(optimizer, device):
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

def insert_module(model, indices, modules):
    indices = indices if isinstance(indices, list) else [indices]
    modules = modules if isinstance(modules, list) else [modules]
    assert len(indices) == len(modules)

    layers_name = [name for name, _ in model.named_modules()][1:]
    for index, module in zip(indices, modules):
        layer_name = re.sub(r'(.)(\d)', r'[\2]', layers_name[index])
        exec("model.{name} = nn.Sequential(model.{name}, module)".format(name = layer_name))


def copy_weights(from_model: torch.nn.Module, to_model: torch.nn.Module):
    """Copies the weights from one model to another model.

    # Arguments:
        from_model: Model from which to source weights
        to_model: Model which will receive weights
    """
    if not from_model.__class__ == to_model.__class__:
        raise(ValueError("Models don't have the same architecture!"))

    for m_from, m_to in zip(from_model.modules(), to_model.modules()):
        is_linear = isinstance(m_to, torch.nn.Linear)
        is_conv = isinstance(m_to, torch.nn.Conv2d)
        is_bn = isinstance(m_to, torch.nn.BatchNorm2d)
        if is_linear or is_conv or is_bn:
            m_to.weight.data = m_from.weight.data.clone()
            if m_to.bias is not None:
                m_to.bias.data = m_from.bias.data.clone() 


def calculate_gradient_penalty(model, real, fake, device):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake data
    alpha = torch.randn((real.size(0), 1, 1), device=device)
    # Get random interpolation between real and fake data
    interpolates = (alpha * real + ((1 - alpha) * fake)).requires_grad_(True)

    model_interpolates = model(interpolates)
    grad_outputs = torch.ones(model_interpolates.size(), device=device, requires_grad=False)

    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=model_interpolates,
        inputs=interpolates,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    return torch.mean((gradients.norm(2, dim=1) - 1) ** 2)



def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default=None, help='Configuration conf.yaml file.', required=True)
    parser.add_argument('--check', type=str, default=None, help='Checkpoint file.', required=False)
    parser.add_argument('--dataset', type=str, default=None, help='Dataset folder.', required=True)
    parser.add_argument('--output', type=str, default='./', help='Output folder.', required=False)
    parser.add_argument('--cont', action='store_true', default=False, help='Continue training.', required=False)
    parser.add_argument('--pre_class', type=str, default=None, help='Pretrained classifier.', required=False)
    config = parser.parse_args()
    return config


if __name__ == '__main__':
    conf = parse()
    train_data = my_feats.MelEmbUtterancesAll(conf.dataset, train=True)
    val_data = my_feats.MelAudioUtterances(conf.dataset, val=True)
    config = os.path.join(conf.conf, 'config.yaml')
    if conf.pre_class:
        files = next(os.walk(conf.pre_class))[2]
        conf.check = os.path.join(conf.pre_class, f'{str(max(sorted([int(f.split(".")[0]) if f.isdigit() else 0 for f in files])))}.ckpt')

    if conf.cont:
        files = next(os.walk(conf.output))[2]
        conf.check = os.path.join(conf.output, f'{str(max(sorted([int(f.split(".")[0]) if (f.split(".")[0]).isdigit() else 0 for f in files])))}.ckpt')
        config = os.path.join(conf.output, 'config.yaml')
    with open(config, 'r') as f:
        hyper = yaml.load(f, Loader=yaml.Loader)
    delim = '_'
    if not 'model_type' in hyper:
        hyper["model_type"] = ''
        delim = ''
    trainer = Trainer(hyper, train_data,
                             val_set=val_data,
                             checkpoint=conf.check,
                             output=conf.output)
    trainer.train()


