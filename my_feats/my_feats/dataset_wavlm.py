from torch.utils import data
import torch
import numpy as np
import os
import librosa
import my_feats
from resemblyzer import VoiceEncoder, preprocess_wav
import soundfile as sf
from tqdm import tqdm

class DataLoader(data.DataLoader):
    def __init__(self, *args, **kwargs):
        print("Using awesome custom dataloader")
        super().__init__(*args, **kwargs, worker_init_fn=self.worker_init_fn)

    @staticmethod
    def worker_init_fn(worker_id):
        np.random.seed(np.random.get_state()[1][0] + worker_id)


def dummy_silence_removal(mel, thresh=1200):
    energy = np.sum(mel ** 2, axis=0)
    silence = energy > thresh
    return torch.from_numpy(np.delete(mel, np.nonzero(silence), axis=1))


class TestUtterances(data.Dataset):
    """Dataset class for the Utterances dataset."""

    def __init__(self, data_dir, return_wav_name=False):
        """Initialize and preprocess the Utterances dataset."""
        np.random.seed(42)
        self.dataset_type = "test"
        self.return_wav_name = return_wav_name
        """Initialize and preprocess the Utterances dataset."""
        self.data_dir = data_dir
        self.wavs = self.load_scp(os.path.join(self.data_dir, f"{self.dataset_type}_wav.scp"))
        self.utt2spk = self.load_scp(os.path.join(self.data_dir, f"{self.dataset_type}_utt2spk"))
        self.utt2mel = self.load_scp(os.path.join(self.data_dir, f"{self.dataset_type}_utt2mel"))
        self.spk2utt = self.load_scp(os.path.join(self.data_dir, f"{self.dataset_type}_spk2utt"))
        self.train_spks = list(self.load_scp(os.path.join(self.data_dir, f"train_spk2utt")).keys())
        self.utts = list(self.utt2mel.keys())
        self.len_utts = len(self.utts)
        print(f'Finished loading {self.dataset_type} dataset with {len(self.utts)} utterances.')

    def load_scp(self, scp):
        ret = {}
        with open(scp, "r") as f:
            for line in f:
                col1, col2 = line.strip().split(" ")
                if col1 in ret:
                    if not type(ret[col1]) is list:
                        tmp = ret[col1]
                        ret[col1] = [tmp]
                    ret[col1].append(col2)
                else:
                    ret[col1] = col2
        return ret

    def __getitem__(self, index):
        utt = self.utts[index]
        wav = self.wavs[utt]
        wav_name = wav
        wav = librosa.load(wav, sr=22050)[0]
        if self.return_wav_name:
            return wav, self.label_to_id(self.utt2spk[utt]), utt, wav_name
        return wav, self.label_to_id(self.utt2spk[utt]), utt

    def __len__(self):
        """Return the number of spkrs."""
        return self.len_utts

    def label_to_id(self, spk_label):
        if spk_label:
            try:
                if isinstance(spk_label, list):
                    return torch.Tensor([int(self.train_spks.index(i)) for i in spk_label])
                else:
                    return int(self.train_spks.index(spk_label))
            except ValueError:
                return -1
        else:
            return -1

    def id_to_label(self, spk_id):
        if isinstance(spk_id, torch.Tensor):
            return [self.train_spks[i] for i in spk_id]
        else:
            return self.train_spks[spk_id]


class MyCollator(object):
    def __init__(self, freq):
        self.freq = freq
        self.mel_extractor = my_feats.Audio2Mel()

    def __call__(self, batch):
        maxx = max(item[0].shape[0] for item in batch)
        maxx += (self.freq * 256) - (maxx % (self.freq * 256))

        mels = torch.zeros((len(batch), 80, maxx // 256))
        spk_ids = []
        pads = []
        for i in range(len(batch)):
            wav, _id, _ = batch[i]
            wav = np.pad(wav, (0, maxx - wav.shape[0]))
            pads.append(maxx - wav.shape[0])
            print(_id, maxx - wav.shape[0], wav.shape[0])
            mel = torch.from_numpy(mel_extractor(wav, trim=False))
            spk_ids.append(_id)
            mels[i, :, :mel.size(1)] = mel

        if mels.size(2) % self.freq > 0:
            print('chyba')
            mels = mels[:, :, :-(mels.size(2) % 4)]
        return mels, torch.tensor(np.array(spk_ids)), [item[2] for item in batch], pads

class MyCollatorWav(object):
    def __init__(self, freq):
        self.freq = freq
        self.mel_extractor = my_feats.Audio2Mel()

    def __call__(self, batch):
        maxx = max(item[0].shape[0] for item in batch)
        maxx += (self.freq * 256) - (maxx % (self.freq * 256))

        mels = torch.zeros((len(batch), 80, maxx // 256))
        wavs = []
        spk_ids = []
        pads = []
        for i in range(len(batch)):
            wav, _id, _, wav_name = batch[i]
            wav = np.pad(wav, (0, maxx - wav.shape[0]))
            wavs.append(wav_name)
            pads.append(maxx - wav.shape[0])
            #print(_id, maxx - wav.shape[0], wav.shape[0])
            mel = torch.from_numpy(mel_extractor(wav, trim=False))
            spk_ids.append(_id)
            mels[i, :, :mel.size(1)] = mel

        if mels.size(2) % self.freq > 0:
            print('chyba')
            mels = mels[:, :, :-(mels.size(2) % 4)]
        return mels, torch.tensor(np.array(spk_ids)), [item[2] for item in batch], pads, wavs


mel_extractor = my_feats.Audio2Mel()

def no_trim_collate(batch):
    # batch is [(mel_spec(T, mel), spk_id)]
    maxx = max(item[0].shape[0] for item in batch)
    maxx += 4*256  # 4 silence frames to enforce full utterance

    mels = torch.zeros((len(batch), 80, maxx//256))
    spk_ids = []
    for i in range(len(batch)):
        wav, _id, _ = batch[i]
        wav = np.pad(wav, (0, maxx - wav.shape[0]))
        mel = torch.from_numpy(mel_extractor(wav, trim=False))
        spk_ids.append(_id)
        mels[i, :, :mel.size(1)] = mel

    if mels.size(2) % 4 > 0:
        mels = mels[:, :, :-(mels.size(2) % 4)]
    return mels, torch.tensor(np.array(spk_ids)), [item[2] for item in batch]


class UtterancesWavlm(data.Dataset):
    """Dataset class for the Utterances dataset."""

    def __init__(self, data_dir,
                 val=False,
                 test=False,
                 train=False,
                 tar_separate=False,
                 tar_multiplier=4):
        """Initialize and preprocess the Utterances dataset."""
        self.tar_separate = tar_separate
        self.tar_multiplier = tar_multiplier
        np.random.seed(42)
        self.dataset_type = "train" if train else "test" if test else "val"
        """Initialize and preprocess the Utterances dataset."""
        self.data_dir = data_dir
        self.wavs = self.load_scp(os.path.join(self.data_dir, f"{self.dataset_type}_wav.scp"))
        self.utt2spk = self.load_scp(os.path.join(self.data_dir, f"{self.dataset_type}_utt2spk"))
        self.utt2mel = self.load_scp(os.path.join(self.data_dir, f"{self.dataset_type}_utt2mel"))
        self.utt2wavlm = self.load_scp(os.path.join(self.data_dir, f"{self.dataset_type}_utt2wavlm"))
        self.spk2utt = self.load_scp(os.path.join(self.data_dir, f"{self.dataset_type}_spk2utt"))
        self.spks = list(self.spk2utt.keys())
        self.utts = list(self.utt2spk.keys())
        self.len_spks = len(self.spks)
        self.len_utts = len(self.utts)
        print(f'Finished loading {self.dataset_type} dataset with {len(self.spks)} speakers.')

    def load_scp(self, scp):
        ret = {}
        with open(scp, "r") as f:
            for line in f:
                col1, col2 = line.strip().split(" ")
                if col1 in ret:
                    if not type(ret[col1]) is list:
                        tmp = ret[col1]
                        ret[col1] = [tmp]
                    ret[col1].append(col2)
                else:
                    ret[col1] = col2
        return ret

    def __getitem__(self, index):
        utt = self.utts[index]
        # pick a random utterance
        spk = self.spks.index(self.utt2spk[utt])
        mel = torch.load(self.utt2mel[utt]).squeeze()
        wavlm = torch.load(self.utt2wavlm[utt]).permute((2,1,0))
        if self.tar_separate:
            utts = self.spk2utt[spk]
            rand_utts = np.random.choice(utts, self.tar_multiplier, replace=False)
            
            mel_tar_utts = [self.utt2mel[i] for i in rand_utts]
            arr_tar_mels = [torch.load(x) for x in mel_tar_utts]
            
            wavlm_tar_utts = [self.utt2wavlm[i] for i in rand_utts]
            arr_tar_wavlms = [torch.load(x) for x in wavlm_tar_utts]
            
            tar_mels = torch.cat(arr_tar_mels, dim=1)
            tar_wavlms = torch.cat(arr_tar_wavlms, dim=2).transpose(1,2)
            return mel, tar_mels, wavlm, tar_wavlms, spk
        return mel, wavlm, spk

    def __len__(self):
        """Return the number of spkrs."""
        return self.len_utts

    def label_to_id(self, spk_label):
        if isinstance(spk_label, list):
            return torch.Tensor([int(self.spks.index(i)) for i in spk_label])
        else:
            return int(self.spks.index(spk_label))

    def id_to_label(self, spk_id):
        if isinstance(spk_id, torch.Tensor):
            return [self.spks[i] for i in spk_id]
        else:
            return self.spks[spk_id]


class AudioUtterances(data.Dataset):
    """Dataset class for the Utterances dataset."""

    def __init__(self, data_dir,
                 val=False,
                 test=False,
                 train=False,
                 tar_separate=False,
                 tar_multiplier=4):
        """Initialize and preprocess the Utterances dataset."""
        self.tar_separate = tar_separate
        self.tar_multiplier = tar_multiplier
        self.mel_extractor = my_feats.Audio2Mel()

        np.random.seed(42)
        self.dataset_type = "train" if train else "test" if test else "val"
        """Initialize and preprocess the Utterances dataset."""
        self.data_dir = data_dir
        self.wavs = self.load_scp(os.path.join(self.data_dir, f"{self.dataset_type}_wav.scp"))
        self.utt2spk = self.load_scp(os.path.join(self.data_dir, f"{self.dataset_type}_utt2spk"))
        self.utt2mel = self.load_scp(os.path.join(self.data_dir, f"{self.dataset_type}_utt2mel"))
        self.spk2utt = self.load_scp(os.path.join(self.data_dir, f"{self.dataset_type}_spk2utt"))
        self.spks = list(self.spk2utt.keys())
        self.len_spks = len(self.spks)
        print(f'Finished loading {self.dataset_type} dataset with {len(self.spks)} speakers.')

    def load_scp(self, scp):
        ret = {}
        with open(scp, "r") as f:
            for line in f:
                col1, col2 = line.strip().split(" ")
                if col1 in ret:
                    if not type(ret[col1]) is list:
                        tmp = ret[col1]
                        ret[col1] = [tmp]
                    ret[col1].append(col2)
                else:
                    ret[col1] = col2
        return ret

    def __getitem__(self, index):
        spk = self.spks[index]
        utts = self.spk2utt[spk]
        # pick a random utterance
        mel = np.empty((1, 1))
        while mel.shape[1] < 100:
            rand_utt = np.random.choice(utts)
            utt = self.wavs[rand_utt]

            mel = self.mel_extractor(utt)
            mel = dummy_silence_removal(mel)
        return mel

    def __len__(self):
        """Return the number of spkrs."""
        return self.len_spks

    def label_to_id(self, spk_label):
        if isinstance(spk_label, list):
            return torch.Tensor([int(self.spks.index(i)) for i in spk_label])
        else:
            return int(self.spks.index(spk_label))

    def id_to_label(self, spk_id):
        if isinstance(spk_id, torch.Tensor):
            return [self.spks[i] for i in spk_id]
        else:
            return self.spks[spk_id]


class MelAudioUtterances(data.Dataset):
    """Dataset class for the Utterances dataset."""

    def __init__(self, data_dir,
                 val=False,
                 test=False,
                 train=False,
                 tar_separate=False,
                 tar_multiplier=4):
        """Initialize and preprocess the Utterances dataset."""
        self.tar_separate = tar_separate
        self.tar_multiplier = tar_multiplier
        self.encoder = VoiceEncoder('cpu')


        np.random.seed(42)
        self.dataset_type = "train" if train else "test" if test else "val"
        """Initialize and preprocess the Utterances dataset."""
        self.data_dir = data_dir
        self.wavs = self.load_scp(os.path.join(self.data_dir, f"{self.dataset_type}_wav.scp"))
        self.utt2spk = self.load_scp(os.path.join(self.data_dir, f"{self.dataset_type}_utt2spk"))
        self.utt2mel = self.load_scp(os.path.join(self.data_dir, f"{self.dataset_type}_utt2mel"))
        self.spk2utt = self.load_scp(os.path.join(self.data_dir, f"{self.dataset_type}_spk2utt"))
        self.spks = list(self.spk2utt.keys())
        self.len_spks = len(self.spks)
        print(f'Finished loading {self.dataset_type} dataset with {len(self.spks)} speakers.')

    def load_scp(self, scp):
        ret = {}
        with open(scp, "r") as f:
            for line in f:
                col1, col2 = line.strip().split(" ")
                if col1 in ret:
                    if not type(ret[col1]) is list:
                        tmp = ret[col1]
                        ret[col1] = [tmp]
                    ret[col1].append(col2)
                else:
                    ret[col1] = col2
        return ret

    def __getitem__(self, index):
        spk = self.spks[index]
        utts = self.spk2utt[spk]
        # pick a random utterance
        mel = np.empty((1, 1))
        while mel.shape[1] < 100:
            rand_utt = np.random.choice(utts)
            wav = self.wavs[rand_utt]
            utt = self.utt2mel[rand_utt]

            mel = np.load(utt)
            mel = dummy_silence_removal(mel)
        #utt = sf.read(wav)    
        emb = self.encoder.embed_utterance(preprocess_wav(wav))
        return mel, emb, index

    def __len__(self):
        """Return the number of spkrs."""
        return self.len_spks

    def label_to_id(self, spk_label):
        if isinstance(spk_label, list):
            return torch.Tensor([int(self.spks.index(i)) for i in spk_label])
        else:
            return int(self.spks.index(spk_label))

    def id_to_label(self, spk_id):
        if isinstance(spk_id, torch.Tensor):
            return [self.spks[i] for i in spk_id]
        else:
            return self.spks[spk_id]


class MelEmbUtterances(data.Dataset):
    """Dataset class for the Utterances dataset."""

    def __init__(self, data_dir,
                 val=False,
                 test=False,
                 train=False,
                 tar_separate=False):
        """Initialize and preprocess the Utterances dataset."""
        self.tar_separate = tar_separate
        self.encoder = VoiceEncoder('cpu')


        np.random.seed(42)
        self.dataset_type = "train" if train else "test" if test else "val"
        """Initialize and preprocess the Utterances dataset."""
        self.data_dir = data_dir
        self.wavs = self.load_scp(os.path.join(self.data_dir, f"{self.dataset_type}_wav.scp"))
        self.utt2spk = self.load_scp(os.path.join(self.data_dir, f"{self.dataset_type}_utt2spk"))
        self.utt2mel = self.load_scp(os.path.join(self.data_dir, f"{self.dataset_type}_utt2mel"))
        self.spk2utt = self.load_scp(os.path.join(self.data_dir, f"{self.dataset_type}_spk2utt"))
        self.spks = list(self.spk2utt.keys())
        self.len_spks = len(self.spks)
        print(f'Finished loading {self.dataset_type} dataset with {len(self.spks)} speakers.')

    def load_scp(self, scp):
        ret = {}
        with open(scp, "r") as f:
            for line in f:
                col1, col2 = line.strip().split(" ")
                if col1 in ret:
                    if not type(ret[col1]) is list:
                        tmp = ret[col1]
                        ret[col1] = [tmp]
                    ret[col1].append(col2)
                else:
                    ret[col1] = col2
        return ret

    def __getitem__(self, index):
        spk = self.spks[index]
        utts = self.spk2utt[spk]
        # pick a random utterance
        mel = np.empty((1, 1))
        while mel.shape[1] < 100:
            rand_utt = np.random.choice(utts)
            wav = self.wavs[rand_utt]
            utt = self.utt2mel[rand_utt]

            mel = np.load(utt)
            mel = dummy_silence_removal(mel)
        #utt = sf.read(wav)    
        emb = self.encoder.embed_utterance(preprocess_wav(wav))
        if self.tar_separate:
            tar_spk = np.random.choice(self.spks)
            tar_index = self.spks.index(tar_spk)
            utts = self.spk2utt[tar_spk]
            rand_utt = np.random.choice(utts)
            tar_utt = self.utt2mel[rand_utt]
            tar_wav = self.wavs[rand_utt]
            tar_mel = np.load(tar_utt)
            tar_emb = self.encoder.embed_utterance(preprocess_wav(tar_wav))
            
            tar_mel = dummy_silence_removal(tar_mel)
            return mel, tar_mel, index, tar_index, emb, tar_emb

        return mel, emb, index

    def __len__(self):
        """Return the number of spkrs."""
        return self.len_spks

    def label_to_id(self, spk_label):
        if isinstance(spk_label, list):
            return torch.Tensor([int(self.spks.index(i)) for i in spk_label])
        else:
            return int(self.spks.index(spk_label))

    def id_to_label(self, spk_id):
        if isinstance(spk_id, torch.Tensor):
            return [self.spks[i] for i in spk_id]
        else:
            return self.spks[spk_id]


class MelEmbUtterancesAll(data.Dataset):
    """Dataset class for the Utterances dataset."""

    def __init__(self, data_dir,
                 val=False,
                 test=False,
                 train=False,
                 tar_separate=False,
                 pre_load=False):
        """Initialize and preprocess the Utterances dataset."""
        self.tar_separate = tar_separate
        self.pre_load = pre_load
        self.encoder = VoiceEncoder('cpu')

        np.random.seed(42)
        self.dataset_type = "train" if train else "test" if test else "val"
        """Initialize and preprocess the Utterances dataset."""
        self.data_dir = data_dir
        self.wavs = self.load_scp(os.path.join(self.data_dir, f"{self.dataset_type}_wav.scp"))
        self.utt2spk = self.load_scp(os.path.join(self.data_dir, f"{self.dataset_type}_utt2spk"))
        self.utt2mel = self.load_scp(os.path.join(self.data_dir, f"{self.dataset_type}_utt2mel"))
        self.utt2dvec = self.load_scp(os.path.join(self.data_dir, f"{self.dataset_type}_utt2dvec"))
        self.spk2utt = self.load_scp(os.path.join(self.data_dir, f"{self.dataset_type}_spk2utt"))
        self.spks = list(self.spk2utt.keys())
        self.utts = list(self.utt2mel.keys())
        self.len_wavs = len(self.utt2mel)
        self.len_spks = len(self.spks)
        if self.pre_load:
            self.loaded_utts = {}
            for utt in tqdm(self.utt2mel):
                self.loaded_utts[utt] = np.load(self.utt2mel[utt])
        print(f'Finished loading {self.dataset_type} dataset with {len(self.spks)} speakers.')

    def load_scp(self, scp):
        ret = {}
        with open(scp, "r") as f:
            for line in f:
                col1, col2 = line.strip().split(" ")
                if col1 in ret:
                    if not type(ret[col1]) is list:
                        tmp = ret[col1]
                        ret[col1] = [tmp]
                    ret[col1].append(col2)
                else:
                    ret[col1] = col2
        return ret

    def __getitem__(self, index):
        utt = self.utts[index]

        wav = self.wavs[utt]
        mel_id = self.utt2mel[utt]
        spk = self.utt2spk[utt]
        src_index = int(self.spks.index(spk))
        if self.pre_load:
            mel = self.loaded_utts[utt]
        else:
            mel = np.load(mel_id)
        
        mel = np.load(mel_id)
        mel = dummy_silence_removal(mel) 
        emb = np.load(self.utt2dvec[utt])
        while mel.shape[1] < 100:
            rand_utt = np.random.choice(self.spk2utt[spk])
            
            wav = self.wavs[rand_utt]
            utt = self.utt2mel[rand_utt]
            if self.pre_load:
                mel = self.loaded_utts[rand_utt]
            else:
                mel = np.load(utt)
            mel = dummy_silence_removal(mel)
            emb = np.load(self.utt2dvec[rand_utt])
        if self.tar_separate:
            tar_spk = np.random.choice(self.spks)
            while tar_spk == spk:
                tar_spk = np.random.choice(self.spks)                
            tar_index = self.spks.index(tar_spk)
            utts = self.spk2utt[tar_spk]
            rand_utt = np.random.choice(utts)
            tar_utt = self.utt2mel[rand_utt]
            tar_wav = self.wavs[rand_utt]
            if self.pre_load:
                mel = self.loaded_utts[tar_utt]
            else:
                mel = np.load(tar_utt)
            tar_emb = self.encoder.embed_utterance(preprocess_wav(tar_wav))
            
            tar_mel = dummy_silence_removal(tar_mel)
            return mel, tar_mel, src_index, tar_index, emb, tar_emb
        return mel, emb, src_index

    def __len__(self):
        """Return the number of spkrs."""
        if self.dataset_type == 'train':
            return self.len_wavs
        elif self.dataset_type == 'val':
            return self.len_spks

    def label_to_id(self, spk_label):
        if isinstance(spk_label, list):
            return torch.Tensor([int(self.spks.index(i)) for i in spk_label])
        else:
            return int(self.spks.index(spk_label))

    def id_to_label(self, spk_id):
        if isinstance(spk_id, torch.Tensor):
            return [self.spks[i] for i in spk_id]
        else:
            return self.spks[spk_id]


class MelEmbRandUtterancesAll(data.Dataset):
    """Dataset class for the Utterances dataset."""

    def __init__(self, data_dir,
                 val=False,
                 test=False,
                 train=False,
                 tar_separate=False,
                 pre_load=False):
        """Initialize and preprocess the Utterances dataset."""
        self.tar_separate = tar_separate
        self.pre_load = pre_load
        self.encoder = VoiceEncoder('cpu')

        np.random.seed(42)
        self.dataset_type = "train" if train else "test" if test else "val"
        """Initialize and preprocess the Utterances dataset."""
        self.data_dir = data_dir
        self.wavs = self.load_scp(os.path.join(self.data_dir, f"{self.dataset_type}_wav.scp"))
        self.utt2spk = self.load_scp(os.path.join(self.data_dir, f"{self.dataset_type}_utt2spk"))
        self.utt2mel = self.load_scp(os.path.join(self.data_dir, f"{self.dataset_type}_utt2mel"))
        self.utt2dvec = self.load_scp(os.path.join(self.data_dir, f"{self.dataset_type}_utt2dvec"))
        self.spk2utt = self.load_scp(os.path.join(self.data_dir, f"{self.dataset_type}_spk2utt"))
        self.spks = list(self.spk2utt.keys())
        self.utts = list(self.utt2mel.keys())
        self.len_wavs = len(self.utt2mel)
        self.len_spks = len(self.spks)
        if self.pre_load:
            self.loaded_utts = {}
            for utt in tqdm(self.utt2mel):
                self.loaded_utts[utt] = np.load(self.utt2mel[utt])
        print(f'Finished loading {self.dataset_type} dataset with {len(self.spks)} speakers.')

    def load_scp(self, scp):
        ret = {}
        with open(scp, "r") as f:
            for line in f:
                col1, col2 = line.strip().split(" ")
                if col1 in ret:
                    if not type(ret[col1]) is list:
                        tmp = ret[col1]
                        ret[col1] = [tmp]
                    ret[col1].append(col2)
                else:
                    ret[col1] = col2
        return ret

    def __getitem__(self, index):
        utt = self.utts[index]

        wav = self.wavs[utt]
        mel_id = self.utt2mel[utt]
        spk = self.utt2spk[utt]
        src_index = int(self.spks.index(spk))
        if self.pre_load:
            mel = self.loaded_utts[utt]
        else:
            mel = np.load(mel_id)
        
        mel = np.load(mel_id)
        mel = dummy_silence_removal(mel) 
        emb_src = np.load(self.utt2dvec[utt])
        while mel.shape[1] < 100:
            rand_utt = np.random.choice(self.spk2utt[spk])
            emb_src = np.load(self.utt2dvec[rand_utt])
            
            wav = self.wavs[rand_utt]
            utt = self.utt2mel[rand_utt]
            if self.pre_load:
                mel = self.loaded_utts[rand_utt]
            else:
                mel = np.load(utt)
            mel = dummy_silence_removal(mel)
        
        emb = np.load(self.utt2dvec[np.random.choice(self.spk2utt[spk])])

        return mel, np.concatenate([emb[np.newaxis, :], emb_src[np.newaxis, :]], axis=0), src_index

    def __len__(self):
        """Return the number of spkrs."""
        if self.dataset_type == 'train':
            return self.len_wavs
        elif self.dataset_type == 'val':
            return self.len_spks

    def label_to_id(self, spk_label):
        if isinstance(spk_label, list):
            return torch.Tensor([int(self.spks.index(i)) for i in spk_label])
        else:
            return int(self.spks.index(spk_label))

    def id_to_label(self, spk_id):
        if isinstance(spk_id, torch.Tensor):
            return [self.spks[i] for i in spk_id]
        else:
            return self.spks[spk_id]


def audiocollate(batch):
    minx = min(item.shape[1] for item in batch)
    minx -= minx % 4
    audio = torch.zeros((len(batch), 80, minx))
    spk_ids = []
    for i in range(len(batch)):
        mel = batch[i]
        rand = np.random.randint(max(mel.shape[1] - minx, 1))
        audio[i] = mel[:, rand:rand + minx]
    return audio

def melaudiocollate(batch):
    minx = min(item[0].shape[1] for item in batch)
    minx -= minx % 4
    
    mels = torch.zeros((len(batch), 80, minx))
    embs = []
    spk_ids = []
    for i in range(len(batch)):
        mel, aud, _id = batch[i]
        spk_ids.append(_id)
        embs.append(aud)

        rand = np.random.randint(max(mel.shape[1] - minx, 1))
        mels[i] = mel[:, rand:rand + minx]
    return mels, torch.tensor(np.array(embs)), torch.tensor(np.array(spk_ids))



def collate(batch):
    minx = min(item[0].shape[1] for item in batch)
    minx -= minx % 4
    audio = torch.zeros((len(batch), 80, minx))
    spk_ids = []
    for i in range(len(batch)):
        mel, _id = batch[i]
        spk_ids.append(_id)
        rand = np.random.randint(max(mel.shape[1] - minx, 1))
        audio[i] = mel[:, rand:rand + minx]
    return audio, torch.tensor(np.array(spk_ids))


def wavlm_collate(batch):
    min_mel = min(item[0].shape[1] for item in batch)
    min_mel -= min_mel % 4
    mels = torch.zeros((len(batch), 80, min_mel))

    min_wmel = min(item[1].shape[1] for item in batch)
    min_wmel -= min_wmel % 4
    melws = torch.zeros((len(batch), 768, min_wmel, 13))

    spk_ids = []
    for i in range(len(batch)):
        mel, melw, _id = batch[i]
        spk_ids.append(_id)
        mels[i] = mel[:, :min_mel]
        melws[i] = melw[:, :min_wmel, :]
    return mels, melws, torch.tensor(np.array(spk_ids))



def taremb_collate(batch):
    minx = min(item[0].shape[1] for item in batch)
    minx -= minx % 4
    #print(batch[0][1].shape)
    tar_minx = min(item[1].shape[1] for item in batch)
    tar_minx -= tar_minx % 4
    audio = torch.zeros((len(batch), 80, minx))
    tar_audio = torch.zeros((len(batch), 80, tar_minx))
    spk_ids = []
    tar_spk_ids = []
    embs = []
    tar_embs = []
    
    for i in range(len(batch)):
        mel, tar_mel, spk_id, tar_spk_id, emb, tar_emb = batch[i]
        spk_ids.append(spk_id)
        tar_spk_ids.append(tar_spk_id)
        rand = np.random.randint(max(mel.shape[1] - minx, 1))
        audio[i] = mel[:, rand:rand + minx]
        rand = np.random.randint(max(tar_mel.shape[1] - tar_minx, 1))
        tar_audio[i] = tar_mel[:, rand:rand + tar_minx]
        embs.append(emb)
        tar_embs.append(emb)

    return audio, tar_audio, torch.tensor(np.array(spk_ids)), torch.tensor(np.array(tar_spk_ids)), torch.from_numpy(np.array(embs)), torch.from_numpy(np.array(tar_embs))

def tar_collate(batch):
    minx = min(item[0].shape[1] for item in batch)
    minx -= minx % 4
    tar_minx = min(item[1].shape[1] for item in batch)
    tar_minx -= tar_minx % 4
    audio = torch.zeros((len(batch), 80, minx))
    tar_audio = torch.zeros((len(batch), 80, tar_minx))
    spk_ids = []
    tar_spk_ids = []
    
    for i in range(len(batch)):
        mel, tar_mel, spk_id, tar_spk_id = batch[i]
        spk_ids.append(spk_id)
        tar_spk_ids.append(tar_spk_id)
        rand = np.random.randint(max(mel.shape[1] - minx, 1))
        audio[i] = mel[:, rand:rand + minx]
        rand = np.random.randint(max(tar_mel.shape[1] - tar_minx, 1))
        tar_audio[i] = tar_mel[:, rand:rand + tar_minx]

    return audio, tar_audio, torch.tensor(np.array(spk_ids)), torch.tensor(np.array(tar_spk_ids))


def tar_only_collate(batch):
    minx = min(item[0].shape[1] for item in batch)
    minx -= minx % 4
    tar_minx = min(item[1].shape[1] for item in batch)
    tar_minx -= tar_minx % 4
    audio = torch.zeros((len(batch), 80, minx))
    tar_audio = torch.zeros((len(batch), 80, tar_minx))
    spk_ids = []
    tar_spk_ids = []
    
    for i in range(len(batch)):
        mel, tar_mel, spk_id = batch[i]
        spk_ids.append(spk_id)
        rand = np.random.randint(max(mel.shape[1] - minx, 1))
        audio[i] = mel[:, rand:rand + minx]
        rand = np.random.randint(max(tar_mel.shape[1] - tar_minx, 1))
        tar_audio[i] = tar_mel[:, rand:rand + tar_minx]

    return audio, tar_audio, torch.tensor(np.array(spk_ids))


class SelfTarUtterances(data.Dataset):
    """Dataset class for the Utterances dataset."""

    def __init__(self, data_dir,
                 val=False,
                 test=False,
                 train=False,
                 tar_separate=False,
                 tar_multiplier=4):
        """Initialize and preprocess the Utterances dataset."""
        self.tar_separate = tar_separate
        self.tar_multiplier = tar_multiplier
        np.random.seed(42)
        self.dataset_type = "train" if train else "test" if test else "val"
        """Initialize and preprocess the Utterances dataset."""
        self.data_dir = data_dir
        self.wavs = self.load_scp(os.path.join(self.data_dir, f"{self.dataset_type}_wav.scp"))
        self.utt2spk = self.load_scp(os.path.join(self.data_dir, f"{self.dataset_type}_utt2spk"))
        self.utt2mel = self.load_scp(os.path.join(self.data_dir, f"{self.dataset_type}_utt2mel"))
        self.spk2utt = self.load_scp(os.path.join(self.data_dir, f"{self.dataset_type}_spk2utt"))
        self.spks = list(self.spk2utt.keys())
        self.len_spks = len(self.spks)
        print(f'Finished loading {self.dataset_type} dataset with {len(self.spks)} speakers.')

    def load_scp(self, scp):
        ret = {}
        with open(scp, "r") as f:
            for line in f:
                col1, col2 = line.strip().split(" ")
                if col1 in ret:
                    if not type(ret[col1]) is list:
                        tmp = ret[col1]
                        ret[col1] = [tmp]
                    ret[col1].append(col2)
                else:
                    ret[col1] = col2
        return ret

    def __getitem__(self, index):
        spk = self.spks[index]
        utts = self.spk2utt[spk]
        # pick a random utterance
        mel = np.empty((1, 1))
        while mel.shape[1] < 100:
            rand_utt = np.random.choice(utts)
            utt = self.utt2mel[rand_utt]

            mel = np.load(utt)
            mel = dummy_silence_removal(mel)

        rand_utts = np.random.choice(utts, self.tar_multiplier, replace=False)
        tar_utts = [self.utt2mel[i] for i in rand_utts]
        arr_tar_mels = [np.load(x) for x in tar_utts]
        tar_mels = dummy_silence_removal(np.concatenate(arr_tar_mels, axis=1))
        return mel, tar_mels, index
    
    def __len__(self):
        """Return the number of spkrs."""
        return self.len_spks

    def label_to_id(self, spk_label):
        if isinstance(spk_label, list):
            return torch.Tensor([int(self.spks.index(i)) for i in spk_label])
        else:
            return int(self.spks.index(spk_label))

    def id_to_label(self, spk_id):
        if isinstance(spk_id, torch.Tensor):
            return [self.spks[i] for i in spk_id]
        else:
            return self.spks[spk_id]




