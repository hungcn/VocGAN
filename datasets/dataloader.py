import os
import glob
import torch
import torch.nn.functional as F
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchaudio
from utils.utils import read_wav_np


def get_dataset_filelist(hp, train):
    if train:
        with open(hp.data.train_file, 'r', encoding='utf-8') as fi:
            train_files = [os.path.join(hp.data.train_dir, 'waves', x.split('|')[0])
                            for x in fi.read().split('\n') if len(x) > 0]
        return train_files
    else:
        with open(hp.data.val_file, 'r', encoding='utf-8') as fi:
            val_files = [os.path.join(hp.data.val_dir, 'waves', x.split('|')[0])
                            for x in fi.read().split('\n') if len(x) > 0]
        return val_files


def create_dataloader(hp, args, train):
    dataset = MelFromDisk(hp, args, train)

    if train:
        return DataLoader(dataset=dataset, batch_size=hp.train.batch_size, shuffle=True,
            num_workers=hp.train.num_workers, pin_memory=True, drop_last=True)
    else:
        return DataLoader(dataset=dataset, batch_size=1, shuffle=False,
            num_workers=0, pin_memory=False, drop_last=False)


class MelFromDisk(Dataset):
    def __init__(self, hp, args, train):
        self.hp = hp
        self.args = args
        self.train = train
        self.data_dir = hp.data.train_dir if train else hp.data.val_dir
        self.wav_list = get_dataset_filelist(hp, train)
        print("Path :", self.data_dir)
        print("Length of wavelist :", len(self.wav_list))
        self.mapping = [i for i in range(len(self.wav_list))]

    def __len__(self):
        return len(self.wav_list)

    def __getitem__(self, idx):

        if self.train:
            idx1 = idx
            idx2 = self.mapping[idx1]
            return self.my_getitem(idx1), self.my_getitem(idx2)
        else:
            return self.my_getitem(idx)

    def shuffle_mapping(self):
        random.shuffle(self.mapping)

    def my_getitem(self, idx):
        wav_path = self.wav_list[idx]
        file_name = os.path.basename(wav_path).split(".")[0]
        speaker = file_name.split('_')[0]

        mel_dir = os.path.join(self.data_dir, 'mels', speaker)
        mel_path = "{}/{}.npy".format(mel_dir, file_name)

        _, audio = read_wav_np(wav_path)

        audio = torch.from_numpy(audio).unsqueeze(0)
        mel = torch.from_numpy(np.load(mel_path))

        if self.train:
            mel_seg_length = int(self.hp.audio.segment_length / self.hp.audio.hop_length) + 2
            gap = mel.size(-1) - mel_seg_length
            if gap < 0:
                gap = -gap
                mel = F.pad(mel, (0, gap), mode='constant')

            mel_start = random.randint(0, mel.size(-1) - mel_seg_length)
            mel_end = mel_start + mel_seg_length
            mel = mel[:, mel_start:mel_end]

            if audio.size(-1) < self.hp.audio.segment_length + self.hp.audio.pad_short:
                audio = F.pad(audio, (0, self.hp.audio.segment_length + self.hp.audio.pad_short - audio.size(-1)), \
                        mode='constant', value=0.0)

            audio_start = mel_start * self.hp.audio.hop_length
            audio = audio[:, audio_start:audio_start+self.hp.audio.segment_length]

        audio = audio + (1/32768) * torch.randn_like(audio)
        return mel, audio


def collate_fn(batch):

    sr = 16000
    # perform padding and conversion to tensor
    mels_g = [x[0][0] for x in batch]
    audio_g = [x[0][1] for x in batch]

    mels_g = torch.stack(mels_g)
    audio_g = torch.stack(audio_g)

    sub_orig_1 = torchaudio.transforms.Resample(sr, (sr // 2))(audio_g)
    sub_orig_2 = torchaudio.transforms.Resample(sr, (sr // 4))(audio_g)
    sub_orig_3 = torchaudio.transforms.Resample(sr, (sr // 8))(audio_g)
    sub_orig_4 = torchaudio.transforms.Resample(sr, (sr // 16))(audio_g)

    mels_d = [x[1][0] for x in batch]
    audio_d = [x[1][1] for x in batch]
    mels_d = torch.stack(mels_d)
    audio_d = torch.stack(audio_d)
    sub_orig_1_d = torchaudio.transforms.Resample(sr, (sr // 2))(audio_d)
    sub_orig_2_d = torchaudio.transforms.Resample(sr, (sr // 4))(audio_d)
    sub_orig_3_d = torchaudio.transforms.Resample(sr, (sr // 8))(audio_d)
    sub_orig_4_d = torchaudio.transforms.Resample(sr, (sr // 16))(audio_d)

    return [mels_g, audio_g, sub_orig_1, sub_orig_2, sub_orig_3, sub_orig_4],\
           [mels_d, audio_d, sub_orig_1_d, sub_orig_2_d, sub_orig_3_d, sub_orig_4_d]