import os
import glob
import tqdm
import torch
import argparse
import numpy as np
from utils.stft import TacotronSTFT
from utils.hparams import HParam
from utils.utils import read_wav_np


def main(hp, args):
    stft = TacotronSTFT(filter_length=hp.audio.filter_length,
                        hop_length=hp.audio.hop_length,
                        win_length=hp.audio.win_length,
                        n_mel_channels=hp.audio.n_mel_channels,
                        sampling_rate=hp.audio.sampling_rate,
                        mel_fmin=hp.audio.mel_fmin,
                        mel_fmax=hp.audio.mel_fmax)

    wav_files = glob.glob(os.path.join(args.data_path, '**', '*.wav'), recursive=True)
    mel_dir = os.path.joint(os.path.split(args.data_path)[0], 'mels')

    # Create all folders
    os.makedirs(mel_dir, exist_ok=True)
    for wavpath in tqdm.tqdm(wav_files, desc='preprocess wav to mel'):
        sr, wav = read_wav_np(wavpath)
        assert sr == hp.audio.sampling_rate, \
            "sample rate mismatch. expected %d, got %d at %s" % \
            (hp.audio.sampling_rate, sr, wavpath)
        
        if len(wav) < hp.audio.segment_length + hp.audio.pad_short:
            wav = np.pad(wav, (0, hp.audio.segment_length + hp.audio.pad_short - len(wav)), \
                    mode='constant', constant_values=0.0)

        wav = torch.from_numpy(wav).unsqueeze(0)
        mel = stft.mel_spectrogram(wav)  # mel [1, num_mel, T]

        mel = mel.squeeze(0)  # [num_mel, T]
        name = os.path.basename(wavpath).split(".")[0]
        mel_path = os.path.join(mel_dir, name.split('_')[0])
        np.save('{}/{}.npy'.format(mel_path, name), mel.numpy(), allow_pickle=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True,
                        help="yaml file for config.")
    parser.add_argument('-d', '--data_path', type=str, required=True,
                        help="root directory of wav files")
    args = parser.parse_args()
    hp = HParam(args.config)

    main(hp, args)
