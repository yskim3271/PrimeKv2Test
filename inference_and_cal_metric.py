from __future__ import absolute_import, division, print_function, unicode_literals

import glob
import os
import argparse
import json
from re import S
import torch
import librosa
from env import AttrDict
from datasets.dataset import mag_pha_stft, mag_pha_istft
from models.generator import LKFCA_Net
import soundfile as sf
import os
import argparse
import librosa
import numpy as np
from compute_metrics import compute_metrics
from rich.progress import track
h = None
device = None


def get_dataset_filelist(a):
    with open(a.input_test_file, 'r', encoding='utf-8') as fi:
        indexes = [x.split('|')[0] for x in fi.read().split('\n') if len(x) > 0]

    return indexes

def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict

def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '*')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return ''
    return sorted(cp_list)[-1]

def inference(a):
    model = LKFCA_Net(h).to(device)

    state_dict = load_checkpoint(a.checkpoint_file, device)
    model.load_state_dict(state_dict['generator'])

    with open(a.input_test_file, 'r', encoding='utf-8') as fi:
        test_indexes = [x.split('|')[0] for x in fi.read().split('\n') if len(x) > 0]

    os.makedirs(a.output_dir, exist_ok=True)

    model.eval()

    with torch.no_grad():
        for i, index in enumerate(test_indexes):
            print(index)
            noisy_wav, _ = librosa.load(os.path.join(a.input_noisy_wavs_dir, index+'.wav'), h.sampling_rate)
            noisy_wav = torch.FloatTensor(noisy_wav).to(device)
            norm_factor = torch.sqrt(len(noisy_wav) / torch.sum(noisy_wav ** 2.0)).to(device)
            noisy_wav = (noisy_wav * norm_factor).unsqueeze(0)
            noisy_amp, noisy_pha, noisy_com = mag_pha_stft(noisy_wav, h.n_fft, h.hop_size, h.win_size, h.compress_factor)
            amp_g, pha_g, com_g = model(noisy_amp, noisy_pha)
            audio_g = mag_pha_istft(amp_g, pha_g, h.n_fft, h.hop_size, h.win_size, h.compress_factor)
            audio_g = audio_g / norm_factor

            output_file = os.path.join(a.output_dir, index+'.wav')

            sf.write(output_file, audio_g.squeeze().cpu().numpy(), h.sampling_rate, 'PCM_16')


def main():
    print('Initializing Inference Process..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_clean_wavs_dir', default='/media/lz-4060ti-linux/SE/SE/VB_DEMAND_16K/clean_train')
    parser.add_argument('--input_noisy_wavs_dir', default='/media/lz-4060ti-linux/SE/SE/VB_DEMAND_16K/noisy_train')
    parser.add_argument('--input_test_file', default='VoiceBank+DEMAND/test.txt')
    parser.add_argument('--output_dir', default='generated_files/g_best')
    parser.add_argument('--checkpoint_file', required=True)

    a = parser.parse_args()

    config_file = os.path.join(os.path.split(a.checkpoint_file)[0], 'config.json')
    with open(config_file) as f:
        data = f.read()

    global h
    json_config = json.loads(data)
    h = AttrDict(json_config)

    torch.manual_seed(h.seed)
    global device
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    inference(a)

    indexes = get_dataset_filelist(a)
    num = len(indexes)
    print(num)
    metrics_total = np.zeros(6)
    for index in track(indexes):
        clean_wav = os.path.join(a.input_clean_wavs_dir, index + '.wav')
        noisy_wav = os.path.join(a.output_dir, index + '.wav')
        clean, sr = librosa.load(clean_wav, h.sampling_rate)
        noisy, sr = librosa.load(noisy_wav, h.sampling_rate)

        metrics = compute_metrics(clean, noisy, sr, 0)
        metrics = np.array(metrics)
        metrics_total += metrics

    metrics_avg = metrics_total / num
    print('pesq: ', metrics_avg[0], 'csig: ', metrics_avg[1], 'cbak: ', metrics_avg[2],
          'covl: ', metrics_avg[3], 'ssnr: ', metrics_avg[4], 'stoi: ', metrics_avg[5])

    file_path = os.path.join(a.output_dir, 'output.txt')
    with open(file_path, 'w') as f:
        print('pesq: ', metrics_avg[0], 'csig: ', metrics_avg[1], 'cbak: ', metrics_avg[2],
              'covl: ', metrics_avg[3], 'ssnr: ', metrics_avg[4], 'stoi: ', metrics_avg[5], file=f)
if __name__ == '__main__':
    main()

