# encoding: utf-8
# 인코더로 이미지를 압축하고 저장하는 코드입니다.코닥 데이터셋에 대해서 수행됩니다.

import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd

from util.metric import MultiScaleSSIM, psnr

import time, os, argparse, sys, random, datetime, tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from pytorch_msssim import ssim, ms_ssim
from util.metric import MultiScaleSSIM, psnr

import torch
from torch.autograd import Variable
import torch.utils.data as data
from torchvision import transforms

# 파라미터
parser = argparse.ArgumentParser()
parser.add_argument('--png_folder', '-f', type=str, default='data/kodim/png', help='folder of test images')
parser.add_argument('--output_folder', '-o', type=str, default='data/kodim/rnn', help='output codes')
parser.add_argument('--model_path', '-m', type=str, default='checkpoint/tiny-imagenet-200-ConvGRUCell/batch32-lr0.0005-l1-06_18_13_46/_best_model_epoch_0192.pth', help='path to model')
parser.add_argument('--reconstruction_metohod', type=str, default='oneshot', choices=['one_shot', 'additive_reconstruction'],help='reconstruction method')
parser.add_argument('--rnn_model', type=str, default='ConvGRUCell', choices=['ConvGRUCell', 'ConvLSTMCell'], help='RNN model')

if __name__ == '__main__':
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    # 원본 PNG 파일이 있는 폴더 경로 (참조용)
    png_images_path = os.listdir(args.png_folder)
    png_images_path.sort()

    png_images = []
    for file_name in png_images_path:
        if file_name.lower().endswith('.png'):
            png_path = os.path.join(args.png_folder, file_name)
            with Image.open(png_path) as img:
                image = torch.from_numpy(np.expand_dims(np.transpose(np.array(img).astype(np.float32) / 255.0, (2, 0, 1)), 0))
                png_images.append(image.to(device))

    ## load networks on GPU
    if args.rnn_model == 'ConvGRUCell':
        from modules import GRU_network as network
    else:
        from modules import LSTM_network as network

    print('','='*120,'\n ||\tdevice: {}\n'.format(device), "="*120, "\n")
    encoder = network.EncoderCell().to(device)
    binarizer = network.Binarizer().to(device)
    decoder = network.DecoderCell().to(device)

    encoder.eval()
    binarizer.eval()
    decoder.eval()

    checkpoint = torch.load(args.model_path)
    encoder.load_state_dict(checkpoint['encoder'])
    binarizer.load_state_dict(checkpoint['binarizer'])
    decoder.load_state_dict(checkpoint['decoder'])

    with torch.no_grad():
        for img_num, image in enumerate(png_images):
            batch_size, input_channels, height, width = image.size()
            assert height % 32 == 0 and width % 32 == 0

            if args.rnn_model == 'ConvGRUCell':
                # init gru state
                encoder_h_1 = (torch.zeros(batch_size, 256, height // 4, width // 4)).to(device)
                encoder_h_2 = (torch.zeros(batch_size, 512, height // 8, width // 8)).to(device)
                encoder_h_3 = (torch.zeros(batch_size, 512, height // 16, width // 16)).to(device)

                decoder_h_1 = (torch.zeros(batch_size, 512, height // 16, width // 16)).to(device)
                decoder_h_2 = (torch.zeros(batch_size, 512, height // 8, width // 8)).to(device)
                decoder_h_3 = (torch.zeros(batch_size, 256, height // 4, width // 4)).to(device)
                decoder_h_4 = (torch.zeros(batch_size, 128, height // 2, width // 2)).to(device)
            else:
                ## init lstm state
                encoder_h_1 = (Variable(torch.zeros(batch_size, 256, height // 4, width // 4)), Variable(torch.zeros(batch_size, 256, height // 4, width // 4)))
                encoder_h_2 = (Variable(torch.zeros(batch_size, 512, height // 8, width // 8)), Variable(torch.zeros(batch_size, 512, height // 8, width // 8)))
                encoder_h_3 = (Variable(torch.zeros(batch_size, 512, height // 16, width // 16)), Variable(torch.zeros(batch_size, 512, height // 16, width // 16)))

                decoder_h_1 = (Variable(torch.zeros(batch_size, 512, height // 16, width // 16)), Variable(torch.zeros(batch_size, 512, height // 16, width // 16)))
                decoder_h_2 = (Variable(torch.zeros(batch_size, 512, height // 8, width // 8)), Variable(torch.zeros(batch_size, 512, height // 8, width // 8)))
                decoder_h_3 = (Variable(torch.zeros(batch_size, 256, height // 4, width // 4)), Variable(torch.zeros(batch_size, 256, height // 4, width // 4)))
                decoder_h_4 = (Variable(torch.zeros(batch_size, 128, height // 2, width // 2)), Variable(torch.zeros(batch_size, 128, height // 2, width // 2)))

            codes = []
            res = image - 0.5

            for iters in range(1, 33):
                encoded, encoder_h_1, encoder_h_2, encoder_h_3 = encoder(res, encoder_h_1, encoder_h_2, encoder_h_3)
                code = binarizer(encoded)
                output, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4 = decoder(code, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4)
                
                res = res - output
                codes.append(code.data.cpu().numpy())

                code_chach = (np.stack(codes).astype(np.int8) + 1) // 2
                export = np.packbits(code_chach.reshape(-1))
                file_name = f"kodim{img_num+1:02}_iter{iters:02}.npz"
                path = os.path.join(args.output_folder, file_name)
                np.savez_compressed(path, shape=code_chach.shape, codes=export)
                print('[{}/{}] Saved: {}'.format(img_num+1, len(png_images), path))
                del code_chach, export, file_name