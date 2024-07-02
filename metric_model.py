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
parser.add_argument('--model_path', '-m', type=str, default='checkpoint/tiny-imagenet-200-ConvGRUCell/batch32-lr0.0005-mix_iter-07_02_14_39/_best_model_epoch_0003.pth', help='path to model')
parser.add_argument('--reconstruction_metohod', type=str, default='oneshot', choices=['one_shot', 'additive_reconstruction'],help='reconstruction method')
parser.add_argument('--rnn_model', type=str, default='ConvGRUCell', choices=['ConvGRUCell', 'ConvLSTMCell'], help='RNN model')
parser.add_argument('--excel_path', type=str, default='data/kodim/results_rnn_metric.xlsx', help='path to save metric')

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


    print("[INFO] Start encoding...")
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
                print('\t[{}/{}] Saved: {}'.format(img_num+1, len(png_images), path))
                del code_chach, export, file_name

    print("[INFO] Encoding done. Decoding start...")

    png_folder = 'data/kodim/png'
    rnn_folder = 'data/kodim/rnn'

    png_images_path = os.listdir(png_folder)
    png_images_path.sort()

    rnn_images_path = os.listdir(rnn_folder)
    rnn_images_path.sort()

    iters = range(1, 33)
    results_dict = {sampling: [] for sampling in iters}
    codes_dict = {}
    decoded_images_dict = {}

    # image 및 codes 로드
    png_images = {}
    for file_name in png_images_path:
        if file_name.lower().endswith('.png'):
            png_path = os.path.join(png_folder, file_name)
            codes_dict[file_name.split('.')[0]] = []
            decoded_images_dict[file_name.split('.')[0]] = []
            with Image.open(png_path) as img:
                png_images[file_name.split('.')[0]] = np.array(img).transpose(2, 0, 1)

    for file_name in rnn_images_path:
        if file_name.lower().endswith('.npz'):
            npz_path = os.path.join(rnn_folder, file_name)
            with np.load(npz_path) as data:
                codes_dict[file_name.split('_')[0]].append([data['shape'], data['codes']])
                # print(npz_path, data['shape'], data['codes'].shape) # >> 32*32*48 / 6144 = 8 -> 즉 code는 8비트, 32, 32이니 실제로는 4, 4패치

    # 이미지 디코딩
    for k in codes_dict.keys(): # k = 'kodim01' 등의 키
        content = codes_dict[k] # 각 이미지에 대한 content, content[0~32]는 각각 iter, content[i][0~1]은 각각 shape, codes
        print("\t[decode process] image: ", k)
        # for i in range(len(content)): # i = 1~32의 각 iter
        codes = np.unpackbits(content[-1][1])
        codes = np.reshape(codes, content[-1][0]).astype(np.float32) * 2 - 1

        codes = torch.from_numpy(codes)
        iters, batch_size, channels, height, width = codes.size()
        height = height * 16
        width = width * 16

        with torch.no_grad():
            if args.rnn_model == 'ConvGRUCell':
                # init gru state
                decoder_h_1 = (torch.zeros(batch_size, 512, height // 16, width // 16)).to(device)
                decoder_h_2 = (torch.zeros(batch_size, 512, height // 8, width // 8)).to(device)
                decoder_h_3 = (torch.zeros(batch_size, 256, height // 4, width // 4)).to(device)
                decoder_h_4 = (torch.zeros(batch_size, 128, height // 2, width // 2)).to(device)
            else:
                ## init lstm state
                decoder_h_1 = (torch.zeros(batch_size, 512, height // 16, width // 16), torch.zeros(batch_size, 512, height // 16, width // 16)).to(device)
                decoder_h_2 = (torch.zeros(batch_size, 512, height // 8, width // 8), torch.zeros(batch_size, 512, height // 8, width // 8)).to(device)
                decoder_h_3 = (torch.zeros(batch_size, 256, height // 4, width // 4), torch.zeros(batch_size, 256, height // 4, width // 4)).to(device)
                decoder_h_4 = (torch.zeros(batch_size, 128, height // 2, width // 2), torch.zeros(batch_size, 128, height // 2, width // 2)).to(device)

            codes = codes.to(device)
            image = torch.zeros(1, 3, height, width) + 0.5

            for iters in range(iters):
                output, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4 = decoder(
                    codes[iters], decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4)
                image = image + output.data.cpu()
                image_disp = np.squeeze(image[0].numpy().clip(0, 1) * 255.0).astype(np.uint8).transpose(1, 2, 0)
                decoded_images_dict[k].append([k, iters+1, image_disp])
    
    results_dict = {}
    for i in range(1, 33):
        results_dict[i] = []

    print("[INFO] Calculate metric...")

    for k in png_images.keys():
        org_image = np.expand_dims(png_images[k], axis=0)
        print("[calculate metric] image: ", k)
        for i in range(len(decoded_images_dict[k])):
            decoded_images = decoded_images_dict[k][i][2].transpose(2, 0, 1)
            decoded_images = np.expand_dims(decoded_images, axis=0)
            
            ms_ssim_score_1 = MultiScaleSSIM(org_image, decoded_images, max_val=255)
            ms_ssim_score_2 = ms_ssim(torch.tensor(org_image).float(), torch.tensor(decoded_images).float(), data_range=255, size_average=True).item()
            ssim_score = ssim(torch.tensor(org_image).float(), torch.tensor(decoded_images).float(), data_range=255, size_average=True).item()
            psnr_score = psnr(org_image, decoded_images)

            path = os.path.join(rnn_folder, f"{k}_iter{i+1:02}.npz")
            rnn_size = os.path.getsize(path) * 8
            bpp = rnn_size / (decoded_images.shape[2] * decoded_images.shape[3])

            results_dict[i+1].append([i, bpp, ssim_score, ms_ssim_score_1, ms_ssim_score_2, psnr_score])
            print(f"\titer: {i+1:02} | bpp: {bpp:.4f} | SSIM: {ssim_score:.4f} | MS-SSIM1: {ms_ssim_score_1:.4f} | MS-SSIM2: {ms_ssim_score_2:.4f} |PSNR: {psnr_score:.4f}")

    results = []
    for datas in results_dict.values():
        data = [datas[0][0]]

        bpp_values = [r[1] for r in datas]
        ssim_scores = [r[2] for r in datas]
        ms_ssim_scores1 = [r[3] for r in datas]
        ms_ssim_scores2 = [r[4] for r in datas]
        psnr_scores = [r[5] for r in datas]
        
        data.append(sum(bpp_values) / len(bpp_values))
        data.append(sum(ssim_scores) / len(ssim_scores))
        data.append(sum(ms_ssim_scores1) / len(ms_ssim_scores1))
        data.append(sum(ms_ssim_scores2) / len(ms_ssim_scores2))
        data.append(sum(psnr_scores) / len(psnr_scores))

        results.append(data)

    # DataFrame 생성
    df = pd.DataFrame(results, columns=['iter', 'bpp', 'SSIM', 'MS-SSIM1', 'MS-SSIM2', 'PSNR'])

    df.to_excel(args.excel_path, index=False)