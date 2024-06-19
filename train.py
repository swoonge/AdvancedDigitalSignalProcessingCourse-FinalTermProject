## train.py
# conda activate torch222
# python train.py -N 32 -e 200 --lr 0.0005 --rnn_model ConvGRUCell
# tensorboard --logdir=runs

#encoding: utf-8
import time, os, argparse, sys, random, datetime
from pathlib import Path
from tqdm import tqdm
import numpy as np
from tensorboardX import SummaryWriter

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as LS
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as data
from torchvision import transforms
from util.metric import MultiScaleSSIM, psnr
from pytorch_msssim import ssim, ms_ssim, MS_SSIM

parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', '-N', type=int, default=32, help='batch size')
parser.add_argument('--train', '-f', type=str, default='data/tiny-imagenet-200/train_set', help='folder of training images')
parser.add_argument('--val', '-vf', type=str, default='data/tiny-imagenet-200/val_set', help='folder of validation images')
parser.add_argument('--dataset', type=str, default='tiny-imagenet-200', help='dataset')
parser.add_argument('--max-epochs', '-e', type=int, default=200, help='max epochs')
parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
parser.add_argument('--random_seed', type=int, default=1, help='random seed')
# parser.add_argument('--cudas', '-g', action='store_true', help='enables cuda')
parser.add_argument('--iterations', type=int, default=16, help='unroll iterations')
parser.add_argument('--checkpoint', type=int, help='checkpoint epoch to resume training')
parser.add_argument('--model_path', '-m', type=str, default='checkpoint/tiny-imagenet-200-ConvGRUCell/batch32-lr0.0005-l1-06_18_13_46/_best_model_epoch_0192.pth', help='path to model)')
parser.add_argument('--loss_method', type=str, default='l1', help='loss method')
parser.add_argument('--reconstruction_metohod', type=str, default='oneshot', choices=['one_shot', 'additive_reconstruction'],help='reconstruction method')
parser.add_argument('--rnn_model', type=str, default='ConvGRUCell', choices=['ConvGRUCell', 'ConvLSTMCell'], help='RNN model')
parser.add_argument('--problom3', type=int, default=0, help='problem 3 select')

if __name__ == '__main__':
    args = parser.parse_args()

    # set random seed
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed) # if use multi-GPU
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    # set up logger
    today = datetime.datetime.now().strftime("%m_%d_%H_%M")
    model_name = 'batch{}-lr{}-{}-{}' .format(args.batch_size, args.lr, args.loss_method, today)

    log_path = Path('./logs/{}-{}/{}'.format(args.dataset, args.rnn_model, model_name))
    log_path.mkdir(exist_ok=True, parents=True)
    logger = SummaryWriter(log_path)

    model_out_path = Path('./checkpoint/{}-{}/{}'.format(args.dataset, args.rnn_model, model_name))
    model_out_path.mkdir(exist_ok=True, parents=True)

    print("\n", "="*120, "\n ||\tTrain network with | bath size: ", args.batch_size, " | lr: ", args.lr, " | loss method: ", args.loss_method, " | random seed: ", args.random_seed, " | iterations: ", args.iterations,
    "\n ||\tmodel_out_path: ", model_out_path, "\n ||\tlog_path: ",log_path, "\n", "="*120, )

    def resume(epoch=None, best=True):
        s = '_best' if best else ''
        file_path = '{}/{}_model_epoch_{:04d}.pth'.format(model_out_path, s, epoch)
        checkpoint = torch.load(args.model_path)
        encoder.load_state_dict(checkpoint['encoder'])
        binarizer.load_state_dict(checkpoint['binarizer'])
        decoder.load_state_dict(checkpoint['decoder'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_schedule'])
        epoch = checkpoint['epoch']
        return
    
    def save(epoch, best=True):
        s = '_best' if best else ''
        checkpoint = {
            'encoder': encoder.state_dict(),
            'binarizer': binarizer.state_dict(),
            'decoder': decoder.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_schedule': optimizer.state_dict()['param_groups'][0]['lr'],
            'epoch': epoch,
            'loss': best_loss
        }
        torch.save(checkpoint, '{}/{}_model_epoch_{:04d}.pth'.format(model_out_path, s, epoch))
        return

    ## load 32x32 patches from images
    import dataset

    # 32x32 random crop
    train_transform = transforms.Compose([transforms.RandomCrop((32, 32)), transforms.ToTensor()])
    val_transform = transforms.Compose([transforms.CenterCrop((32, 32)), transforms.ToTensor()])

    # load training set
    train_set = dataset.ImageFolder(root=args.train, transform=train_transform)
    train_loader = data.DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True, num_workers=1)
    print(' ||\t[train loader] total images: {}; total batches: {}\n'.format(len(train_set), len(train_loader)), "="*120)
    val_set = dataset.ImageFolder(root=args.val, transform=val_transform)
    val_loader = data.DataLoader(dataset=val_set, batch_size=args.batch_size, shuffle=False, num_workers=1)
    print(' ||\t[val loader] total images: {}; total batches: {}\n'.format(len(val_set), len(val_loader)), "="*120)

    ## load networks on GPU
    if args.rnn_model == 'ConvGRUCell':
        from modules import GRU_network as network
    else:
        from modules import LSTM_network as network

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(' ||\tdevice: {}\n'.format(device), "="*120, "\n")
    encoder = network.EncoderCell().to(device)
    binarizer = network.Binarizer().to(device)
    decoder = network.DecoderCell().to(device)

    # set up optimizer and scheduler
    optimizer = optim.Adam([ {'params': encoder.parameters()}, {'params': binarizer.parameters()}, {'params': decoder.parameters()} ], lr=args.lr)
    lr_scheduler = LS.MultiStepLR(optimizer, milestones=[20, 25, 30, 40, 50, 70, 100, 150, 200, 300, 400, 500, 700], gamma=0.5)
    # lr_scheduler = LS.MultiStepLR(optimizer, milestones=[10, 30, 50, 100, 200, 300, 500, 700], gamma=0.5)
    # lr_scheduler = LS.MultiStepLR(optimizer, milestones=[5, 20, 50, 100, 200, 400, 600, 800], gamma=0.5)

    # if checkpoint is provided, resume from the checkpoint
    last_epoch = 0
    if args.checkpoint:
        resume(args.checkpoint)
        last_epoch = args.checkpoint
        lr_scheduler.last_epoch = last_epoch - 1

    ## training
    total_t = 0
    model_t = 0
    loss_t = 0
    bp_t = 0

    best_loss = 1e9
    epoch_loss = 0
    best_eval_loss = 1e9
    for epoch in range(last_epoch + 1, args.max_epochs + 1):
        encoder.train(), binarizer.train(), decoder.train()

        train_loader = tqdm(train_loader)
        for batch, data in enumerate(train_loader):
            ## 이게 한 베치에서 이뤄지는 것. 로스랑 시간 재는거 다시 확인해보기
            batch_t0 = time.time()
            model_t0 = time.time()
            if args.rnn_model == 'ConvGRUCell':
                # init gru state
                encoder_h_1 = Variable(torch.zeros(data.size(0), 256, 8, 8).to(device))
                encoder_h_2 = Variable(torch.zeros(data.size(0), 512, 4, 4).to(device))
                encoder_h_3 = Variable(torch.zeros(data.size(0), 512, 2, 2).to(device))

                decoder_h_1 = Variable(torch.zeros(data.size(0), 512, 2, 2).to(device))
                decoder_h_2 = Variable(torch.zeros(data.size(0), 512, 4, 4).to(device))
                decoder_h_3 = Variable(torch.zeros(data.size(0), 256, 8, 8).to(device))
                decoder_h_4 = Variable(torch.zeros(data.size(0), 128, 16, 16).to(device))
            else:
                ## init lstm state
                encoder_h_1 = (Variable(torch.zeros(data.size(0), 256, 8, 8).to(device)),
                               Variable(torch.zeros(data.size(0), 256, 8, 8).to(device)))
                encoder_h_2 = (Variable(torch.zeros(data.size(0), 512, 4, 4).to(device)),
                               Variable(torch.zeros(data.size(0), 512, 4, 4).to(device)))
                encoder_h_3 = (Variable(torch.zeros(data.size(0), 512, 2, 2).to(device)),
                               Variable(torch.zeros(data.size(0), 512, 2, 2).to(device)))

                decoder_h_1 = (Variable(torch.zeros(data.size(0), 512, 2, 2).to(device)),
                               Variable(torch.zeros(data.size(0), 512, 2, 2).to(device)))
                decoder_h_2 = (Variable(torch.zeros(data.size(0), 512, 4, 4).to(device)),
                               Variable(torch.zeros(data.size(0), 512, 4, 4).to(device)))
                decoder_h_3 = (Variable(torch.zeros(data.size(0), 256, 8, 8).to(device)),
                               Variable(torch.zeros(data.size(0), 256, 8, 8).to(device)))
                decoder_h_4 = (Variable(torch.zeros(data.size(0), 128, 16, 16).to(device)),
                               Variable(torch.zeros(data.size(0), 128, 16, 16).to(device)))
            model_t1 = time.time()
            model_t += model_t1 - model_t0

            patches = Variable(data.to(device))
            optimizer.zero_grad()
            losses = []
            res = patches - 0.5  # r_0 = x
            x_org = patches - 0.5

            loss_t0 = time.time()
            output = torch.zeros_like(patches) # ^x_{t-1} = 0
            org_image = data
            decoded_images = torch.zeros_like(patches) + 0.5
            decoded_images_list = []
            for iter_n in range(args.iterations): # args.iterations = 16
                encoded, encoder_h_1, encoder_h_2, encoder_h_3 = encoder(
                    res, encoder_h_1, encoder_h_2, encoder_h_3)

                codes = binarizer(encoded)

                if args.reconstruction_metohod == 'additive_reconstruction':
                    output, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4 = decoder(codes, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4) + output
                else:
                    output, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4 = decoder(codes, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4)

                ## 문제 3: res = res - output || res = x_orsg - output
                if args.problom3:
                    res = res - output
                else:
                    res = x_org - output # output = ^x_{t-1}
                # l1 loss의 경우, 각 iteration마다 loss의 평균을 계산하고, 이를 모두 더한 후 iteration 수로 나누어 평균을 구함
                # 이 때, loss 값 -> 
                if args.loss_method == 'l1':
                    losses.append(res.abs().mean())
                    # print( f'[TRAIN] Epoch[{epoch}] Batch[{batch}] Iteration[{iter_n}] L1 Loss: {losses[-1]}')
                else: 
                    decoded_images = decoded_images + output
                    loss = 1 - ssim(decoded_images, x_org, data_range=1.0, size_average=True)
                    # print( f'[TRAIN] Epoch[{epoch}] Batch[{batch}] Iteration[{iter_n}] SSIM Loss: {loss.item()}')
                    losses.append(loss)
            loss_t1 = time.time()
            loss_t += loss_t1 - loss_t0
            loss = sum(losses) / args.iterations
            epoch_loss += loss.item()

            bp_t0 = time.time()
            loss.backward()
            optimizer.step()
            bp_t1 = time.time()
            bp_t += bp_t1 - bp_t0
            batch_t1 = time.time()
            total_t += batch_t1 - batch_t0

            del patches, res, losses, output, loss

        lr_scheduler.step()
        epoch_loss /= len(train_loader)
        model_t /= len(train_set)
        loss_t /= len(train_set)
        bp_t /= len(train_set)
        total_t_per_batch = total_t/len(train_loader)
        print('[TRAIN] Epoch[{}] Loss: {:.6f} | Model inference: {:.5f} sec | Loss: {:.5f} sec | Backpropagation: {:.5f} sec'.format(epoch, epoch_loss, model_t, loss_t, bp_t))
        
        # validation
        val_loss = 0
        val_loss_ssim = 0
        val_total_t0 = time.time()
        with torch.no_grad():
            if epoch >= 0 and epoch % 1 == 0:
                val_losses = []
                for i, pred in enumerate(val_loader):
                    ### eval ###
                    encoder.eval(), binarizer.eval(), decoder.eval()
                    if args.rnn_model == 'ConvGRUCell':
                        # init gru state
                        encoder_h_1 = Variable(torch.zeros(data.size(0), 256, 8, 8).to(device))
                        encoder_h_2 = Variable(torch.zeros(data.size(0), 512, 4, 4).to(device))
                        encoder_h_3 = Variable(torch.zeros(data.size(0), 512, 2, 2).to(device))

                        decoder_h_1 = Variable(torch.zeros(data.size(0), 512, 2, 2).to(device))
                        decoder_h_2 = Variable(torch.zeros(data.size(0), 512, 4, 4).to(device))
                        decoder_h_3 = Variable(torch.zeros(data.size(0), 256, 8, 8).to(device))
                        decoder_h_4 = Variable(torch.zeros(data.size(0), 128, 16, 16).to(device))
                    else:
                        ## init lstm state
                        encoder_h_1 = (Variable(torch.zeros(data.size(0), 256, 8, 8).to(device)),
                                    Variable(torch.zeros(data.size(0), 256, 8, 8).to(device)))
                        encoder_h_2 = (Variable(torch.zeros(data.size(0), 512, 4, 4).to(device)),
                                    Variable(torch.zeros(data.size(0), 512, 4, 4).to(device)))
                        encoder_h_3 = (Variable(torch.zeros(data.size(0), 512, 2, 2).to(device)),
                                    Variable(torch.zeros(data.size(0), 512, 2, 2).to(device)))

                        decoder_h_1 = (Variable(torch.zeros(data.size(0), 512, 2, 2).to(device)),
                                    Variable(torch.zeros(data.size(0), 512, 2, 2).to(device)))
                        decoder_h_2 = (Variable(torch.zeros(data.size(0), 512, 4, 4).to(device)),
                                    Variable(torch.zeros(data.size(0), 512, 4, 4).to(device)))
                        decoder_h_3 = (Variable(torch.zeros(data.size(0), 256, 8, 8).to(device)),
                                    Variable(torch.zeros(data.size(0), 256, 8, 8).to(device)))
                        decoder_h_4 = (Variable(torch.zeros(data.size(0), 128, 16, 16).to(device)),
                                    Variable(torch.zeros(data.size(0), 128, 16, 16).to(device)))
                    
                    patches = Variable(data.to(device))
                    losses = []
                    res = patches - 0.5  # r_0 = x
                    x_org = patches - 0.5
                    org_image = data.permute(0, 2, 3, 1)

                    output = torch.zeros_like(patches) # ^x_{t-1} = 0
                    decoded_images = torch.zeros_like(patches) + 0.5
                    for iter_n in range(args.iterations): # args.iterations = 16
                        encoded, encoder_h_1, encoder_h_2, encoder_h_3 = encoder(
                            res, encoder_h_1, encoder_h_2, encoder_h_3)

                        codes = binarizer(encoded)

                        if args.reconstruction_metohod == 'additive_reconstruction':
                            output, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4 = decoder(codes, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4) + output
                        else:
                            output, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4 = decoder(codes, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4)

                        ## 문제 3: res = res - output || res = x_orsg - output
                        if args.problom3:
                            res = res - output
                        else:
                            res = x_org - output # output = ^x_{t-1}
                        decoded_images = decoded_images + output
                        losses.append(res.abs().mean())
                    val_losses.append(torch.mean(torch.stack(losses)))
                val_loss = torch.mean(torch.stack(val_losses))
                val_total_t1 = time.time()
                val_loss_ssim = ssim(decoded_images, x_org, data_range=1.0, size_average=True)
                print('[Val] Epoch[{}] l1 Loss: {:.6f} | ssim_score: {:.5f} | Time: {:.5f} sec'.format(epoch, val_loss, val_loss_ssim, val_total_t1 - val_total_t0))
        
            if (val_loss <= best_loss + 1e-5): 
                best_loss = val_loss
                save(epoch, True)
                print('[Save] Best model saved at {} epoch'.format(epoch))
            elif epoch % 10 == 0:
                save(epoch, False)
                print('[Save] model saved at {} epoch'.format(epoch))

        # ================================================================== #
        #                        Tensorboard Logging                         #
        # ================================================================== #
        # logger.add_scalar('Train/val_loss',mean_val_loss,epoch)
        logger.add_scalar('Train/epoch_loss', epoch_loss, epoch)
        logger.add_scalar('Train/rl', lr_scheduler.get_last_lr()[0], epoch)
        logger.add_scalar('Train/val_loss', val_loss, epoch)
        logger.add_scalar('Train/val_loss_ssim', val_loss_ssim, epoch)
        print("log file saved to {}\n".format(log_path))
        
         