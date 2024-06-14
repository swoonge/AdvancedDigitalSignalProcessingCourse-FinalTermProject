## train.py
# python train.py --train /path/to/training/images -N 32 -e 200 --lr 0.0005 -f data/tiny-imagenet-200/test/images

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

parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', '-N', type=int, default=32, help='batch size')
parser.add_argument('--train', '-f', required=True, type=str, help='folder of training images')
parser.add_argument('--dataset', type=str, default='tiny-imagenet-200', help='dataset')
parser.add_argument('--max-epochs', '-e', type=int, default=1000, help='max epochs')
parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
parser.add_argument('--random_seed', type=int, default=0, help='random seed')
# parser.add_argument('--cudas', '-g', action='store_true', help='enables cuda')
parser.add_argument('--iterations', type=int, default=16, help='unroll iterations')
parser.add_argument('--checkpoint', type=int, help='checkpoint epoch to resume training')
parser.add_argument('--loss_method', type=str, default='l1', help='loss method')

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

    log_path = Path('./logs/{}/{}'.format(args.dataset, model_name))
    log_path.mkdir(exist_ok=True, parents=True)
    logger = SummaryWriter(log_path)

    model_out_path = Path('./checkpoint/{}/{}'.format(args.dataset, model_name))
    model_out_path.mkdir(exist_ok=True, parents=True)

    print("\n", "="*120, "\n ||\tTrain network with | bath size: ", args.batch_size, " | lr: ", args.lr, " | loss method: ", args.loss_method, " | random seed: ", args.random_seed, " | iterations: ", args.iterations,
    "\n ||\tmodel_out_path: ", model_out_path, "\n ||\tlog_path: ",log_path, "\n", "="*120, )

    def resume(epoch=None, best=True):
        s = '_best' if best else ''
        encoder.load_state_dict(torch.load('{}/encoder{}_{:08d}.pth'.format(model_out_path, s, epoch)))
        binarizer.load_state_dict(torch.load('{}/binarizer{}_{:08d}.pth'.format(model_out_path, s, epoch)))
        decoder.load_state_dict(torch.load('{}/decoder{}_{:08d}.pth'.format(model_out_path, s, epoch)))
        return

    def save(epoch, best=True):
        s = '_best' if best else ''
        torch.save(encoder.state_dict(), '{}/encoder{}_{:08d}.pth'.format(model_out_path, s, epoch))
        torch.save(binarizer.state_dict(), '{}/binarizer{}_{:08d}.pth'.format(model_out_path, s, epoch))
        torch.save(decoder.state_dict(), '{}/decoder{}_{:08d}.pth'.format(model_out_path, s, epoch))
        return

    ## load 32x32 patches from images
    import dataset

    # 32x32 random crop
    train_transform = transforms.Compose([transforms.RandomCrop((32, 32)), transforms.ToTensor()])

    # load training set
    train_set = dataset.ImageFolder(root=args.train, transform=train_transform)
    train_loader = data.DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True, num_workers=1)
    print(' ||\ttotal images: {}; total batches: {}\n'.format(len(train_set), len(train_loader)), "="*120)

    ## load networks on GPU
    from modules import network
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
    lr_scheduler = LS.MultiStepLR(optimizer, milestones=[3, 10, 20, 50, 100, 200, 400, 800], gamma=0.5)
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

    epoch_loss = 0
    best_eval_loss = 1e9
    for epoch in range(last_epoch + 1, args.max_epochs + 1):
        encoder.train(), binarizer.train(), decoder.train()

        train_loader = tqdm(train_loader)
        for batch, data in enumerate(train_loader):
            ## 이게 한 베치에서 이뤄지는 것. 로스랑 시간 재는거 다시 확인해보기
            batch_t0 = time.time()
            model_t0 = time.time()
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
            res = patches - 0.5

            loss_t0 = time.time()
            for _ in range(args.iterations):
                encoded, encoder_h_1, encoder_h_2, encoder_h_3 = encoder(
                    res, encoder_h_1, encoder_h_2, encoder_h_3)

                codes = binarizer(encoded)

                output, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4 = decoder(
                    codes, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4)

                res = res - output
                losses.append(res.abs().mean())
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

        lr_scheduler.step()
        epoch_loss /= len(train_loader)
        model_t /= len(train_set)
        loss_t /= len(train_set)
        bp_t /= len(train_set)
        total_t_per_batch = total_t/len(train_loader)
        print('[TRAIN] Epoch[{}] Loss: {:.6f} | Model inference: {:.5f} sec | Loss: {:.5f} sec | Backpropagation: {:.5f} sec'.format(epoch, epoch_loss, model_t, loss_t, bp_t))
        
        # evaluate model
        #
        #
        #
        #

        # mean_val_loss = torch.mean(torch.stack(mean_val_loss)).item()
        # epoch_loss /= len(train_loader)
        # print('Validation loss: {:.4f}, epoch_loss: {:.4f}, best val loss: {:.4f}, lr: {:.10f}' .format(mean_val_loss, mean_train_loss, best_loss, lr_schedule.get_last_lr()[0]))

        # checkpoint = {
        #         'optimizer':optimizer.state_dict(),
        #         "epoch": epoch,
        #         'lr_schedule': optimizer.state_dict()['param_groups'][0]['lr'],
        #         'loss': mean_val_loss
        #     }

        # if (mean_val_loss <= best_loss + 1e-5): 
        #     best_loss = mean_val_loss
        #     model_out_fullpath = "{}/best_model_epoch_{}(val_loss{}).pth".format(model_out_path, epoch, best_loss)
        #     torch.save(checkpoint, model_out_fullpath)
        #     print('time consume: {:.1f}s, So far best loss: {:.4f}, Checkpoint saved to {}' .format(timeconsume, best_loss, model_out_fullpath))
        # else:
        #     model_out_fullpath = "{}/model_epoch_{}.pth".format(model_out_path, epoch)
        #     torch.save(checkpoint, model_out_fullpath)
        #     print("Epoch [{}/{}] done. Epoch Loss {:.4f}. Checkpoint saved to {}"
        #         .format(epoch, opt.epoch, epoch_loss, model_out_fullpath))
        
        # if best_eval_loss > mean_val_loss:
        #     best_eval_loss = mean_val_loss
        #     save(epoch, True)
        # else:
        #     save(epoch, False)
        save(epoch, False)

        # ================================================================== #
        #                        Tensorboard Logging                         #
        # ================================================================== #
        # logger.add_scalar('Train/val_loss',mean_val_loss,epoch)
        logger.add_scalar('Train/epoch_loss', epoch_loss, epoch)
        logger.add_scalar('Train/rl', lr_scheduler.get_last_lr()[0], epoch)
        print("log file saved to {}\n".format(log_path))
