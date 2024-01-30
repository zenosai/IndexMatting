import os
import argparse

import cv2 as cv
import numpy as np

import paddle
import paddle.optimizer as optim
from paddle.vision.transforms import Compose
from paddle.io import DataLoader
from paddle.optimizer.lr import MultiStepDecay

from models.lib import patch_replication_callback
from models.mobilenetv2 import mobilenetv2
from models.vggnet import vgg16
from datasets.dataset import AdobeImageMattingDataset, RandomCrop, RandomFlip, Normalize, ToTensor
from core.utils import *
from core.trainval import *

# prevent dataloader deadlock
cv.setNumThreads(0)

device = paddle.device.set_device("gpu:0")

backbone = {
    'mobilenetv2': mobilenetv2,
    'vgg16': vgg16
}

# constant
IMG_SCALE = 1. / 255
IMG_MEAN = np.array([0.485, 0.456, 0.406, 0]).reshape((1, 1, 4))
IMG_STD = np.array([0.229, 0.224, 0.225, 1]).reshape((1, 1, 4))
SCALES = [1, 1.5, 2]

# system-io-related parameters
DATASET = 'Adobe_Image_Matting'
DATA_DIR = 'datasets/Combined_Dataset'
EXP = 'indexnet_matting'
DATA_LIST = './datasets/lists/train.txt'
DATA_VAL_LIST = './datasets/lists/test.txt'
RESTORE_FROM = 'model_ckpt.pdparams'
SNAPSHOT_DIR = './snapshots'
RESULT_DIR = './results'

# model-related parameters
OUTPUT_STRIDE = 32
CONV_OPERATOR = 'std_conv'  # choose in ['std_conv', 'dep_sep_conv']
DECODER = 'indexnet'  # choose in ['unet_style', 'deeplabv3+', 'refinenet', 'indexnet']
DECODER_KERNEL_SIZE = 5
BACKBONE = 'mobilenetv2'  # choose in ['mobilenetv2', 'vgg16']
INDEXNET = 'depthwise'  # choose in ['holistic', 'depthwise']
INDEX_MODE = 'm2o'  # choose in ['o2o', 'm2o']
# training-related parameters
BATCH_SIZE = 5
CROP_SIZE = 320
LEARNING_RATE = 1e-5
MOMENTUM = 0.9
MULT = 100
NUM_EPOCHS = 50
NUM_CPU_WORKERS = 0
PRINT_EVERY = 1
RANDOM_SEED = 6
WEIGHT_DECAY = 1e-4
RECORD_EVERY = 20

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Deep-Image-Matting")
    # constant
    parser.add_argument("--image-scale", type=float, default=IMG_SCALE, help="Scale factor used in normalization.")
    parser.add_argument("--image-mean", type=float, default=IMG_MEAN, help="Mean used in normalization.")
    parser.add_argument("--image-std", type=float, default=IMG_STD, help="Std used in normalization.")
    parser.add_argument("--scales", type=int, default=SCALES, help="Scales of crop.")
    # system-related parameters
    parser.add_argument("--dataset", type=str, default=DATASET, help="Dataset type.")
    parser.add_argument("--exp", type=str, default=EXP, help="Experiment path.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIR, help="Path to the directory containing the dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--data-val-list", type=str, default=DATA_VAL_LIST,
                        help="Path to the file listing the images in the val dataset.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM, help="Where restore model parameters from.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR, help="Where to save snapshots of the model.")
    parser.add_argument("--result-dir", type=str, default=RESULT_DIR, help="Where to save inferred results.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--evaluate-only", action="store_true", help="Whether to perform evaluation.")
    # model-related parameters
    parser.add_argument("--output-stride", type=int, default=OUTPUT_STRIDE, help="Output stride of the model.")
    parser.add_argument("--conv-operator", type=str, default=CONV_OPERATOR,
                        help="Convolutional operator used in decoder.")
    parser.add_argument("--backbone", type=str, default=BACKBONE, help="Backbone used.")
    parser.add_argument("--decoder", type=str, default=DECODER, help="Decoder style.")
    parser.add_argument("--decoder-kernel-size", type=int, default=DECODER_KERNEL_SIZE, help="Decoder kernel size.")
    parser.add_argument("--indexnet", type=str, default=INDEXNET, choices=['holistic', 'depthwise'],
                        help="Use holistic or depthwise index networks.")
    parser.add_argument("--index-mode", type=str, default=INDEX_MODE, choices=['o2o', 'm2o'],
                        help="Type of depthwise index network.")
    parser.add_argument("--use-nonlinear", action="store_true", help="Whether to use nonlinearity in IndexNet.")
    parser.add_argument("--use-context", action="store_true", help="Whether to use context in IndexNet.")
    parser.add_argument("--apply-aspp", action="store_true", help="Whether to perform ASPP.")
    parser.add_argument("--sync-bn", action="store_true", help="Whether to apply synchronized batch normalization.")
    # training-related parameters
    parser.add_argument("--crop-size", type=int, default=CROP_SIZE, help="Size of crop.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE, help="Base learning rate for training.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM, help="Momentum component of the optimiser.")
    parser.add_argument("--mult", type=float, default=MULT, help="LR multiplier for pretrained layers.")
    parser.add_argument("--num-epochs", type=int, default=NUM_EPOCHS, help="Number of training steps.")
    parser.add_argument("--num-workers", type=int, default=NUM_CPU_WORKERS, help="Number of CPU cores used.")
    parser.add_argument("--print-every", type=int, default=PRINT_EVERY, help="Print information every often.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--record-every", type=int, default=RECORD_EVERY, help="Record loss every often.")
    return parser.parse_args()


def main():
    args = get_arguments()

    # seeding for reproducbility
    if paddle.device.is_compiled_with_cuda():
        paddle.seed(args.random_seed)
    paddle.seed(args.random_seed)
    # fix random seed bugs in numpy
    # np.random.seed(args.random_seed)

    # instantiate dataset
    dataset = AdobeImageMattingDataset

    snapshot_dir = os.path.join(args.snapshot_dir, args.dataset.lower(), args.exp)
    if not os.path.exists(snapshot_dir):
        os.makedirs(snapshot_dir)

    args.result_dir = os.path.join(args.result_dir, args.exp)
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    args.restore_from = os.path.join(args.snapshot_dir, args.dataset.lower(), args.exp, args.restore_from)

    arguments = vars(args)
    for item in arguments:
        print(item, ':\t', arguments[item])

    # instantiate network
    net_ = backbone[args.backbone]
    net = net_(
        pretrained=True,
        freeze_bn=True,
        output_stride=args.output_stride,
        input_size=args.crop_size,
        apply_aspp=args.apply_aspp,
        conv_operator=args.conv_operator,
        decoder=args.decoder,
        decoder_kernel_size=args.decoder_kernel_size,
        indexnet=args.indexnet,
        index_mode=args.index_mode,
        use_nonlinear=args.use_nonlinear,
        use_context=args.use_context,
        sync_bn=args.sync_bn
    )

    if args.backbone == 'mobilenetv2':
        net = paddle.DataParallel(net)
    if args.sync_bn:
        patch_replication_callback(net)
    # net.cuda()

    # filter parameters
    pretrained_params = []
    learning_params = []
    for p in net.named_parameters():
        if 'dconv' in p[0] or 'pred' in p[0] or 'index' in p[0]:
            learning_params.append(p[1])
        else:
            pretrained_params.append(p[1])

    # restored parameters
    start_epoch = 0
    if args.restore_from is not None:
        if os.path.isfile(args.restore_from):
            checkpoint = paddle.load(args.restore_from)
            net.set_state_dict(checkpoint['state_dict'])
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch']
        else:
            with open(os.path.join(args.result_dir, args.exp + '.txt'), 'a') as f:
                for item in arguments:
                    print(item, ':\t', arguments[item], file=f)
            print("==> no checkpoint found at '{}'".format(args.restore_from))
    resume_epoch = -1 if start_epoch == 0 else start_epoch

    # define optimizer scheduler
    scheduler = MultiStepDecay(args.learning_rate, milestones=[20, 26, 40], gamma=0.1, last_epoch=resume_epoch)

    # define optimizer
    optimizer = optim.Adam(
        parameters=[
            {'params': learning_params},
            {'params': pretrained_params, 'learning_rate':
                args.learning_rate / args.mult},
        ],
        learning_rate=scheduler
    )

    # restored parameters
    net.train_loss = {
        'running_loss': [],
        'epoch_loss': []
    }
    net.val_loss = {
        'running_loss': [],
        'epoch_loss': []
    }
    net.measure = {
        'sad': [],
        'mse': [],
        'grad': [],
        'conn': []
    }
    if start_epoch != 0:
        if 'optimizer' in checkpoint:
            optimizer.set_state_dict(checkpoint['optimizer'])
        if 'train_loss' in checkpoint:
            net.train_loss = checkpoint['train_loss']
        if 'val_loss' in checkpoint:
            net.val_loss = checkpoint['val_loss']
        if 'measure' in checkpoint:
            net.measure = checkpoint['measure']
        print("==> load checkpoint '{}' (epoch {})"
                .format(args.restore_from, start_epoch))
        

    # define transform
    transform_train_val = [
        RandomCrop(args.crop_size, args.scales),
        RandomFlip()
    ]
    transform_all = [
        Normalize(args.image_scale, args.image_mean, args.image_std),
        ToTensor()
    ]
    composed_transform_train = Compose(transform_train_val + transform_all)
    composed_transform_val = Compose(transform_all)

    # define dataset loader
    trainset = dataset(
        data_file=args.data_list,
        data_dir=args.data_dir,
        train=True,
        transform=composed_transform_train
    )
    train_loader = DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        worker_init_fn=worker_init_fn,
        # pin_memory=True,
        use_shared_memory=True,
        drop_last=True
    )
    valset = dataset(
        data_file=args.data_val_list,
        data_dir=args.data_dir,
        train=False,
        transform=composed_transform_val
    )
    val_loader = DataLoader(
        valset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        # pin_memory=True
        use_shared_memory=True
    )

    print('alchemy start...')
    if True:
        validate(net, val_loader, start_epoch + 1, args)
        return
    
    for epoch in range(start_epoch, args.num_epochs):
        np.random.seed()

        # validate(net, val_loader, epoch, args)

        # train
        start = time()
        train(net, train_loader, optimizer, epoch + 1, args)
        end = time()
        scheduler.step()

        print('lr: ', optimizer.get_lr())

        # val
        if (epoch + 1) % 10 == 0:
            validate(net, val_loader, epoch + 1, args)

        # save checkpoint
        state = {
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'train_loss': net.train_loss,
            'val_loss': net.val_loss,
            'measure': net.measure
        }
        save_checkpoint(state, snapshot_dir, filename='model_ckpt.pdparams', epoch=epoch + 1)
        print(args.exp + ' epoch {} finished!'.format(epoch + 1))
        if len(net.measure['grad']) > 1 and net.measure['grad'][-1] <= min(net.measure['grad'][:-1]):
            save_checkpoint(state, snapshot_dir, filename='model_best.pdparams')
    print('Experiments with ' + args.exp + ' done!')


if __name__ == "__main__":
    main()
