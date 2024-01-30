import os
from time import time

import cv2 as cv
import numpy as np
from PIL import Image

import paddle
from core.utils import *

def save_checkpoint(state, snapshot_dir, filename='model_ckpt.pdparams', epoch=None):
    if os.path.isfile(os.path.join(snapshot_dir, filename)) & (filename=='model_ckpt.pdparams'):
        os.rename(os.path.join(snapshot_dir, filename), os.path.join(snapshot_dir, 'model_ckpt_{}.pdparams'.format(epoch-1)))
    paddle.save(state, '{}/{}'.format(snapshot_dir, filename))


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

# implementation of the composition loss and alpha loss
def weighted_loss(pd, gt, wl=0.5, epsilon=1e-6):
    bs, _, h, w = pd.shape
    mask = gt[:, 1, :, :].reshape([bs, 1, h, w])
    alpha_gt = gt[:, 0, :, :].reshape([bs, 1, h, w])
    diff_alpha = (pd - alpha_gt) * mask
    loss_alpha = paddle.sqrt(diff_alpha * diff_alpha + epsilon ** 2)
    loss_alpha = loss_alpha.sum(axis=2).sum(axis=2) / mask.sum(axis=2).sum(axis=2)
    loss_alpha = loss_alpha.sum() / bs

    fg = gt[:, 2:5, :, :]
    bg = gt[:, 5:8, :, :]
    c_p = pd * fg + (1 - pd) * bg
    c_g = gt[:, 8:11, :, :]
    diff_color = (c_p - c_g) * mask
    loss_composition = paddle.sqrt(diff_color * diff_color + epsilon ** 2)
    loss_composition = loss_composition.sum(axis=2).sum(axis=2) / mask.sum(axis=2).sum(axis=2)
    loss_composition = loss_composition.sum() / bs

    return wl * loss_alpha + (1 - wl) * loss_composition


def train(net, train_loader, optimizer, epoch, args):
    # switch to train mode
    net.train()

    running_loss = 0.0
    avg_frame_rate = 0.0
    start = time()
    for i, sample in enumerate(train_loader):
        inputs, targets = paddle.to_tensor(sample['image']), paddle.to_tensor(sample['alpha'])
        inputs, targets = inputs.cuda(), targets.cuda()
        # forward
        outputs = net(inputs)
        # zero the parameter gradients
        optimizer.clear_grad()
        # compute loss
        loss = weighted_loss(outputs, targets)
        # backward + optimize
        loss.backward()
        optimizer.step()
        # collect and print statistics
        running_loss += loss.item()

        end = time()
        running_frame_rate = args.batch_size * float(1 / (end - start))
        avg_frame_rate = (avg_frame_rate * i + running_frame_rate) / (i + 1)
        if i % args.record_every == args.record_every - 1:
            net.train_loss['running_loss'].append(running_loss / (i + 1))
        if i % args.print_every == args.print_every - 1:
            print('epoch: %d, train: %d/%d, '
                  'loss: %.5f, frame: %.2fHz/%.2fHz' % (
                      epoch,
                      i + 1,
                      len(train_loader),
                      running_loss / (i + 1),
                      running_frame_rate,
                      avg_frame_rate
                  ))
        start = time()
    net.train_loss['epoch_loss'].append(running_loss / (i + 1))


def validate(net, val_loader, epoch, args):
    # switch to eval mode
    net.eval()

    image_list = [name.split('\t') for name in open(args.data_val_list).read().splitlines()]

    epoch_result_dir = os.path.join(args.result_dir, str(epoch))
    if not os.path.exists(epoch_result_dir):
        os.makedirs(epoch_result_dir)

    with paddle.no_grad():
        sad = []
        mse = []
        grad = []
        conn = []
        avg_frame_rate = 0.0
        # scale = 0.5
        stride = args.output_stride
        start = time()
        for i, sample in enumerate(val_loader):
            image, targets = sample['image'], sample['alpha']

            h, w = image.shape[2:]
            image = image.squeeze().numpy().transpose(1, 2, 0)
            # image = image_rescale(image, scale)
            image = image_alignment(image, stride, odd=args.crop_size % 2 == 1)
            inputs = paddle.to_tensor(np.expand_dims(image.transpose(2, 0, 1), axis=0))

            # inference
            outputs = net(inputs.cuda()).squeeze().cpu().numpy()

            alpha = cv.resize(outputs, dsize=(w, h), interpolation=cv.INTER_CUBIC)
            alpha = np.clip(alpha, 0, 1) * 255.
            trimap = targets[:, 1, :, :].squeeze().numpy()
            mask = np.equal(trimap, 128).astype(np.float32)

            alpha = (1 - mask) * trimap + mask * alpha
            gt_alpha = targets[:, 0, :, :].squeeze().numpy() * 255.

            _, image_name = os.path.split(image_list[i][0])
            Image.fromarray(alpha.astype(np.uint8)).save(
                os.path.join(epoch_result_dir, image_name)
            )
            # Image.fromarray(alpha.astype(np.uint8)).show()

            # compute loss
            sad.append(compute_sad_loss(alpha, gt_alpha, mask))
            mse.append(compute_mse_loss(alpha, gt_alpha, mask))
            grad.append(compute_gradient_loss(alpha, gt_alpha, mask))
            conn.append(compute_connectivity_loss(alpha, gt_alpha, mask))

            end = time()
            running_frame_rate = 1 * float(1 / (end - start))  # batch_size = 1
            avg_frame_rate = (avg_frame_rate * i + running_frame_rate) / (i + 1)
            if i % args.print_every == args.print_every - 1:
                print(
                    'epoch: {0}, test: {1}/{2}, sad: {3:.2f}, SAD: {4:.2f}, MSE: {5:.4f}, '
                    'Grad: {6:.2f}, Conn: {7:.2f}, frame: {8:.2f}Hz/{9:.2f}Hz'
                    .format(epoch, i + 1, len(val_loader), sad[-1], np.mean(sad), np.mean(mse),
                            np.mean(grad), np.mean(conn), running_frame_rate, avg_frame_rate)
                )
            start = time()
    # write to files
    with open(os.path.join(args.result_dir, args.exp + '.txt'), 'a') as f:
        print(
            'epoch: {0}, test: {1}/{2}, SAD: {3:.2f}, MSE: {4:.4f}, Grad: {5:.2f}, Conn: {6:.2f}'
            .format(epoch, i + 1, len(val_loader), np.mean(sad), np.mean(mse), np.mean(grad), np.mean(conn)),
            file=f
        )
    # save stats
    net.val_loss['epoch_loss'].append(np.mean(sad))
    net.measure['sad'].append(np.mean(sad))
    net.measure['mse'].append(np.mean(mse))
    net.measure['grad'].append(np.mean(grad))
    net.measure['conn'].append(np.mean(conn))

