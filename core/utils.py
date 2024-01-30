import numpy as np
import cv2 as cv
from scipy.ndimage import gaussian_filter, morphology
from skimage.measure import label, regionprops


# compute the SAD error given a pdiction, a ground truth and a mask
def compute_sad_loss(pd, gt, mask):
    cv.normalize(pd, pd, 0.0, 255.0, cv.NORM_MINMAX, dtype=cv.CV_32F)
    cv.normalize(gt, gt, 0.0, 255.0, cv.NORM_MINMAX, dtype=cv.CV_32F)
    error_map = np.abs(pd - gt) / 255.
    loss = np.sum(error_map * mask)
    # the loss is scaled by 1000 due to the large images
    loss = loss / 1000
    return loss


# compute the MSE error
def compute_mse_loss(pd, gt, mask):
    cv.normalize(pd, pd, 0.0, 255.0, cv.NORM_MINMAX, dtype=cv.CV_32F)
    cv.normalize(gt, gt, 0.0, 255.0, cv.NORM_MINMAX, dtype=cv.CV_32F)
    error_map = (pd - gt) / 255.
    loss = np.sum(np.square(error_map) * mask) / np.sum(mask)
    return loss


# compute the gradient error
def compute_gradient_loss(pd, gt, mask):
    cv.normalize(pd, pd, 0.0, 255.0, cv.NORM_MINMAX, dtype=cv.CV_32F)
    cv.normalize(gt, gt, 0.0, 255.0, cv.NORM_MINMAX, dtype=cv.CV_32F)
    pd = pd / 255.
    gt = gt / 255.
    pd_x = gaussian_filter(pd, sigma=1.4, order=[1, 0], output=np.float32)
    pd_y = gaussian_filter(pd, sigma=1.4, order=[0, 1], output=np.float32)
    gt_x = gaussian_filter(gt, sigma=1.4, order=[1, 0], output=np.float32)
    gt_y = gaussian_filter(gt, sigma=1.4, order=[0, 1], output=np.float32)
    pd_mag = np.sqrt(pd_x**2 + pd_y**2)
    gt_mag = np.sqrt(gt_x**2 + gt_y**2)

    error_map = np.square(pd_mag - gt_mag)
    loss = np.sum(error_map * mask) / 10
    return loss


# compute the connectivity error
def compute_connectivity_loss(pd, gt, mask, step=0.1):
    cv.normalize(pd, pd, 0, 255, cv.NORM_MINMAX)
    cv.normalize(gt, gt, 0, 255, cv.NORM_MINMAX)
    pd = pd / 255.
    gt = gt / 255.

    h, w = pd.shape

    thresh_steps = np.arange(0, 1.1, step)
    l_map = -1 * np.ones((h, w), dtype=np.float32)
    lambda_map = np.ones((h, w), dtype=np.float32)
    for i in range(1, thresh_steps.size):
        pd_th = pd >= thresh_steps[i]
        gt_th = gt >= thresh_steps[i]

        label_image = label(pd_th & gt_th, connectivity=1)
        cc = regionprops(label_image)
        size_vec = np.array([c.area for c in cc])
        if len(size_vec) == 0:
            continue
        max_id = np.argmax(size_vec)
        coords = cc[max_id].coords

        omega = np.zeros((h, w), dtype=np.float32)
        omega[coords[:, 0], coords[:, 1]] = 1

        flag = (l_map == -1) & (omega == 0)
        l_map[flag == 1] = thresh_steps[i-1]

        dist_maps = morphology.distance_transform_edt(omega==0)
        dist_maps = dist_maps / dist_maps.max()
        # lambda_map[flag == 1] = dist_maps.mean()
    l_map[l_map == -1] = 1
    
    # the definition of lambda is ambiguous
    d_pd = pd - l_map
    d_gt = gt - l_map
    # phi_pd = 1 - lambda_map * d_pd * (d_pd >= 0.15).astype(np.float32)
    # phi_gt = 1 - lambda_map * d_gt * (d_gt >= 0.15).astype(np.float32)
    phi_pd = 1 - d_pd * (d_pd >= 0.15).astype(np.float32)
    phi_gt = 1 - d_gt * (d_gt >= 0.15).astype(np.float32)
    loss = np.sum(np.abs(phi_pd - phi_gt) * mask) / 1000
    return loss


def image_alignment(x, output_stride, odd=False):
    imsize = np.asarray(x.shape[:2], dtype=np.float64)
    if odd:
        new_imsize = np.ceil(imsize / output_stride) * output_stride + 1
    else:
        new_imsize = np.ceil(imsize / output_stride) * output_stride
    h, w = int(new_imsize[0]), int(new_imsize[1])

    x1 = x[:, :, 0:3]
    x2 = x[:, :, 3]
    new_x1 = cv.resize(x1, dsize=(w,h), interpolation=cv.INTER_CUBIC)
    new_x2 = cv.resize(x2, dsize=(w,h), interpolation=cv.INTER_NEAREST)

    new_x2 = np.expand_dims(new_x2, axis=2)
    new_x = np.concatenate((new_x1, new_x2), axis=2)

    return new_x


def image_rescale(x, scale):
    x1 = x[:, :, 0:3]
    x2 = x[:, :, 3]
    new_x1 = cv.resize(x1, None, fx=scale, fy=scale, interpolation=cv.INTER_CUBIC)
    new_x2 = cv.resize(x2, None, fx=scale, fy=scale, interpolation=cv.INTER_NEAREST)
    new_x2 = np.expand_dims(new_x2, axis=2)
    new_x = np.concatenate((new_x1,new_x2), axis=2)
    return new_x
