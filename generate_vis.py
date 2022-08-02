import math

from PIL import Image
import requests
import matplotlib.pyplot as plt

import ipywidgets as widgets
from IPython.display import display, clear_output

import torch
from torch import nn
from torchvision.models import resnet50
import torchvision.transforms as T
torch.set_grad_enabled(False);
from models import build_model
import shutil, random, os
import json
from pycocotools.coco import COCO
from datasets.coco import make_coco_transforms, ConvertCocoPolysToMask
import numpy as np
random.seed(0)
val_path = "/nobackup/yiwei/coco/images/val2017"
save_path = "/nobackup/yiwei/coco/images/detr_focal_loss_hm"
# save_path_2 = "/nobackup/yiwei/coco/images/20_conddetr_att"

def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size

    a1  = 1
    b1  = (height + width)
    c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1  = (b1 + sq1) / 2

    a2  = 4
    b2  = 2 * (height + width)
    c2  = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2  = (b2 + sq2) / 2

    a3  = 4 * min_overlap
    b3  = -2 * min_overlap * (height + width)
    c3  = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3  = (b3 + sq3) / 2
    return min(r1, r2, r3)

def draw_gaussian(heatmap, center, sigma):
    tmp_size = sigma * 3
    mu_x = int(center[0] + 0.5)
    mu_y = int(center[1] + 0.5)
    w, h = heatmap.shape[0], heatmap.shape[1]
    ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
    br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
    if ul[0] >= h or ul[1] >= w or br[0] < 0 or br[1] < 0:
        return heatmap
    size = 2 * tmp_size + 1
    x = np.arange(0, size, 1, np.float32)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    g_x = max(0, -ul[0]), min(br[0], h) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], w) - ul[1]
    img_x = max(0, ul[0]), min(br[0], h)
    img_y = max(0, ul[1]), min(br[1], w)
    heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]] = np.maximum(
        heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]],
        g[g_y[0]:g_y[1], g_x[0]:g_x[1]])
    return heatmap

def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1,-n:n+1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)
    gaussian = torch.from_numpy(gaussian)
    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap  = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0: # TODO debug
        torch.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap

def _neg_loss(pred, gt):
    ''' Modified focal loss. Exactly the same as CornerNet.
        Runs faster and costs a little bit more memory
        Arguments:
        pred (batch x c x h x w)
        gt_regr (batch x c x h x w)
    '''
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4)

    loss = 0

    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    num_pos  = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss

# COCO classes
CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

# standard PyTorch mean-std input image normalization
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

def plot_results(pil_img, prob, boxes):
    plt.figure()
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        cl = p.argmax()
        text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.show()

def plot_results2(pil_img, boxes):
    plt.figure()
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    for (xmin, ymin, xmax, ymax), c in zip(boxes.tolist(), colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
    plt.axis('off')
    plt.show()

checkpoint = torch.load("/nobackup/yiwei/DistributedQueries/exps/r50_deformable_detr_vis/checkpoint.pth")

args = checkpoint['args']
args.num_feature_levels = 1
model, criterion, postprocessors = build_model(args)
model.load_state_dict(checkpoint['model'])
model.eval()

coco = COCO("/nobackup/yiwei/coco/annotations/instances_val2017.json")
prepare = ConvertCocoPolysToMask(False)
transform_val = make_coco_transforms('val')

filenames = ['000000427338.jpg', '000000424975.jpg', '000000120853.jpg', '000000099810.jpg', '000000370818.jpg', '000000016502.jpg', '000000416256.jpg', '000000338905.jpg', '000000423617.jpg',  '000000295138.jpg', '000000066523.jpg', '000000031269.jpg', '000000014439.jpg', '000000453584.jpg', '000000009914.jpg', '000000210230.jpg', '000000306136.jpg', '000000263425.jpg', '000000288042.jpg', '000000396526.jpg', '000000016598.jpg', '000000323799.jpg', '000000159282.jpg', '000000474078.jpg', '000000564280.jpg', '000000175387.jpg', '000000223959.jpg', '000000492110.jpg', '000000186345.jpg', '000000106757.jpg', '000000495732.jpg', '000000495054.jpg', '000000218249.jpg', '000000537964.jpg', '000000050165.jpg', '000000163746.jpg', '000000020247.jpg', '000000500565.jpg', '000000287527.jpg', '000000365207.jpg', '000000068833.jpg', '000000499181.jpg', '000000521141.jpg', '000000434996.jpg', '000000281179.jpg', '000000214200.jpg']


# filenames = random.sample(os.listdir(val_path), 50)
for fname in filenames:
    srcpath = os.path.join(val_path, fname)
    im = Image.open(srcpath)
    # mean-std normalize the input image (batch-size: 1)
    img = transform(im).unsqueeze(0)

    # propagate through the model
    outputs = model(img)
    bs, c, h, w = img.shape
    img_id = int(fname[:-4])
    ann_ids = coco.getAnnIds(imgIds=img_id)
    ann = coco.loadAnns(ann_ids)
    # target = {'image_id': img_id, 'annotations': ann}
    # img, target = prepare(transform(im), target)
    # img, target = transform_val(img, target)
    ann = [obj for obj in ann if 'iscrowd' not in obj or obj['iscrowd'] == 0]
    boxes = [obj["bbox"] for obj in ann]
    # guard against no boxes via resizing
    boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
    boxes[:, 2:] += boxes[:, :2]
    boxes[:, 0::2].clamp_(min=0, max=w)
    boxes[:, 1::2].clamp_(min=0, max=h)
    keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
    boxes = boxes[keep]
    target = {}
    target["boxes"] = boxes
    target = transform_val(target)
    print(target)
    exit(0)

    # plot_results2(im, rescale_bboxes(target['boxes'], im.size))

    # keep only predictions with 0.7+ confidence
    # probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    # keep = probas.max(-1).values > 0.5

    # convert boxes from [0; 1] to image scales
    # bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)

    # plot_results(im, probas[keep], bboxes_scaled)
    # plt.savefig(os.path.join(save_path, fname))

    # use lists to store the outputs via up-values
    conv_features, enc_attn_weights, dec_attn_weights = [], [], []

    hooks = [
        model.backbone[-2].register_forward_hook(
            lambda self, input, output: conv_features.append(output)
        ),
        # model.transformer.encoder.layers[-1].self_attn.register_forward_hook(
        #     lambda self, input, output: enc_attn_weights.append(output[1])
        # ),
        model.transformer.decoder.layers[5].multihead_attn.register_forward_hook(
            lambda self, input, output: dec_attn_weights.append(output[1])
        ),
    ]

    # propagate through the model
    outputs = model(img)

    for hook in hooks:
        hook.remove()
    # don't need the list anymore
    conv_features = conv_features[0]
    # enc_attn_weights = enc_attn_weights[0]
    dec_attn_weights = dec_attn_weights
    # print(dec_attn_weights[0].shape)

    # get the feature map shape
    h, w = conv_features['0'].tensors.shape[-2:]
    # if len(keep.nonzero()) == 0:
    #     continue

    # print(dec_attn_weights[5].shape)
    # print(outputs['pred_hms'][0].view(h,w).shape)

    fig, axs = plt.subplots(ncols=4, nrows=1, squeeze=False, figsize=(5, 5))
    colors = COLORS * 100

    for row in range(1):
        for col in range(4):
            ax = axs[row][col]
            if col == 0:
                ax.imshow(im)
                # keep only predictions with 0.7+ confidence
                # probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
                # keep = probas.max(-1).values > 0.5

                # convert boxes from [0; 1] to image scales
                bboxes_scaled = rescale_bboxes(target['boxes'], im.size)
                for (xmin, ymin, xmax, ymax), c in zip(bboxes_scaled.tolist(), colors):
                    ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                            fill=False, color=c, linewidth=3))
                # for p, (xmin, ymin, xmax, ymax), c in zip(probas[keep], bboxes_scaled.tolist(), colors):
                #     ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                #                             fill=False, color=c, linewidth=3))
                #     cl = p.argmax()
                #     text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
                #     ax.text(xmin, ymin, text, fontsize=15,
                #             bbox=dict(facecolor='yellow', alpha=0.5))
                ax.set_title("gt")
                ax.axis('off')
            elif col ==1:
                ax.imshow(dec_attn_weights[0][0,0,:].view(h, w))
                ax.set_title(f"decoder")
                ax.axis('off')
            elif col ==2:
                # print(outputs['pred_hms'][0].view(h, w).numpy())
                # exit(0)
                ax.imshow(outputs['pred_hms'][0].view(h, w).numpy())
                ax.set_title(f"output")
                ax.axis('off')
            else:
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = max(0, int(radius))
                target = target['boxes']
                hm = torch.zeros((h, w))
                for j in range(target.size()[0]):
                    ct = np.array([target[j][0]*w, target[j][1]*h], dtype=np.float32)
                    ct_int = ct.astype(np.int32)
                    draw_umich_gaussian(hm, ct_int, radius)
                ax.imshow(hm)
                ax.set_title(f"gt_hm")
                ax.axis('off')
    fig.tight_layout()
    plt.savefig(os.path.join(save_path, fname))

    # for ax_i in axs.T:
    #     ax = ax_i[0]
    #     # print(dec_attn_weights[0].shape)
    #     if counter == 0:
    #         ax.imshow(im)
    #         for p, (xmin, ymin, xmax, ymax), c in zip(probas[keep], bboxes_scaled.tolist(), colors):
    #             ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
    #                                     fill=False, color=c, linewidth=3))
    #             cl = p.argmax()
    #             text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
    #             ax.text(xmin, ymin, text, fontsize=15,
    #                     bbox=dict(facecolor='yellow', alpha=0.5))
    #         ax.axis('off')
    #     else:
    #         ax.imshow(dec_attn_weights[counter - 1][0, 0].view(h, w))
    #         ax.axis('off')
    #     counter += 1
    # fig.tight_layout()
    # plt.savefig(os.path.join(save_path, fname))
    # for idx, ax_i, (xmin, ymin, xmax, ymax) in zip(keep.nonzero(), axs.T, bboxes_scaled):
    #     ax = ax_i[0]
    #     ax.imshow(dec_attn_weights[0, idx].view(h, w))
    #     ax.axis('off')
    #     ax.set_title(f'query id: {idx.item()}')
    #     ax = ax_i[1]
    #     ax.imshow(im)
    #     ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
    #                             fill=False, color='blue', linewidth=3))
    #     ax.axis('off')
    #     ax.set_title(str(CLASSES[probas[idx].argmax()])+"   "+"{:.3f}".format(probas.max(-1).values[idx].item()))
    # fig.tight_layout()
    # plt.savefig(os.path.join(save_path_2, fname))

