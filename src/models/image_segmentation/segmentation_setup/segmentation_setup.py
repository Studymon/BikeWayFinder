from functools import partial
from os import path

from PIL import Image
import glob
import gzip
import io

import models
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
from inplace_abn import InPlaceABN, ABN
from torch.utils.data import Dataset
from torchvision.transforms import functional as tfn
from itertools import chain


#################################################	
#### Functions and classes for image segmentation
#################################################

# Needed for segmentation head
def try_index(scalar_or_list, i):
    try:
        return scalar_or_list[i]
    except TypeError:
        return scalar_or_list

# Segmentation head
class DeeplabV3(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels=256,
        dilations=(12, 24, 36),
        norm_act=ABN,
        pooling_size=None,
    ):
        super(DeeplabV3, self).__init__()
        self.pooling_size = pooling_size

        self.map_convs = nn.ModuleList(
            [
                nn.Conv2d(in_channels, hidden_channels, 1, bias=False),
                nn.Conv2d(
                    in_channels,
                    hidden_channels,
                    3,
                    bias=False,
                    dilation=dilations[0],
                    padding=dilations[0],
                ),
                nn.Conv2d(
                    in_channels,
                    hidden_channels,
                    3,
                    bias=False,
                    dilation=dilations[1],
                    padding=dilations[1],
                ),
                nn.Conv2d(
                    in_channels,
                    hidden_channels,
                    3,
                    bias=False,
                    dilation=dilations[2],
                    padding=dilations[2],
                ),
            ]
        )
        self.map_bn = norm_act(hidden_channels * 4)

        self.global_pooling_conv = nn.Conv2d(
            in_channels, hidden_channels, 1, bias=False
        )
        self.global_pooling_bn = norm_act(hidden_channels)

        self.red_conv = nn.Conv2d(hidden_channels * 4, out_channels, 1, bias=False)
        self.pool_red_conv = nn.Conv2d(hidden_channels, out_channels, 1, bias=False)
        self.red_bn = norm_act(out_channels)

        self.reset_parameters(self.map_bn.activation, self.map_bn.activation_param)

    def reset_parameters(self, activation, slope):
        gain = nn.init.calculate_gain(activation, slope)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data, gain)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, ABN):
                if hasattr(m, "weight") and m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Map convolutions
        out = torch.cat([m(x) for m in self.map_convs], dim=1)
        out = self.map_bn(out)
        out = self.red_conv(out)

        # Global pooling
        pool = self._global_pooling(x)
        pool = self.global_pooling_conv(pool)
        pool = self.global_pooling_bn(pool)
        pool = self.pool_red_conv(pool)
        if self.training or self.pooling_size is None:
            pool = pool.repeat(1, 1, x.size(2), x.size(3))

        out += pool
        out = self.red_bn(out)
        return out

    def _global_pooling(self, x):
        if self.training or self.pooling_size is None:
            pool = x.view(x.size(0), x.size(1), -1).mean(dim=-1)
            pool = pool.view(x.size(0), x.size(1), 1, 1)
        else:
            pooling_size = (
                min(try_index(self.pooling_size, 0), x.shape[2]),
                min(try_index(self.pooling_size, 1), x.shape[3]),
            )
            padding = (
                (pooling_size[1] - 1) // 2,
                (pooling_size[1] - 1) // 2
                if pooling_size[1] % 2 == 1
                else (pooling_size[1] - 1) // 2 + 1,
                (pooling_size[0] - 1) // 2,
                (pooling_size[0] - 1) // 2
                if pooling_size[0] % 2 == 1
                else (pooling_size[0] - 1) // 2 + 1,
            )

            pool = functional.avg_pool2d(x, pooling_size, stride=1)
            pool = functional.pad(pool, pad=padding, mode="replicate")
        return pool


# Function for flipping an image horizontally
def flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(
        x.size(dim) - 1, -1, -1, dtype=torch.long, device=x.device
    )
    return x[tuple(indices)]

# Function for loading a snapshot of the trained model
def load_snapshot(snapshot_file):
    """Load a training snapshot"""
    print("--- Loading model from snapshot")

    # Create network
    norm_act = partial(InPlaceABN, activation="leaky_relu", activation_param=0.01)
    body = models.__dict__["net_wider_resnet38_a2"](
        norm_act=norm_act, dilation=(1, 2, 4, 4)
    )
    head = DeeplabV3(4096, 256, 256, norm_act=norm_act, pooling_size=(84, 84))

    # Load snapshot and recover network state
    data = torch.load(snapshot_file)
    body.load_state_dict(data["state_dict"]["body"])
    head.load_state_dict(data["state_dict"]["head"])

    return body, head, data["state_dict"]["cls"]

# Image segmentation module
class SegmentationModule(nn.Module):
    _IGNORE_INDEX = 255

    class _MeanFusion:
        def __init__(self, x, classes):
            self.buffer = x.new_zeros(x.size(0), classes, x.size(2), x.size(3))
            self.counter = 0

        def update(self, sem_logits):
            probs = functional.softmax(sem_logits, dim=1)
            self.counter += 1
            self.buffer.add_((probs - self.buffer) / self.counter)

        def output(self):
            probs, cls = self.buffer.max(1)
            return probs, cls

    class _VotingFusion:
        def __init__(self, x, classes):
            self.votes = x.new_zeros(x.size(0), classes, x.size(2), x.size(3))
            self.probs = x.new_zeros(x.size(0), classes, x.size(2), x.size(3))

        def update(self, sem_logits):
            probs = functional.softmax(sem_logits, dim=1)
            probs, cls = probs.max(1, keepdim=True)

            self.votes.scatter_add_(1, cls, self.votes.new_ones(cls.size()))
            self.probs.scatter_add_(1, cls, probs)

        def output(self):
            cls, idx = self.votes.max(1, keepdim=True)
            probs = self.probs / self.votes.clamp(min=1)
            probs = probs.gather(1, idx)
            return probs.squeeze(1), cls.squeeze(1)

    class _MaxFusion:
        def __init__(self, x, _):
            self.buffer_cls = x.new_zeros(
                x.size(0), x.size(2), x.size(3), dtype=torch.long
            )
            self.buffer_prob = x.new_zeros(x.size(0), x.size(2), x.size(3))

        def update(self, sem_logits):
            probs = functional.softmax(sem_logits, dim=1)
            max_prob, max_cls = probs.max(1)

            replace_idx = max_prob > self.buffer_prob
            self.buffer_cls[replace_idx] = max_cls[replace_idx]
            self.buffer_prob[replace_idx] = max_prob[replace_idx]

        def output(self):
            return self.buffer_prob, self.buffer_cls

    def __init__(self, body, head, head_channels, classes, fusion_mode="mean"):
        super(SegmentationModule, self).__init__()
        self.body = body
        self.head = head
        self.cls = nn.Conv2d(head_channels, classes, 1)

        self.classes = classes
        if fusion_mode == "mean":
            self.fusion_cls = SegmentationModule._MeanFusion
        elif fusion_mode == "voting":
            self.fusion_cls = SegmentationModule._VotingFusion
        elif fusion_mode == "max":
            self.fusion_cls = SegmentationModule._MaxFusion

    def _network(self, x, scale):
        if scale != 1:
            scaled_size = [round(s * scale) for s in x.shape[-2:]]
            x_up = functional.interpolate(x, size=scaled_size, mode="bilinear", align_corners=False)
        else:
            x_up = x

        x_up = self.body(x_up)
        x_up = self.head(x_up)
        sem_logits = self.cls(x_up)

        del x_up
        return sem_logits

    def forward(self, x, scales, do_flip=False):
        out_size = x.shape[-2:]
        fusion = self.fusion_cls(x, self.classes)

        for scale in scales:
            # Main orientation
            sem_logits = self._network(x, scale)
            sem_logits = functional.interpolate(sem_logits, size=out_size, mode="bilinear", align_corners=False)
            fusion.update(sem_logits)

            # Flipped orientation
            if do_flip:
                # Main orientation
                sem_logits = self._network(flip(x, -1), scale)
                sem_logits = functional.upsample(
                    sem_logits, size=out_size, mode="bilinear"
                )
                fusion.update(flip(sem_logits, -1))

        return fusion.output()
    
# Class for transforming images
class SegmentationTransform:
    def __init__(self, longest_max_size, rgb_mean, rgb_std):
        self.longest_max_size = longest_max_size
        self.rgb_mean = rgb_mean
        self.rgb_std = rgb_std

    def __call__(self, img):
        # Scaling
        scale = self.longest_max_size / float(max(img.size[0], img.size[1]))
        if scale != 1.0:
            out_size = tuple(int(dim * scale) for dim in img.size)
            img = img.resize(out_size, resample=Image.BILINEAR)

        # Convert to torch and normalize
        img = tfn.to_tensor(img)
        img.sub_(img.new(self.rgb_mean).view(-1, 1, 1))
        img.div_(img.new(self.rgb_std).view(-1, 1, 1))

        return img
    
# Class for creating a dataset
class SegmentationDataset(Dataset):
    _EXTENSIONS = ["*.jpg", "*.jpeg", "*.png"]

    def __init__(self, in_dir, transform):
        super(SegmentationDataset, self).__init__()

        self.in_dir = in_dir
        self.transform = transform

        # Find all images
        self.images = []
        for img_path in chain(
            *(
                glob.iglob(path.join(self.in_dir, ext))
                for ext in SegmentationDataset._EXTENSIONS
            )
        ):
            _, name_with_ext = path.split(img_path)
            idx, _ = path.splitext(name_with_ext)
            self.images.append({"idx": idx, "path": img_path})

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        # Load image
        with Image.open(self.images[item]["path"]) as img_raw:
            size = img_raw.size
            img = self.transform(img_raw.convert(mode="RGB"))

        return {"img": img, "meta": {"idx": self.images[item]["idx"], "size": size}}

# Function for collating items in the dataloader
def segmentation_collate(items):
    imgs = torch.stack([item["img"] for item in items])
    metas = [item["meta"] for item in items]

    return {"img": imgs, "meta": metas}


### Palette and functions for creating output images

_PALETTE = np.array(
    [
        [165, 42, 42],
        [0, 192, 0],
        [196, 196, 196],
        [190, 153, 153],
        [180, 165, 180],
        [90, 120, 150],
        [102, 102, 156],
        [128, 64, 255],
        [140, 140, 200],
        [170, 170, 170],
        [250, 170, 160],
        [96, 96, 96],
        [230, 150, 140],
        [128, 64, 128],
        [110, 110, 110],
        [244, 35, 232],
        [150, 100, 100],
        [70, 70, 70],
        [150, 120, 90],
        [220, 20, 60],
        [255, 0, 0],
        [255, 0, 100],
        [255, 0, 200],
        [200, 128, 128],
        [255, 255, 255],
        [64, 170, 64],
        [230, 160, 50],
        [70, 130, 180],
        [190, 255, 255],
        [152, 251, 152],
        [107, 142, 35],
        [0, 170, 30],
        [255, 255, 128],
        [250, 0, 30],
        [100, 140, 180],
        [220, 220, 220],
        [220, 128, 128],
        [222, 40, 40],
        [100, 170, 30],
        [40, 40, 40],
        [33, 33, 33],
        [100, 128, 160],
        [142, 0, 0],
        [70, 100, 150],
        [210, 170, 100],
        [153, 153, 153],
        [128, 128, 128],
        [0, 0, 80],
        [250, 170, 30],
        [192, 192, 192],
        [220, 220, 0],
        [140, 140, 20],
        [119, 11, 32],
        [150, 0, 255],
        [0, 60, 100],
        [0, 0, 142],
        [0, 0, 90],
        [0, 0, 230],
        [0, 80, 100],
        [128, 64, 64],
        [0, 0, 110],
        [0, 0, 70],
        [0, 0, 192],
        [32, 32, 32],
        [120, 10, 10],
    ],
    dtype=np.uint8,
)

_PALETTE = np.concatenate(
    [_PALETTE, np.zeros((256 - _PALETTE.shape[0], 3), dtype=np.uint8)], axis=0
)

# Function for creating a prediction image
def get_pred_image(tensor, out_size, with_palette):
    tensor = tensor.numpy()
    if with_palette:
        img = Image.fromarray(tensor.astype(np.uint8), mode="P")
        img.putpalette(_PALETTE)
    else:
        img = Image.fromarray(tensor.astype(np.uint8), mode="L")

    return img.resize(out_size, Image.NEAREST)

# Function for saving and loading compressed tensors
def save_compressed_tensor(tensor, filename):
    # Create an in-memory buffer
    buffer = io.BytesIO()
    # Save the tensor to the buffer
    torch.save(tensor, buffer)
    # Move to the beginning of the buffer
    buffer.seek(0)
    
    # Compress and write the buffer content to a file
    with gzip.open(filename + '.gz', 'wb') as f:
        f.write(buffer.read())
        
def load_compressed_tensor(filename):
    with gzip.open(filename + '.gz', 'rb') as f:
        buffer = f.read()
    tensor = torch.load(io.BytesIO(buffer))
    return tensor
                