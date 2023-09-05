import torch.utils.data as data
from torchvision import transforms
from PIL import Image
import os
import torch
import torch as th
import numpy as np
from pathlib import Path
from einops import repeat

from dev_basics.utils import vid_io
from .util.mask import (bbox2mask, brush_stroke_mask, get_irregular_mask, random_bbox, random_cropping_bbox)

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset_video(dir):
    if os.path.isfile(dir):
        vids = [i for i in np.genfromtxt(dir, dtype=np.str, encoding='utf-8')]
    else:
        length = 0
        vids = {}
        assert os.path.isdir(dir), '%s is not a valid directory' % dir
        for vid_path in (Path(dir)/"JPEGImages").iterdir():
            vid_path = Path(vid_path)
            vid_name = vid_path.name
            vids[vid_name] = []
            for fname in sorted(vid_path.iterdir()):
                vids[vid_name].append(fname)
    return vids

def get_num_subvids(vids,nframes):
    total = 0
    for name in vids:
        ntotal = len(vids[name])
        nsubs = (ntotal-1)/nframes+1
        total += nsubs
    return total

def compute_name_from_index(vids,nframes):
    names = []
    fstarts = []
    total = 0
    for name in vids:
        ntotal = len(vids[name])
        sub_frames = nframes if nframes > 0 else ntotal
        nsubs = ntotal - (sub_frames-1)
        names.extend([name for _ in range(nsubs)])
        fstarts.extend([t for t in range(nsubs)])
    return names,fstarts

def make_dataset(dir):
    if os.path.isfile(dir):
        images = [i for i in np.genfromtxt(dir, dtype=np.str, encoding='utf-8')]
    else:
        images = []
        assert os.path.isdir(dir), '%s is not a valid directory' % dir
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)

    return images

def pil_loader(path):
    return Image.open(path).convert('RGB')

def pil_loader_anno(path):
    return Image.open(path)


class InpaintDataset(data.Dataset):
    def __init__(self, data_root, mask_config={}, data_len=-1, image_size=[256, 256], loader=pil_loader):
        assert data_len <= 0
        imgs = make_dataset(data_root)
        self.imgs = imgs
        self.names = list(self.imgs.keys())
        self.ntotal = get_num_subvids(self.imgs,nframes)
        self.tfs = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])
        self.loader = loader
        self.mask_config = mask_config
        self.mask_mode = self.mask_config['mask_mode']
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        path = self.imgs[index]
        img = self.tfs(self.loader(path))
        mask = self.get_mask()
        cond_image = img*(1. - mask) + mask*torch.randn_like(img)
        mask_img = img*(1. - mask) + mask

        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['mask_image'] = mask_img
        ret['mask'] = mask
        ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]
        return ret

    def __len__(self):
        return self.ntotal

    def get_mask(self):
        if self.mask_mode == 'bbox':
            mask = bbox2mask(self.image_size, random_bbox())
        elif self.mask_mode == 'center':
            h, w = self.image_size
            mask = bbox2mask(self.image_size, (h//4, w//4, h//2, w//2))
        elif self.mask_mode == 'irregular':
            mask = get_irregular_mask(self.image_size)
        elif self.mask_mode == 'free_form':
            mask = brush_stroke_mask(self.image_size)
        elif self.mask_mode == 'hybrid':
            regular_mask = bbox2mask(self.image_size, random_bbox())
            irregular_mask = brush_stroke_mask(self.image_size, )
            mask = regular_mask | irregular_mask
        elif self.mask_mode == 'file':
            pass
        else:
            raise NotImplementedError(
                f'Mask mode {self.mask_mode} has not been implemented.')
        return torch.from_numpy(mask).permute(2,0,1)
