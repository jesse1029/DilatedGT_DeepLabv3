import torch,cv2
import random
import numpy as np

from PIL import Image, ImageOps, ImageFilter


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img = np.array(sample['image']).astype(np.float32)
        MI = sample['mean']
        mask = sample['label']
        mask = np.array(mask).astype(np.float32)
        img = img - MI[:,:,::-1]
        img /= 255.0
        img -= self.mean
        img /= self.std

        return {'image': img,
                'label': mask}

    
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = sample['image']
        mask = sample['label']
        img = np.array(img).astype(np.float32).transpose((2, 0, 1))
        mask = np.array(mask).astype(np.float32)

        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).float()

        return {'image': img,
                'label': mask}
    


class RandomHorizontalFlip(object):
    def __call__(self, sample):
        img = np.array(sample['image'])
        mask = np.array(sample['label'])
        MI = np.array(sample['mean'])
        
        if random.random() < 0.5:
            img = img[:, ::-1, :]
            mask = mask[:, ::-1]
            MI = MI[:, ::-1, :]
        if random.random() < 0.5:
            img = img[::-1, :, :]
            mask = mask[::-1, :]
            MI = MI[::-1, :, :]
        img = Image.fromarray(img)
        mask = Image.fromarray(mask)
        return {'image': img,
                'label': mask,
                'mean': MI}


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        rotate_degree = random.uniform(-1*self.degree, self.degree)
        img = img.rotate(rotate_degree, Image.BILINEAR)
        mask = mask.rotate(rotate_degree, Image.NEAREST)

        return {'image': img,
                'label': mask,
                'mean': MI}


class RandomGaussianBlur(object):
    def __call__(self, sample):
        global M, MI
        img = sample['image']
        mask = sample['label']
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))

        return {'image': img,
                'label': mask,
                'mean': sample['mean']}

class RandomScaleCrop(object):
    def __init__(self, base_size, crop_size, fill=0):
        self.base_size = base_size
        self.crop_size = crop_size
        self.fill = fill

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        # random scale (short edge)
        MI = np.load('meanimage.npy')

        short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        MI = cv2.resize(MI, dsize=(ow, oh))
        # pad crop
        if short_size < self.crop_size:
            padh = self.crop_size - oh if oh < self.crop_size else 0
            padw = self.crop_size - ow if ow < self.crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=self.fill)
            pm = np.zeros((oh+padh, ow+padw, 3))
            pm[:oh, :ow, :] = MI
            MI = pm
            
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - self.crop_size)
        y1 = random.randint(0, h - self.crop_size)
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        MI = MI[y1:y1+self.crop_size, x1:x1+self.crop_size,:]

        return {'image': img,
                'label': mask,
                'mean': MI}


class FixScaleCrop(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        MI = np.load('meanimage.npy')

        w, h = img.size
        if w > h:
            oh = self.crop_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = self.crop_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        MI = cv2.resize(MI, dsize=(ow, oh))
        # center crop
        w, h = img.size
        x1 = int(round((w - self.crop_size) / 2.))
        y1 = int(round((h - self.crop_size) / 2.))
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        MI = MI[y1:y1+self.crop_size, x1:x1+self.crop_size,:]

        return {'image': img,
                'label': mask,
                'mean': MI}

class FixedResize(object):
    def __init__(self, size):
        self.size = (size, size)  # size: (h, w)

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        MI = sample['mean']

        assert img.size == mask.size

        img = img.resize(self.size, Image.BILINEAR)
        mask = mask.resize(self.size, Image.NEAREST)
        MI = cv2.resize(MI, dsize=self.size)

        return {'image': img,
                'label': mask,
                'mean': MI}