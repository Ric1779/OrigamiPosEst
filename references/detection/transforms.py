import random
import torch
import torch.nn.functional as Q

from torchvision.transforms import functional as F


def _flip_coco_person_keypoints(kps, width):
    flip_inds = [8,7,6,5,4,3,2,1,0]
    flipped_data = kps[:, flip_inds]
    flipped_data[..., 0] = width - flipped_data[..., 0]
    # Maintain COCO convention that if visibility == 0, then x, y = 0
    inds = flipped_data[..., 2] == 0
    flipped_data[inds] = 0
    return flipped_data


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox
            if "masks" in target:
                target["masks"] = target["masks"].flip(-1)
            if "keypoints" in target:
                keypoints = target["keypoints"]
                keypoints = _flip_coco_person_keypoints(keypoints, width)
                target["keypoints"] = keypoints
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target

class Rescale(object):
 
    def __call__(self, image, target):
        height, width = image.shape[-2:]
        image = image.view(1,3,height,width)
        image = Q.interpolate(image,(512,512),mode='bilinear')
        image = image.view(3,512,512)
        bbox = target["boxes"]
        M = round(512/height)
        bbox[:, [0,2]] = bbox[:,[0,2]]*M
        bbox[:, [1,3]] = bbox[:,[1,3]]*M
        target["boxes"] = bbox
        if "masks" in target:
            target["masks"] = Q.interpolate(target["masks"],(1,3,512,512),mode='bilinear')
        if "keypoints" in target:
            keypoints = target["keypoints"]
            keypoints = keypoints[:,:]*M
            inds = keypoints[..., 2] == 0
            keypoints[inds] = 0
            target["keypoints"] = keypoints
        return image, target