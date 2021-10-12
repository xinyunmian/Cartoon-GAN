import torch
import numpy as np
from GetFaceMask.FaceMask import FaceMask
from torch.nn import functional as F
import torchvision.transforms as transforms
# read image
def parser(parsing_img):
    parsing_img = torch.squeeze(parsing_img, 0)  # [3, 256, 256]
    img_tensor = F.interpolate(parsing_img, scale_factor=2, mode='linear', align_corners=True)
    faceMask = FaceMask()
    # Get hair mask
    part_mask = faceMask.get_part_mask_from_image(img_tensor, "hair")
    part_mask.squeeze(0)
    # cover face
    part_mask1 = part_mask
    part_mask2 = part_mask
    out_mask = torch.stack((part_mask, part_mask1, part_mask2), 0)
    hair_mask = out_mask * img_tensor
    # transform = transforms.Compose([
    #             transforms.Resize((256, 256)),
    #             transforms.ToTensor(),
    #             transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
    # tensor_face_mask = transform(face_mask)
    return hair_mask.unsqueeze(0)




