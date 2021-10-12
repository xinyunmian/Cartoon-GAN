import torch
from torch.nn import functional as F
from GetFaceMask.maskUtils.FaceParser import FaceParser
# read image
def parsing_feat(parsing_img):
    input_tensor = F.interpolate(parsing_img, size=(512, 512))
    input_tensor = torch.squeeze(input_tensor, 0)
    input_tensor = input_tensor * 0.5 + 0.5
    faceparser = FaceParser()
    # out_feature = faceparser.parse(input_tensor)[1]
    out_map = faceparser.parse(input_tensor)[0]
    return out_map

if __name__ == "__main__":
    img = torch.randn(1, 3, 256, 256).cuda()
    out = parsing_feat(img)
    print(out.shape)
