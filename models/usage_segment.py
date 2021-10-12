import os
import torch
import torch.nn.functional as F
from U2.model import U2NET

def segment(img):
    model_dir = os.path.join(os.getcwd(), 'U2', 'saved_models', 'u2net.pth')
    im = F.interpolate(img, size=(320, 320))
    im = im * 0.5 + 0.5
    # im = im[:, 0, :, :]
    im[:, 0, :, :] = (im[:, 0, :, :] - 0.485) / 0.229
    im[:, 1, :, :] = (im[:, 1, :, :] - 0.456) / 0.224
    im[:, 2, :, :] = (im[:, 2, :, :] - 0.406) / 0.225

    net = U2NET(3, 1)
    if torch.cuda.is_available():
        net.load_state_dict(torch.load(model_dir))
        net.cuda()
    else:
        net.load_state_dict(torch.load(model_dir, map_location='cpu'))
    net.eval()
    with torch.no_grad():

        d1,d2,d3,d4,d5,d6,d7 = net(im)
        pred = d1
        # pred = d1[:,0,:,:]
        pred = normPRED(pred)
    return pred

def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)
    out = (d-mi) / (ma-mi)
    return out

if __name__ == '__main__':
    image = torch.randn(1, 3, 320, 320).cuda()
    out = segment(image)
    print(out.shape)