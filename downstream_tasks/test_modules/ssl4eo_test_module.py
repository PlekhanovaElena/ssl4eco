from cvtorchvision import cvtransforms
from torchvision import models

from downstream_tasks.utils.root import get_root
from downstream_tasks.transforms import *


class SSL4EOTestModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        checkpoint_path = get_root() + "/checkpoints/B13_rn50_moco_0099_ckpt.pth"
        print("=> loading checkpoint '{}'".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path, map_location="cuda")

        # rename moco pre-trained keys
        state_dict = checkpoint['state_dict']
        for k in list(state_dict.keys()):
            if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
                state_dict[k[len("module.encoder_q."):]] = state_dict[k]
            del state_dict[k]

        self.net = models.resnet50(weights=None)
        self.net.conv1 = torch.nn.Conv2d(13, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        for name, param in self.net.named_parameters():
            param.requires_grad = False
        #removing last layer
        msg = self.net.load_state_dict(state_dict, strict=False)
        print("=> loaded pre-trained model")
        self.net = torch.nn.Sequential(*(list(self.net.children())[:-1]))


        bands = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 11] # the 2 times 10 is for a reason
        print("Using 13 bands")

        self.transform = cvtransforms.Compose([
            SelectBandsTransform(bands),
            NormTransform(),
            SetBandToZero(10),
            SetPatchSizeToPretraingSize(224),
            cvtransforms.ToTensor()])

        self.probe_input_dim = 2048

    def forward(self, image, location):
        batch_size = image.shape[0]
        return self.net(image).view(batch_size, self.probe_input_dim)

