import warnings
from cvtorchvision import cvtransforms

import satlaspretrain_models
weights_manager = satlaspretrain_models.Weights()

from downstream_tasks.transforms import *


class SatlasTestModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        #load model
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = weights_manager.get_pretrained_model(model_identifier="Sentinel2_SwinB_SI_MS",fpn=False)

        self.net = model 

        for name, param in self.net.named_parameters():
            param.requires_grad = False
        print("=> loaded pre-trained model")

        bands = [3, 2, 1, 4, 5, 6, 7, 10,11]
        print("Using RGB, B05, B06, B07, B08, B11, B12")

        self.transform = cvtransforms.Compose([
            SetPatchSizeToPretraingSize(512),
            SelectBandsTransform(bands),
            SatlasNormRGB(),
            SatlasNormOther(),
            cvtransforms.ToTensor()])

        self.probe_input_dim = 1024

    def forward(self, image, location):
        #NOTE tbd what to use here exactly
        output = self.net(image)[3]
        output = torch.amax(output, dim=(2,3))
        return output
