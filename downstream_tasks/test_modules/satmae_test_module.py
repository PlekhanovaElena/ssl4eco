from cvtorchvision import cvtransforms

from rshf.satmae import SatMAE_Fine_MS

from downstream_tasks.transforms import *


MEANS = np.array([1370.19151926, 1184.3824625 , 1120.77120066, 1136.26026392,1263.73947144, 1645.40315151, 1846.87040806, 1762.59530783, 1972.62420416,  582.72633433, 1732.16362238, 1247.91870117])
STDS = np.array([633.15169573,  650.2842772 ,  712.12507725,  965.23119807, 948.9819932 , 1108.06650639, 1258.36394548, 1233.1492281, 1364.38688993, 472.37967789, 1310.36996126, 1087.6020813])


class SatMAETestModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        #load model
        model = SatMAE_Fine_MS.from_pretrained("MVRL/satmae-vitlarge-multispec-finetune")

        self.net = model 

        for name, param in self.net.named_parameters():
            param.requires_grad = False
        print("=> loaded pre-trained model")

        
        bands = [1,2,3,4,5,6,7,8,10,11]
        print("Using bands 1,2,3,4,5,6,7,8,10,11")

        self.transform = cvtransforms.Compose([
            SetPatchSizeToPretraingSize(96),
            SelectBandsTransform(bands),
            SentinelNormalize(MEANS[bands], STDS[bands]),
            cvtransforms.ToTensor(),])

        self.probe_input_dim = 1024

    def forward(self, image, location):
        
        output = self.net.forward_features(image)      

        return output

