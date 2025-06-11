import warnings
from cvtorchvision import cvtransforms

from downstream_tasks.utils.root import get_root
from downstream_tasks.transforms import *
from downstream_tasks.foundation_models.use_croma import PretrainedCROMA

MEANS = np.array([1370.19151926, 1184.3824625 , 1120.77120066, 1136.26026392,1263.73947144, 1645.40315151, 1846.87040806, 1762.59530783, 1972.62420416,  582.72633433, 1732.16362238, 1247.91870117])
STDS = np.array([633.15169573,  650.2842772 ,  712.12507725,  965.23119807, 948.9819932 , 1108.06650639, 1258.36394548, 1233.1492281, 1364.38688993, 472.37967789, 1310.36996126, 1087.6020813])


class CromaTestModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        checkpoint_path = get_root() + "/checkpoints/CROMA_large.pt"
        print("=> loading checkpoint '{}'".format(checkpoint_path))

        #load model
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            #NOTE loading model directly from repo results in better performance, preprocessing steps clearer, ...
            #model = CROMA.from_pretrained("MVRL/croma-large", config={"modality": "optical", "size":"large", "image_resolution": 120})
            model = PretrainedCROMA(pretrained_path=checkpoint_path, size='large', modality='optical', image_resolution=120)
        self.net = model 

        for name, param in self.net.named_parameters():
            param.requires_grad = False
        print("=> loaded pre-trained model")

        bands = [0,1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        print("Using 12 bands")

        self.transform = cvtransforms.Compose([
            SetPatchSizeToPretraingSize(120),
            SelectBandsTransform(bands),
            SentinelNormalize(MEANS[bands], STDS[bands]),
            cvtransforms.ToTensor()])

        self.probe_input_dim = 1024

    def forward(self, image, location):
        output = self.net(optical_images = image)
        return output["optical_GAP"]
