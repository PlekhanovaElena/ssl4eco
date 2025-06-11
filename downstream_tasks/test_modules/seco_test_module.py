import os
import copy
import warnings
from cvtorchvision import cvtransforms

from downstream_tasks.utils.root import get_root
from downstream_tasks.models.moco2_module_multiband import MocoV2
from downstream_tasks.transforms import *

MEANS = np.array([340.76769064,429.9430203,614.21682446,590.23569706,950.68368468,1792.46290469,2075.46795189,2218.94553375,2266.46036911,2246.0605464,1594.42694882,1009.32729131])
STDS = np.array([554.81258967, 572.41639287, 582.87945694, 675.88746967, 729.89827633, 1096.01480586, 1273.45393088, 1365.45589904, 1356.13789355, 1302.3292881, 1079.19066363, 818.86747235])

#MEANS = np.array([1370.19151926, 1184.3824625 , 1120.77120066, 1136.26026392,1263.73947144, 1645.40315151, 1846.87040806, 1762.59530783, 1972.62420416,  582.72633433, 1732.16362238, 1247.91870117])
#STDS = np.array([633.15169573,  650.2842772 ,  712.12507725,  965.23119807, 948.9819932 , 1108.06650639, 1258.36394548, 1233.1492281, 1364.38688993, 472.37967789 , 1310.36996126, 1087.6020813])


class SeCoTestModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        checkpoint_path = get_root() + "/checkpoints/seco_resnet50_1m.ckpt"
        print("=> loading checkpoint '{}'".format(checkpoint_path))
        moco_k = 16384
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Hacky fix to prevent checkpoint loading crash
            if os.environ.get("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD") is None:
                os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = '1'
            checkpoint = MocoV2.load_from_checkpoint(
                checkpoint_path,
                arch="resnet50",
                moco_dim=128,
                moco_k=moco_k,
                bands="RGB")
        self.net = copy.deepcopy(checkpoint.encoder_q)       
        for name, param in self.net.named_parameters():
            param.requires_grad = False

        print("=> loaded pre-trained model")

        bands = [3,2,1]
        print("Using 3 bands")

        self.transform = cvtransforms.Compose([
            SelectBandsTransform(bands),
            SetPatchSizeToPretraingSize(128),
            SentinelNormalize(MEANS[bands], STDS[bands]),
            cvtransforms.ToTensor()])

        self.probe_input_dim = 2048

    def forward(self, image, location):
        return self.net(image)
