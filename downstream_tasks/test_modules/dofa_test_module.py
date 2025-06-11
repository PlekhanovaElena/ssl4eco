from cvtorchvision import cvtransforms

from downstream_tasks.utils.root import get_root
from downstream_tasks.models.models_dwv import vit_base_patch16, vit_large_patch16
from downstream_tasks.transforms import *


class NormDOFATransform(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, img):
        img[:,:, 0:3] = (img[:,:, 0:3].astype(float) / 2500.0) * 255.0
        img[:,:, 3:] = (img[:,:, 3:].astype(float) / 8160.0) * 255.0
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img


class NormMeanStdTransform(torch.nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.MEAN = mean
        self.STD = std
    def forward(self, img):
        min_value = [self.MEAN[i] - 2 * self.STD[i] for i in range(len(self.MEAN))]
        max_value = [self.MEAN[i] + 2 * self.STD[i] for i in range(len(self.MEAN))]
        range_value = [max_value[i] - min_value[i] for i in range(len(max_value))] 
        img = (img - min_value) / range_value * 255.0
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img


class DOFAbaseTestModule(torch.nn.Module):
    def __init__(self):
        S2_MEAN = [114.1099739 , 114.81779093, 126.63977424,  84.33539309,
        97.84789168, 103.94461911, 101.435633  ,  72.32804172,
        56.66528851]
        S2_STD = [77.84352553, 69.96844919, 67.42465279, 64.57022983, 61.72545487,
            61.34187099, 60.29744676, 47.88519516, 42.55886798]

        super().__init__()
        checkpoint_path = get_root() + "/checkpoints/DOFA_ViT_base_e100.pth"
        print("=> loading checkpoint '{}'".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path, map_location="cuda")
        self.net = vit_base_patch16(num_classes = 0) #large or base
        #
        msg = self.net.load_state_dict(checkpoint, strict=False)
        for name, param in self.net.named_parameters():
            param.requires_grad = False
 
        bands = [3, 2, 1, 4, 5, 6, 7, 10, 11] # yes, 3,2,1
        print("Using DOFA bands")


        self.transform = cvtransforms.Compose([
            SelectBandsTransform(bands),
            NormDOFATransform(),
            NormMeanStdTransform(S2_MEAN, S2_STD),
            SetPatchSizeToPretraingSize(224),
            cvtransforms.ToTensor()])

        self.probe_input_dim = 768  # change this to 1024 when loading large

    def forward(self, image, location):
        batch_size = image.shape[0]
        return self.net(image, wave_list=[0.665, 0.56, 0.49, 0.705, 0.74, 0.783, 0.842, 1.61, 2.19]).view(batch_size, self.probe_input_dim)
        # 

class DOFAlargeTestModule(torch.nn.Module):
    def __init__(self):
        S2_MEAN = [114.1099739 , 114.81779093, 126.63977424,  84.33539309,
        97.84789168, 103.94461911, 101.435633  ,  72.32804172,
        56.66528851]
        S2_STD = [77.84352553, 69.96844919, 67.42465279, 64.57022983, 61.72545487,
            61.34187099, 60.29744676, 47.88519516, 42.55886798]

        super().__init__()
        checkpoint_path = get_root() + "/checkpoints/DOFA_ViT_large_e100.pth"
        print("=> loading checkpoint '{}'".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path, map_location="cuda")
        self.net = vit_large_patch16(num_classes = 0) #large or base
        #
        msg = self.net.load_state_dict(checkpoint, strict=False)
        for name, param in self.net.named_parameters():
            param.requires_grad = False
 
        bands = [3, 2, 1, 4, 5, 6, 7, 10, 11] # yes, 3,2,1
        print("Using DOFA bands")


        self.transform = cvtransforms.Compose([
            SelectBandsTransform(bands),
            NormDOFATransform(),
            NormMeanStdTransform(S2_MEAN, S2_STD),
            SetPatchSizeToPretraingSize(224),
            cvtransforms.ToTensor()])

        self.probe_input_dim = 1024 

    def forward(self, image, location):
        batch_size = image.shape[0]
        return self.net(image, wave_list=[0.665, 0.56, 0.49, 0.705, 0.74, 0.783, 0.842, 1.61, 2.19]).view(batch_size, self.probe_input_dim)
   