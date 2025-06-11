import copy
import warnings
from cvtorchvision import cvtransforms

from downstream_tasks.utils.root import get_root
from downstream_tasks.models.moco2_module_multiband import MocoV2
from downstream_tasks.transforms import *


class AblB9TestModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        checkpoint_path = get_root() + "/checkpoints/ablation_B9.ckpt"
        print("=> loading checkpoint '{}'".format(checkpoint_path))
        moco_k = 65536
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            checkpoint = MocoV2.load_from_checkpoint(
                checkpoint_path,
                arch="resnet50",
                emb_dim=128,
                moco_k=moco_k,
                bands="B9")
        self.net = copy.deepcopy(checkpoint.encoder_q)       
        for name, param in self.net.named_parameters():
            param.requires_grad = False
        print("=> loaded pre-trained model")

        bands = [1, 2, 3, 4, 5, 6, 7, 8, 12]
        print("Using 9 bands")

        self.transform = cvtransforms.Compose([
            SelectBandsTransform(bands),
            SetPatchSizeToPretraingSize(224),
            cvtransforms.ToTensor()])

        self.probe_input_dim = 2048

    def forward(self, image, location):
        return self.net(image)

class AblCalendarTestModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        checkpoint_path = get_root() + "/checkpoints/seco_calendar_099.ckpt"
        print("=> loading checkpoint '{}'".format(checkpoint_path))
        moco_k = 65536
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            checkpoint = MocoV2.load_from_checkpoint(
                checkpoint_path,
                arch="resnet50",
                emb_dim=128,
                moco_k=moco_k,
                bands="B9")
        self.net = copy.deepcopy(checkpoint.encoder_q)       
        for name, param in self.net.named_parameters():
            param.requires_grad = False
        print("=> loaded pre-trained model")

        bands = [1, 2, 3, 4, 5, 6, 7, 8, 12]
        print("Using 9 bands")

        self.transform = cvtransforms.Compose([
            SelectBandsTransform(bands),
            SetPatchSizeToPretraingSize(224),
            cvtransforms.ToTensor()])

        self.probe_input_dim = 2048

    def forward(self, image, location):
        return self.net(image)


class AblB9weightsTestModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        checkpoint_path = get_root() + "/checkpoints/ablation_B9_weights.ckpt"
        print("=> loading checkpoint '{}'".format(checkpoint_path))
        moco_k = 65536
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            checkpoint = MocoV2.load_from_checkpoint(checkpoint_path,
                arch="resnet50", emb_dim=128, moco_k=moco_k, bands="B9")
        self.net = copy.deepcopy(checkpoint.encoder_q)       
        for name, param in self.net.named_parameters():
            param.requires_grad = False
        print("=> loaded pre-trained model")

        bands = [1, 2, 3, 4, 5, 6, 7, 8, 12]
        print("Using 9 bands")

        self.transform = cvtransforms.Compose([
            SelectBandsTransform(bands),
            SetPatchSizeToPretraingSize(224),
            cvtransforms.ToTensor()])

        self.probe_input_dim = 2048

    def forward(self, image, location):
        return self.net(image)

class AblB12TestModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        checkpoint_path = get_root() + "/checkpoints/ablation_B12.ckpt"
        print("=> loading checkpoint '{}'".format(checkpoint_path))
        moco_k = 65536
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            checkpoint = MocoV2.load_from_checkpoint(checkpoint_path,
                arch="resnet50", emb_dim=128, moco_k=moco_k, bands="B12")
        self.net = copy.deepcopy(checkpoint.encoder_q)       
        for name, param in self.net.named_parameters():
            param.requires_grad = False
        print("=> loaded pre-trained model")

        bands = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        print("Using 12 bands")

        self.transform = cvtransforms.Compose([
            SelectBandsTransform(bands),
            SetPatchSizeToPretraingSize(224),
            cvtransforms.ToTensor()])

        self.probe_input_dim = 2048

    def forward(self, image, location):
        return self.net(image)

class AblB12weightsTestModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        checkpoint_path = get_root() + "/checkpoints/ablation_B12_weights.ckpt"
        print("=> loading checkpoint '{}'".format(checkpoint_path))
        moco_k = 65536
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            checkpoint = MocoV2.load_from_checkpoint(checkpoint_path,
                arch="resnet50", emb_dim=128, moco_k=moco_k, bands="B12")
        self.net = copy.deepcopy(checkpoint.encoder_q)       
        for name, param in self.net.named_parameters():
            param.requires_grad = False
        print("=> loaded pre-trained model")

        bands = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        print("Using 12 bands")

        self.transform = cvtransforms.Compose([
            SelectBandsTransform(bands),
            SetPatchSizeToPretraingSize(224),
            cvtransforms.ToTensor()])

        self.probe_input_dim = 2048

    def forward(self, image, location):
        return self.net(image)

class AblB12NTestModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        checkpoint_path = get_root() + "/checkpoints/ablation_B12N.ckpt"
        print("=> loading checkpoint '{}'".format(checkpoint_path))
        moco_k = 65536 
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            checkpoint = MocoV2.load_from_checkpoint(checkpoint_path,
                arch="resnet50", emb_dim=128, moco_k=moco_k, bands="B12N")
        self.net = copy.deepcopy(checkpoint.encoder_q)       
        for name, param in self.net.named_parameters():
            param.requires_grad = False
        print("=> loaded pre-trained model")

        bands = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        print("Using 12N bands")

        self.transform = cvtransforms.Compose([
            SelectBandsTransform(bands),
            SetPatchSizeToPretraingSize(224),
            cvtransforms.ToTensor()])

        self.probe_input_dim = 2048

    def forward(self, image, location):
        return self.net(image)

class AblB12NweightsTestModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        checkpoint_path = get_root() + "/checkpoints/ablation_B12N_weights.ckpt"
        print("=> loading checkpoint '{}'".format(checkpoint_path))
        moco_k = 65536 
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            checkpoint = MocoV2.load_from_checkpoint(checkpoint_path,
                arch="resnet50", emb_dim=128, moco_k=moco_k, bands="B12N")
        self.net = copy.deepcopy(checkpoint.encoder_q)       
        for name, param in self.net.named_parameters():
            param.requires_grad = False
        print("=> loaded pre-trained model")

        bands = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        print("Using 12N bands")

        self.transform = cvtransforms.Compose([
            SelectBandsTransform(bands),
            SetPatchSizeToPretraingSize(224),
            cvtransforms.ToTensor()])

        self.probe_input_dim = 2048

    def forward(self, image, location):
        return self.net(image)

