import numpy as np
import torch


class PFLDFaceAligner:

    def __init__(self, weights, device):
        self.device = device
        self.model = torch.jit.load(weights).to(device)
        _ = self.model(torch.zeros(1, 3, 112, 112).to(self.device))

    @staticmethod
    def preprocess(faces):
        return torch.from_numpy(faces.transpose((0, 3, 1, 2)) / 255.0)

    def align(self, faces):
        result = self.model(self.preprocess(faces).to(device=self.device, dtype=torch.float32)).detach().cpu()
        return result.view(-1, 68, 2)


class MobileNetSEFaceAligner:
    mean = np.asarray([0.485, 0.456, 0.406]).reshape(1, 1, 1, 3)
    std = np.asarray([0.229, 0.224, 0.225]).reshape(1, 1, 1, 3)

    def __init__(self, weights, device):
        self.device = device
        self.model = torch.jit.load(weights).to(device)
        _ = self.model(torch.zeros(1, 3, 56, 56).to(self.device))

    def preprocess(self, faces):
        faces = (faces / 255 - self.mean) / self.std
        return torch.from_numpy(faces.transpose((0, 3, 1, 2)))

    def align(self, faces):
        result = self.model(self.preprocess(faces).to(device=self.device, dtype=torch.float32)).detach().cpu()
        return result.view(-1, 68, 2)
