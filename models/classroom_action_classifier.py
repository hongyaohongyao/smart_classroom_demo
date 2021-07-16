import torch


class ClassroomActionClassifier:
    def __init__(self, weights, device):
        self.device = device
        self.model = torch.jit.load(weights).to(device)
        _ = self.model(torch.zeros(1, 28).to(self.device))

    @staticmethod
    def preprocess(keypoints):
        x_min = torch.min(keypoints[:, :, 0], dim=1).values
        y_min = torch.min(keypoints[:, :, 1], dim=1).values
        x_max = torch.max(keypoints[:, :, 0], dim=1).values
        y_max = torch.max(keypoints[:, :, 1], dim=1).values
        x1_y1 = torch.stack([x_min, y_min], dim=1).unsqueeze(1)
        width = torch.stack([x_max - x_min, y_max - y_min], dim=1).unsqueeze(1)
        scaled_keypoints = (keypoints - x1_y1) / width
        scaled_keypoints = (scaled_keypoints - 0.5) / 0.5
        return scaled_keypoints.flatten(start_dim=1)

    def classify(self, keypoints):
        return self.model(self.preprocess(keypoints)).detach().cpu()
