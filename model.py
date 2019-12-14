import torch
import torch.nn as nn
import torchvision.models as models

class Vqna(nn.Module):

    def __init__(self, num_classes):
        super(Vqna, self).__init__()

        self.vgg = models.vgg19(pretrained=True)
        self.vgg.classifier = self.vgg.classifier[:5]
        for param in self.vgg.features:
            param.requires_grad = False
        for param in self.vgg.classifier:
            param.requires_grad = True
        
        self.fc1 = nn.Sequential(
            nn.Linear(4096, 768, bias=True),
            nn.ReLU(inplace=True)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(768, 256, bias=True),
            nn.ReLU(inplace=True)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(256, num_classes, bias=True),
            nn.Softmax()
        )
    
    def forward(self, images, embeddings):
        features = self.vgg(images)
        x = torch.cat((features, embeddings), 1)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x

