import torch
from torchvision.models import swin_t, Swin_T_Weights
from torchvision.models import resnet50, ResNet50_Weights
# https://pytorch.org/vision/master/models/swin_transformer.html
# Swin-T和Swin-S的复杂度分别与ResNet-50(DeiT-S)和ResNet-101相似
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor


class MyModel_swin_OCT(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()  
        m1 = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
        return_nodes1 = {
            'flatten': '[batch, 768]',
        }
        self.body_1 = create_feature_extractor(m1, return_nodes1)
        
        self.fc = torch.nn.Linear(768, num_classes)

    def forward(self, images):
        x_features = self.body_1(images)
        output = self.fc(x_features['[batch, 768]'])
        return output
    

class MyModel_swin_fundus(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()  
        m2 = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
        return_nodes2 = {
            'flatten': '[batch, 768]',
        }
        self.body_2 = create_feature_extractor(m2, return_nodes2)
        
        self.fc = torch.nn.Linear(768, num_classes)

    def forward(self, images):
        x_features = self.body_2(images)
        output = self.fc(x_features['[batch, 768]'])
        return output
    
class MyModel_res_OCT(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()  
        # Get a resnet50 backbone
        # m = resnet50(pretrained=True)
        m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        return_nodes = {
            'flatten': '[batch, 2048]',
        }
        self.body_1 = create_feature_extractor(m, return_nodes)        
        self.fc = torch.nn.Linear(2048, num_classes)

    def forward(self, images):
        x_features1 = self.body_1(images)
        output = self.fc(x_features1['[batch, 2048]'])
        return output

class MyModel_res_fundus(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()  
        # Get a resnet50 backbone
        # m = resnet50(pretrained=True)
        m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        return_nodes = {
            'flatten': '[batch, 2048]',
        }
        self.body_2 = create_feature_extractor(m, return_nodes)        
        self.fc = torch.nn.Linear(2048, num_classes)

    def forward(self, images):
        x_features1 = self.body_2(images)
        output = self.fc(x_features1['[batch, 2048]'])
        return output