import torch
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import swin_t, Swin_T_Weights
# from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor
# from torchvision.models.detection.backbone_utils import LastLevelMaxPool
# from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork


class MyModel_res_1(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()  
        # Get a resnet50 backbone
        # m = resnet50(pretrained=True)
        m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        return_nodes = {
            'flatten': '[batch, 2048]',
        }
        # Extract 4 main layers (note: MaskRCNN needs this particular name
        # mapping for return nodes)
        # self.body = create_feature_extractor(
        #     m, return_nodes={f'layer{k}': str(v)
        #                      for v, k in enumerate([1, 2, 3, 4])})
            # {'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'}
        self.body_1 = create_feature_extractor(m, return_nodes)        
        self.fc = torch.nn.Linear(2048, num_classes)

    def forward(self, images1, images2):
        x_features1 = self.body_1(images1)
        output = self.fc(x_features1['[batch, 2048]'])
        # x = self.body(x)
        # x = self.fpn(x)
        return output


class MyModel_swin_1(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()  
        # Get a Swin_T backbone
        m = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
        return_nodes = {
            'flatten': '[batch, 768]',
        }
        self.body_1 = create_feature_extractor(m, return_nodes)
        
        self.fc = torch.nn.Linear(768, num_classes)

    def forward(self, images1, images2):
        x_features1 = self.body_1(images1)
        output = self.fc(x_features1['[batch, 768]'])
        return output


class MyModel_res_2(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()  
        # Get a resnet50 backbone
        # m = resnet50(pretrained=True)
        m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        return_nodes = {
            'flatten': '[batch, 2048]',
        }
        # Extract 4 main layers (note: MaskRCNN needs this particular name
        # mapping for return nodes)
        # self.body = create_feature_extractor(
        #     m, return_nodes={f'layer{k}': str(v)
        #                      for v, k in enumerate([1, 2, 3, 4])})
            # {'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'}
        self.body_2 = create_feature_extractor(m, return_nodes)        
        self.fc = torch.nn.Linear(2048, num_classes)

    def forward(self, images1, images2):
        x_features2 = self.body_2(images2)
        output = self.fc(x_features2['[batch, 2048]'])
        # x = self.body(x)
        # x = self.fpn(x)
        return output


class MyModel_swin_2(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()  
        # Get a Swin_T backbone
        m = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
        return_nodes = {
            'flatten': '[batch, 768]',
        }
        self.body_2 = create_feature_extractor(m, return_nodes)
        
        self.fc = torch.nn.Linear(768, num_classes)

    def forward(self, images1, images2):
        x_features2 = self.body_2(images2)
        output = self.fc(x_features2['[batch, 768]'])
        return output