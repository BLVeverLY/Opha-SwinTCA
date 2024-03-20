import torch
from torchvision.models import resnet50, ResNet50_Weights
# from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor
# from torchvision.models.detection.backbone_utils import LastLevelMaxPool
# from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork
from cross_model_attention import Transformer


class MyModel_res_cat(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()  
        # Get a resnet50 backbone
        # m = resnet50(pretrained=True)
        m1 = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        return_nodes1 = {
            'flatten': '[batch, 2048]',
        }
        # Extract 4 main layers (note: MaskRCNN needs this particular name
        # mapping for return nodes)
        # self.body = create_feature_extractor(
        #     m, return_nodes={f'layer{k}': str(v)
        #                      for v, k in enumerate([1, 2, 3, 4])})
            # {'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'}
        self.body_1 = create_feature_extractor(m1, return_nodes1)
        m2 = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        return_nodes2 = {
            'flatten': '[batch, 2048]',
        }
        self.body_2 = create_feature_extractor(m2, return_nodes2)
        
        # train_nodes, eval_nodes = get_graph_node_names(resnet50())
        # print(train_nodes)
        
        self.fc_4096 = torch.nn.Linear(2048*2, num_classes)

        # # Dry run to get number of channels for FPN
        # inp = torch.randn(2, 3, 224, 224)
        # with torch.no_grad():
        #     out = self.body(inp)
        # in_channels_list = [o.shape[1] for o in out.values()]
        # # Build FPN
        # self.out_channels = 256
        # self.fpn = FeaturePyramidNetwork(
        #     in_channels_list, out_channels=self.out_channels,
        #     extra_blocks=LastLevelMaxPool())

    def forward(self, images1, images2):
        x_features1 = self.body_1(images1)
        x_features2 = self.body_2(images2)
        # print(x_features1)
        # print(x_features1['[batch, 2048]'].shape)
        # for k,v in x_features1.items():
        #     print(k)
        #     print(v.shape)
        # print(x_features1.shape)
        x_feature = torch.cat((x_features1['[batch, 2048]'],x_features2['[batch, 2048]']),axis=1)
        output = self.fc_4096(x_feature)
        # x = self.body(x)
        # x = self.fpn(x)
        return output
    

class MyModel_res_trans(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()  
        m1 = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        return_nodes1 = {
            'flatten': '[batch, 2048]',
        }
        self.body_1 = create_feature_extractor(m1, return_nodes1)
        m2 = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        return_nodes2 = {
            'flatten': '[batch, 2048]',
        }
        self.body_2 = create_feature_extractor(m2, return_nodes2)
        self.Transformer_1 = Transformer(2048)
        self.Transformer_2 = Transformer(2048)
        self.fc_8192 = torch.nn.Linear(2048*4, num_classes)

    def forward(self, images1, images2):
        x_features1 = self.body_1(images1)
        x_features2 = self.body_2(images2)
        x12 = self.Transformer_1(x_features1['[batch, 2048]'],x_features2['[batch, 2048]'])
        x21 = self.Transformer_2(x_features2['[batch, 2048]'],x_features1['[batch, 2048]'])
        x_feature = torch.cat((x12,x21,x_features1['[batch, 2048]'],x_features2['[batch, 2048]']),axis=1)
        output = self.fc_8192(x_feature)
        return output


class MyModel_res_6channel(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()  
        m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        model_dict = m.state_dict() # 获取整个网络的预训练权重
        # 取出从conv1权重并进行平均和拓展
        conv_weight = torch.mean(m.conv1.weight,dim=1,keepdim=True).repeat(1,6,1,1) # input_ch = 6
        # 原先的conv1: nn.Conv2d(3, 64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)
        conv = torch.nn.Conv2d(6, 64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False) # 新的conv1层
        m.conv1 = conv # 修改第一层
        model_dict['conv1.weight'] = conv_weight # 将conv1权重替换为新conv1权重
        model_dict.update(model_dict) # 更新整个网络的预训练权重
        m.load_state_dict(model_dict) # 载入新预训练权重

        return_nodes = {
            # 'x': 'x',
            # 'conv1': 'conv1',
            'flatten': '[batch, 2048]',
        }
        
        self.body = create_feature_extractor(m, return_nodes)
        self.fc = torch.nn.Linear(2048, num_classes)

    def forward(self, images1, images2):
        x_6channnel = torch.cat((images1, images2), axis=1) # torch.Size([10, 6, 224, 224])
        x_features = self.body(x_6channnel)
        output = self.fc(x_features['[batch, 2048]'])
        return output