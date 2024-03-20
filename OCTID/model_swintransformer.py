import torch
from torchvision.models import swin_t, Swin_T_Weights
# https://pytorch.org/vision/master/models/swin_transformer.html
# Swin-T和Swin-S的复杂度分别与ResNet-50(DeiT-S)和ResNet-101相似
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor
from cross_model_attention import Transformer
from torchvision import transforms


class MyModel_swin_cat(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()  
        # Get a Swin_T backbone
        m1 = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
        return_nodes1 = {
            'flatten': '[batch, 768]',
        }
        self.body_1 = create_feature_extractor(m1, return_nodes1)
        m2 = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
        return_nodes2 = {
            'flatten': '[batch, 768]',
        }
        self.body_2 = create_feature_extractor(m2, return_nodes2)
        
        # train_nodes, eval_nodes = get_graph_node_names(swin_t())
        # print(train_nodes)
        
        self.fc_1536 = torch.nn.Linear(768*2, num_classes)

    def forward(self, images1, images2):
        x_features1 = self.body_1(images1)
        x_features2 = self.body_2(images2)
        # print(x_features1)
        # print(x_features1['[batch, 768]'].shape)
        # for k,v in x_features1.items():
        #     print(k)
        #     print(v.shape)
        x_feature = torch.cat((x_features1['[batch, 768]'],x_features2['[batch, 768]']),axis=1)
        output = self.fc_1536(x_feature)
        # x = self.body(x)
        # x = self.fpn(x)
        return output


class MyModel_swin_trans(torch.nn.Module):
    def __init__(self,  num_classes):
        super().__init__()
        m1 = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
        return_nodes1 = {
            'flatten': '[batch, 768]',
        }
        self.body_1 = create_feature_extractor(m1, return_nodes1)
        m2 = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
        return_nodes2 = {
            'flatten': '[batch, 768]',
        }
        self.body_2 = create_feature_extractor(m2, return_nodes2)
        self.Transformer_1 = Transformer(768)
        self.Transformer_2 = Transformer(768)
        self.fc_3072 = torch.nn.Linear(768*4, num_classes)

    def forward(self, images1, images2):
        x_features1 = self.body_1(images1)
        x_features2 = self.body_2(images2)
        x12 = self.Transformer_1(x_features1['[batch, 768]'],x_features2['[batch, 768]'])
        x21 = self.Transformer_2(x_features2['[batch, 768]'],x_features1['[batch, 768]'])
        x_feature = torch.cat((x12,x21,x_features1['[batch, 768]'],x_features2['[batch, 768]']),axis=1)
        output = self.fc_3072(x_feature)
        return output
    
class MyModel_swin_6channel(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()  
        m = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
        model_dict = m.state_dict() # 获取整个网络的预训练权重
        # 取出从conv1权重并进行平均和拓展
        conv_weight = torch.mean(m.features[0][0].weight,dim=1,keepdim=True).repeat(1,6,1,1) # input_ch = 6
        # 原先的features.0.0: nn.Conv2d(3, embed_dim, kernel_size=(patch_size[0], patch_size[1]), stride=(patch_size[0], patch_size[1])) 
        # 其中 patch_size=[4, 4], embed_dim=96
        conv = torch.nn.Conv2d(6, 96, kernel_size=(4,4), stride=(4,4)) # 新的features.0.0层
        m.features[0][0] = conv # 修改第一层
        model_dict['features.0.0.weight'] = conv_weight # 将conv1权重替换为新conv1权重
        model_dict.update(model_dict) # 更新整个网络的预训练权重
        m.load_state_dict(model_dict) # 载入新预训练权重

        return_nodes = {
            # 'x': 'x',
            # 'features.0.0': 'features.0.0'
            'flatten': '[batch, 768]',
        }
        
        self.body = create_feature_extractor(m, return_nodes)
        self.fc = torch.nn.Linear(768, num_classes)

    def forward(self, images1, images2):
        x_6channnel = torch.cat((images1, images2), axis=1) # torch.Size([10, 6, 224, 224])
        x_features = self.body(x_6channnel)
        output = self.fc(x_features['[batch, 768]'])
        return output