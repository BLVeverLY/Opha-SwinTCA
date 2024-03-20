import torch
from torchvision.models import swin_t, Swin_T_Weights
# https://pytorch.org/vision/master/models/swin_transformer.html
# Swin-T and Swin-S are similar in complexity to ResNet-50(DeiT-S) and ResNet-101, respectively
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
        x_feature = torch.cat((x_features1['[batch, 768]'],x_features2['[batch, 768]']),axis=1)
        output = self.fc_1536(x_feature)
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
        model_dict = m.state_dict() # gets the pre-training weights
        # ke the weights from conv1 and average and expand them
        conv_weight = torch.mean(m.features[0][0].weight,dim=1,keepdim=True).repeat(1,6,1,1) # input_ch = 6
        # origin features.0.0: nn.Conv2d(3, embed_dim, kernel_size=(patch_size[0], patch_size[1]), stride=(patch_size[0], patch_size[1])) 
        # 其中 patch_size=[4, 4], embed_dim=96
        conv = torch.nn.Conv2d(6, 96, kernel_size=(4,4), stride=(4,4)) # new features.0.0
        m.features[0][0] = conv # replace
        model_dict['features.0.0.weight'] = conv_weight # replace the conv1 weight with the new conv1 weight
        model_dict.update(model_dict) # update the pre-training weights for the entire network
        m.load_state_dict(model_dict) # load the new pre-training weight

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