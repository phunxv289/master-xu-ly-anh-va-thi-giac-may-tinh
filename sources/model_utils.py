from torch import nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
import torchvision.models as models


def remove_last_layer(model):
    """ remove last layer of the base model and return the last feature """
    last_layer = list(model.children())[-1]
    feat_size = last_layer.in_features
    module_list = nn.Sequential(*list(model.children())[:-1])
    return module_list, feat_size


class ResNetCustom(nn.Module):
    def __init__(self, base_model, drop_out, n_class):
        super(ResNetCustom, self).__init__()
        self.base_model = base_model
        self.base_model, feat_size = remove_last_layer(base_model)
        self.dropout = nn.Dropout(drop_out)
        self.output_layer = nn.Linear(feat_size, n_class)

    def forward(self, batch_img):
        model_logits = self.base_model(batch_img)
        model_logits = model_logits.squeeze()
        model_logits = self.dropout(model_logits)
        output_logits = self.output_layer(model_logits)

        return output_logits


class DensenetCustom(nn.Module):
    def __init__(self, base_model, drop_out, n_class):
        super(DensenetCustom, self).__init__()
        self.base_model = base_model
        self.base_model, feat_size = remove_last_layer(base_model)
        self.dropout = nn.Dropout(drop_out)
        self.output_layer = nn.Linear(feat_size, n_class)

    def forward(self, batch_img):
        model_logits = self.base_model(batch_img)
        model_logits = F.adaptive_avg_pool2d(model_logits, (1, 1))
        model_logits = model_logits.squeeze()
        model_logits = self.dropout(model_logits)
        output_logits = self.output_layer(model_logits)

        return output_logits


class PatchCamelyonModel(nn.Module):
    def __init__(self, model_signature, drop_out, n_class=1):
        """
            Build model by provided model signature.
            :param model_signature: string to identify which model to use. ['efficient-net', 'res-net', 'dense-net']
        """
        super(PatchCamelyonModel, self).__init__()
        self.model_signature = model_signature
        assert model_signature in ['efficient-net', 'res-net', 'dense-net']
        if model_signature == 'efficient-net':
            self.model = EfficientNet.from_pretrained('efficientnet-b5', num_classes=n_class)
        elif model_signature == 'dense-net':
            base_model = models.densenet161(pretrained=True)
            self.model = DensenetCustom(base_model, drop_out=drop_out, n_class=n_class)
        else:
            base_model = models.resnet50(pretrained=True)
            self.model = ResNetCustom(base_model, drop_out=drop_out, n_class=n_class)

    def forward(self, img_batch):
        """
        :param img_batch: batch of images. [N x C x W x H]
        :return: image logits with the shape of [N]
        """
        logits = self.model(img_batch)
        logits = logits.squeeze()
        return logits

    def get_total_params(self):
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print('Model: %s - total parameters: %s - trainable parameter: %s ' % (
            self.model_signature, total_params, trainable_params))
