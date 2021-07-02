import os
import torch
import transformers
from os.path import join
import h5py
from torch.utils.data import Dataset
from torch import nn
from efficientnet_pytorch import EfficientNet
import torchvision.models as models
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from torch.nn import functional as F
from PIL import Image

RESOURCE_PATH = './resources'
CHECKPOINT_FOLDER = join(RESOURCE_PATH, 'checkpoint')
CHECKPOINT_PATTERN = join(RESOURCE_PATH, 'checkpoint/{}_{}_{}_{}.pt')
PATIENCE = 5
THRESHOLD = 0.5


class PCLoader(Dataset):
    def __init__(self, img_path, label_path, preprocess):
        self.preprocess = preprocess
        self.img_path = img_path
        self.label_path = label_path
        self.data_size = self.load_data_size()

    def load_data_size(self):
        with h5py.File(self.label_path, 'r') as file_reader:
            label_list = list(file_reader['y'])
        data_size = len(label_list)

        return data_size

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        with h5py.File(self.img_path, 'r') as img_list, h5py.File(self.label_path, 'r') as label_list:
            _img = img_list['x'][idx]
            _img = Image.fromarray(_img)
            _img = self.preprocess(_img)
            _lbl = label_list['y'][idx].item()
        return _img, _lbl


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


class PatchCamelyonTrainer:
    def __init__(self, model_signature, train_path, train_lb_path, test_path, test_lb_path, dev_path, dev_lb_path,
                 drop_out, batch_size, learn_rate, n_epochs, n_class=1, n_max_cp=3):
        self.model_signature = model_signature
        self.n_max_cp = n_max_cp
        self.n_epochs = n_epochs
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        train_preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomVerticalFlip(),
            # transforms.RandomAffine(degrees=40, scale=(.9, 1.1), shear=0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        nontrain_preprocess = transforms.Compose([transforms.Resize(224), transforms.ToTensor()])

        self.trainer_iter = PCLoader(train_path, train_lb_path, train_preprocess)
        self.test_iter = PCLoader(test_path, test_lb_path, nontrain_preprocess)
        self.dev_iter = PCLoader(dev_path, dev_lb_path, nontrain_preprocess)

        self.train_loader = DataLoader(self.trainer_iter,
                                       batch_size=batch_size,
                                       shuffle=True)
        self.test_loader = DataLoader(self.test_iter,
                                      batch_size=batch_size,
                                      shuffle=False)
        self.dev_loader = DataLoader(self.dev_iter,
                                     batch_size=batch_size,
                                     shuffle=False)
        n_train_step = len(self.train_loader) * n_epochs
        self.model = PatchCamelyonModel(model_signature, drop_out=drop_out, n_class=n_class)
        self.model = self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learn_rate)
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.scheduler = transformers.get_linear_schedule_with_warmup(self.optimizer, len(self.train_loader),
                                                                      n_train_step)

    def perform_training(self):
        scaler = GradScaler()
        max_score = 0
        patient_count = 0
        for ep in range(self.n_epochs):
            self.model.train()
            total_loss = 0
            p_bar = tqdm(total=len(self.train_loader), ncols=800)
            for idx, (imgs, lbls) in enumerate(self.train_loader):
                lbls = torch.tensor(lbls, dtype=torch.float)
                imgs, lbls = imgs.to(self.device), lbls.to(self.device)
                self.optimizer.zero_grad()
                with autocast():
                    logits = self.model(imgs)
                    loss = self.criterion(logits, lbls)
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
                self.scheduler.step()

                total_loss += loss.item()
                p_bar.set_description('Epoch: %s - iteration %s - loss %.3f - avg_loss: %.3f - learn_rate %.8f' % (
                    ep + 1, idx + 1, loss.item(), total_loss / (idx + 1), self.optimizer.param_groups[0]['lr']))
                p_bar.update(1)

            print('--- Done training for epoch %s ' % ep)
            print('--- Perform evaluation ...')
            acc_score, roc_auc, nll = self.evaluate(self.dev_loader)
            if acc_score > max_score:
                print('Model performance improve from %.3f to %.3f. Saving model. ' % (max_score, acc_score))
                self.save_model(acc_score, roc_auc, nll)
                max_score = acc_score
            else:
                patient_count += 1
                if patient_count > PATIENCE:
                    print('Model performance is not improved for the last %s epochs. Exiting. ' % PATIENCE)
                    break

    def evaluate(self, data_loader):
        self.model.eval()
        predicts = []
        predict_logits = []
        ground_truths = []
        with torch.no_grad():
            for idx, (imgs, lbls) in enumerate(data_loader):
                imgs = imgs.to(self.device)
                ground_truths.extend(lbls.numpy().tolist())
                logits = self.model(imgs)
                logits = F.sigmoid(logits)
                predict_logits.extend(logits.cpu().numpy().tolist())
                predict_idx = (logits > THRESHOLD).cpu().numpy().astype(int)
                predicts.extend(predict_idx)

        acc_score = accuracy_score(ground_truths, predicts)
        fpr, tpr, thresholds = roc_curve(ground_truths, predicts, pos_label=1)
        roc_auc = auc(fpr, tpr)
        nll = F.nll_loss(torch.tensor(predict_logits), torch.tensor(ground_truths, dtype=torch.long))
        print(classification_report(ground_truths, predicts))

        return acc_score, roc_auc, nll

    def save_model(self, acc_score, roc_auc, nll):
        checkpoint = {'model_cp': self.model.state_dict(),
                      'optim_cp': self.optimizer.state_dict(),
                      'sched_cp': self.scheduler.state_dict()}
        cp_path = CHECKPOINT_PATTERN.format(self.model_signature, nll, roc_auc, acc_score)
        torch.save(checkpoint, cp_path)
        cp_files = [join(CHECKPOINT_FOLDER, w) for w in os.listdir(CHECKPOINT_FOLDER)]
        keep_cp = sorted(cp_files, reverse=True)[:self.n_max_cp]
        remove_cp = set(cp_files).difference(keep_cp)
        os.remove(list(remove_cp)[0])
        print('Model {} saved successfully.'.format(cp_path))

    def load_model(self, cp_path):
        checkpoint = torch.load(cp_path)
        self.model.load_state_dict(checkpoint['model_cp'])
        self.optimizer.load_state_dict(checkpoint['optim_cp'])
        self.scheduler.load_state_dict(checkpoint['sched_cp'])

        print('Model load sucessfully.')


if __name__ == '__main__':
    model_signature = ''
    train_path = ''
    train_lb_path = ''
    test_path = ''
    test_lb_path = ''
    dev_path = ''
    dev_lb_path = ''
    drop_out = ''
    batch_size = 32
    learn_rate = 0.0001
    n_epochs = 10
    n_class = 1
    n_max_cp = 3

    trainer = PatchCamelyonTrainer(model_signature, train_path, train_lb_path, test_path, test_lb_path, dev_path,
                                   dev_lb_path, drop_out, batch_size, learn_rate, n_epochs, n_class, n_max_cp)

    trainer.perform_training()
