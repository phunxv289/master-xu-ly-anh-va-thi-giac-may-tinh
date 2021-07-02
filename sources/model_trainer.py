import os

import torch
import transformers
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from torch.nn import functional as F
from sources.data_utils import PatchCamelyonIter
from sources.model_utils import PatchCamelyonModel
from sources.config import *

PATIENCE = 5
THRESHOLD = 0.5


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
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomAffine(degrees=40, scale=(.9, 1.1), shear=0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        nontrain_preprocess = transforms.Compose([transforms.Resize(224), transforms.ToTensor()])

        self.trainer_iter = PatchCamelyonIter(train_path, train_lb_path, train_preprocess)
        self.test_iter = PatchCamelyonIter(test_path, test_lb_path, nontrain_preprocess)
        self.dev_iter = PatchCamelyonIter(dev_path, dev_lb_path, nontrain_preprocess)

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
            p_bar = tqdm(len(self.train_loader), ncols=500)
            for idx, (imgs, lbls) in enumerate(self.train_loader):
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
                p_bar.set_description('Epoch: %s - iteration %s - loss %.3f - avg_loss: %.3f - learn_rate %.5f' % (
                    ep + 1, idx + 1, loss.item(), total_loss / (idx + 1), self.scheduler.get_last_lr()[0]))

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
