from utils import TotalMeter, count_params
import torch
import numpy as np
from pathlib import Path
import torch.nn.functional as F
from utils import continus_mixup_data
import wandb
from omegaconf import DictConfig
from typing import List
import torch.utils.data as utils
from components import LRScheduler
import logging
from .metrics import calculate_metrics


class Train:

    def __init__(self, cfg: DictConfig,
                 model: torch.nn.Module,
                 optimizers: List[torch.optim.Optimizer],
                 lr_schedulers: List[LRScheduler],
                 dataloaders: List[utils.DataLoader],
                 logger: logging.Logger) -> None:

        self.config = cfg
        self.logger = logger
        self.model = model
        self.logger.info(f'#model params: {count_params(self.model)}')
        self.train_dataloader, self.val_dataloader, self.test_dataloader = dataloaders
        self.epochs = cfg.training.epochs
        self.total_steps = cfg.total_steps
        self.optimizers = optimizers
        self.lr_schedulers = lr_schedulers
        self.save_path = Path(cfg.log_path) / cfg.unique_id
        self.save_learnable_graph = cfg.save_learnable_graph

        self.loss = torch.nn.CrossEntropyLoss()

        self.init_meters()

    def init_meters(self):
        self.train_accuracy1 = None
        self.val_accuracy1 = None
        self.test_accuracy1 = None

        self.train_loss, self.val_loss,\
            self.test_loss, self.train_accuracy,\
            self.val_accuracy, self.test_accuracy = [
                TotalMeter() for _ in range(6)]
        
    def reset_meters(self):
        self.train_accuracy1 = 0.
        self.val_accuracy1 = 0.
        self.test_accuracy1 = 0.

        for meter in [self.train_accuracy, self.val_accuracy,
                      self.test_accuracy, self.train_loss,
                      self.val_loss, self.test_loss]:
            meter.reset()
       

    def train_per_epoch(self, optimizer, lr_scheduler):
        self.model.train()
        pred_list = []
        pred_list1 = []
        label_list = []
        for fMRI_Feature, fMRI_Con, label, sMRI_Feature in self.train_dataloader:
            self.current_step += 1
            lr_scheduler.update(optimizer=optimizer, step=self.current_step)
            fMRI_Feature, fMRI_Con, label, sMRI_Feature = fMRI_Feature.cuda(), fMRI_Con.cuda(), label.cuda(), sMRI_Feature.cuda()
            fMRI_Con, label = continus_mixup_data(fMRI_Con, y=label)
            predict, predict1= self.model(fMRI_Feature, fMRI_Con, sMRI_Feature)
            loss = self.model.loss_function(predict, label, predict1)

            self.train_loss.update_with_weight_loss(loss.item(), label.shape[0])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pred_list.extend(predict1.tolist())
            pred_list1.extend(predict.tolist())
            label_list.extend(label.tolist())

        self.train_accuracy.update_with_weight_acc(pred_list, label_list)
        self.train_accuracy1, _, _, _, _ = calculate_metrics(pred_list, label_list)
        self.train_mse, _, _, _, _ = calculate_metrics(pred_list1, label_list)

    @torch.no_grad()  # upstream built autograd graphs during evaluation; outputs are unchanged
    def test_per_epoch(self, dataloader, loss_meter, str_select):
        labels = []
        result = []
        label_list = []
        pred_list = []
        pred_list1 = []
        self.model.eval()
        for fMRI_Feature, fMRI_Con, label, sMRI_Feature in dataloader:
            fMRI_Feature, fMRI_Con, label, sMRI_Feature = fMRI_Feature.cuda(), fMRI_Con.cuda(), label.cuda(), sMRI_Feature.cuda()
            output, output1 = self.model(fMRI_Feature, fMRI_Con, sMRI_Feature)
            label = label.float()
            loss = self.model.loss_function(output, label, output1)
            loss_meter.update_with_weight_loss(
                loss.item(), label.shape[0])
            result += F.softmax(output, dim=1)[:, 1].tolist()
            labels += label[:, 1].tolist()

            pred_list.extend(output1.tolist())
            pred_list1.extend(output.tolist())

            label_list.extend(label.tolist())


        if str_select == "val":
            self.val_accuracy.update_with_weight_acc(pred_list, label_list)
            self.val_accuracy1, _, _, _, _ = calculate_metrics(pred_list, label_list)
            self.val_mse, _, _, _, _ = calculate_metrics(pred_list1, label_list)
        else:
            self.test_accuracy.update_with_weight_acc(pred_list, label_list)
            self.test_accuracy1, _, _, _, _ = calculate_metrics(pred_list, label_list)
            self.test_mse, _, _, _, _ = calculate_metrics(pred_list1, label_list)

        accuracy, specificity, sensitivity, f1, auc = calculate_metrics(pred_list, label_list)
        accuracy1, specificity1, sensitivity1, f11, auc1 = calculate_metrics(pred_list1, label_list)
        return [accuracy, specificity, sensitivity, f1, auc], [accuracy1, specificity1, sensitivity1, f11, auc1], pred_list, pred_list1, label_list

    def save_result(self, results, epoch):
        self.save_path.mkdir(exist_ok=True, parents=True)
        np.save(self.save_path/"test_max_metrics.npy",
                results, allow_pickle=True)
        torch.save(self.model.state_dict(), self.save_path/"model.npy")

    def train(self):
        training_process = []
        self.current_step = 0
        
        max_test_accuracy = float('-inf')
        best_epoch_metrics = {}

        for epoch in range(self.epochs):
            self.reset_meters()
            self.train_per_epoch(self.optimizers[0], self.lr_schedulers[0])
            val_result, val_result1, pred_list_val, pred_list_val1, label_list_val = self.test_per_epoch(self.val_dataloader,
                                            self.val_loss, "val")

            test_result, test_result1, pred_list_test, pred_list_test1, label_list_test = self.test_per_epoch(self.test_dataloader,
                                            self.test_loss, "test")
            

            self.logger.info(" | ".join([
                f'Epoch[{epoch}/{self.epochs}]',
                f'Train Loss:{self.train_loss.avg: .5f}',
                f'Train Accuracy:{self.train_accuracy.final: .5f}',
                f'Train Acc:{self.train_mse: .5f}',
                f'Val Loss:{self.val_loss.avg: .5f}',
                f'Val Accuracy:{self.val_accuracy.final: .5f}',
                f'Val Acc:{self.val_mse: .5f}',
                f'Test Loss:{self.test_loss.avg: .5f}',
                f'Test Accuracy:{self.test_accuracy.final: .5f}',
                f'Test Acc:{self.test_mse: .5f}',
            ]))

            if self.test_accuracy.final > max_test_accuracy:
                max_test_accuracy = self.test_accuracy.final
                best_epoch_metrics = {
                    "Epoch": epoch,

                    'Train Loss': self.train_loss.avg,
                    'Train Accuracy': self.train_accuracy.final,

                    'Val Loss': self.val_loss.avg,
                    'Val Accuracy': self.val_accuracy.final,
                    "Val SPE": val_result[1],
                    "Val SEN": val_result[2],
                    "Val F1": val_result[3],
                    "Val AUC": val_result[4],

                    'Test Loss': self.test_loss.avg,
                    'Test Accuracy': self.test_accuracy.final,
                    "Test SPE": test_result[1],
                    "Test SEN": test_result[2],
                    "Test F1": test_result[3],
                    "Test AUC": test_result[4],
                }

            self.save_result(best_epoch_metrics, epoch)
            wandb.run.summary['Max Test Accuracy'] = max_test_accuracy
            wandb.run.summary.update(best_epoch_metrics)

            training_process.append({
                "Epoch": epoch,

                "Train Loss": self.train_loss.avg,
                "Train Accuracy": self.train_accuracy.final,
                "Test Loss": self.test_loss.avg,
                "Test Accuracy": self.test_accuracy.final,
                "Test SPE": test_result[1],
                "Test SEN": test_result[2],
                "Test F1": test_result[3],
                "Test AUC": test_result[4],
            })

        return test_result