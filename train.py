import os

import torch

from datetime import datetime
from util import Logger, accuracy, TotalMeter
import numpy as np
from pathlib import Path
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from util.prepossess import mixup_criterion, mixup_data
from util.loss import mixup_cluster_loss, topk_loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"


class BasicTrain:

    def __init__(self, train_config, model, optimizers, dataloaders, log_folder) -> None:
        self.logger = Logger()
        self.model = model.to(device)
        self.train_dataloader, self.val_dataloader, self.test_dataloader = dataloaders
        self.epochs = train_config['epochs']
        self.optimizers = optimizers
        self.best_acc = 0
        self.best_model = None
        self.best_acc_val = 0
        self.best_auc_val = 0
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
        self.pool_ratio = train_config['pool_ratio']
        self.group_loss = train_config['group_loss']

        self.sparsity_loss = train_config['sparsity_loss']
        self.sparsity_loss_weight = train_config['sparsity_loss_weight']

        self.save_path = log_folder

        self.save_learnable_graph = True

        self.init_meters()

    def init_meters(self):
        self.train_loss, self.val_loss, self.test_loss, self.train_accuracy,\
            self.val_accuracy, self.test_accuracy, self.edges_num = [
                TotalMeter() for _ in range(7)]

        self.loss1, self.loss2, self.loss3 = [TotalMeter() for _ in range(3)]

    def reset_meters(self):
        for meter in [self.train_accuracy, self.val_accuracy, self.test_accuracy,
                      self.train_loss, self.val_loss, self.test_loss, self.edges_num,
                      self.loss1, self.loss2, self.loss3]:
            meter.reset()

    def train_per_epoch(self, optimizer):


        self.model.train()
        # scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
        for data_in, pearson, label, pseudo in self.train_dataloader:

            label = label.long()

            data_in, pearson, label, pseudo = data_in.to(
                device), pearson.to(device), label.to(device), pseudo.to(device)

            inputs, nodes, targets_a, targets_b, lam = mixup_data(
                data_in, pearson, label, 1, device)

            [output, score, cor_matrix], learnable_matrix, edge_variance = self.model(inputs, nodes, pseudo)

            loss = 2 * mixup_criterion(
                self.loss_fn, output, targets_a, targets_b, lam)

            # if self.group_loss:
            #     loss += mixup_cluster_loss(learnable_matrix,
            #                                targets_a, targets_b, lam)

            # if self.sparsity_loss:
            #     sparsity_loss = self.sparsity_loss_weight * \
            #         torch.norm(learnable_matrix, p=1)
            #     loss += sparsity_loss


            loss += 0.001*topk_loss(score, self.pool_ratio)


            self.train_loss.update_with_weight(loss.item(), label.shape[0])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # scheduler.step()
            top1 = accuracy(output, label)[0]
            self.train_accuracy.update_with_weight(top1, label.shape[0])
            self.edges_num.update_with_weight(edge_variance, label.shape[0])

    def test_per_epoch(self, dataloader, loss_meter, acc_meter):
        labels = []
        result = []

        self.model.eval()

        for data_in, pearson, label, pseudo in dataloader:
            label = label.long()
            data_in, pearson, label, pseudo = data_in.to(
                device), pearson.to(device), label.to(device), pseudo.to(device)
            [output, score, cor_matrix], _, _ = self.model(data_in, pearson, pseudo)
            loss = self.loss_fn(output, label)
            loss_meter.update_with_weight(
                loss.item(), label.shape[0])
            top1 = accuracy(output, label)[0]
            acc_meter.update_with_weight(top1, label.shape[0])
            result += F.softmax(output, dim=1)[:, 1].tolist()
            labels += label.tolist()

        auc = roc_auc_score(labels, result)

        result = np.array(result)
        result[result > 0.5] = 1
        result[result <= 0.5] = 0
        metric = precision_recall_fscore_support(
            labels, result, average='micro')
        con_matrix = confusion_matrix(labels, result)
        return [auc] + list(metric), con_matrix

    def generate_save_learnable_matrix(self):
        learable_matrixs = []

        labels = []

        for data_in, nodes, label, pseudo in self.test_dataloader:
            label = label.long()
            data_in, nodes, label, pseudo = data_in.to(
                device), nodes.to(device), label.to(device), pseudo.to(device)
            _, learable_matrix, _ = self.model(data_in, nodes, pseudo)

            learable_matrixs.append(learable_matrix.cpu().detach().numpy())
            labels += label.tolist()

        self.save_path.mkdir(exist_ok=True, parents=True)
        np.save(self.save_path/"learnable_matrix.npy", {'matrix': np.vstack(
            learable_matrixs), "label": np.array(labels)}, allow_pickle=True)

    def save_result(self, results, txt, train_loss, test_loss):

        self.save_path.mkdir(exist_ok=True, parents=True)
        np.save(self.save_path/"training_process.npy",
                results, allow_pickle=True)
        
        np.save(self.save_path/"train_loss.npy",
                train_loss, allow_pickle=True)
        np.save(self.save_path/"test_loss.npy",
                test_loss, allow_pickle=True)
        
        with open(self.save_path / "training_info.txt", 'a', encoding='utf-8') as f:
            f.write(txt)
        torch.save(self.best_model.state_dict(), self.save_path/f"model_{self.best_acc}%.pt")

    def train(self):
        training_process = []
        txt = ''
        train_loss = []
        test_loss = []

        best_val_accuracy = 0
        best_val_auc = 0
        best_val_f1 = 0
        best_val_precision = 0
        best_val_error_rate = 1
        best_val_sen = 0
        best_val_spe = 0

        for epoch in range(self.epochs):
            self.reset_meters()
            self.train_per_epoch(self.optimizers[0])

            val_result, con_matrix = self.test_per_epoch(self.val_dataloader,
                                                         self.val_loss, self.val_accuracy)

            test_result, con_matrix = self.test_per_epoch(self.test_dataloader,
                                                          self.test_loss, self.test_accuracy)

            if self.best_acc <= self.test_accuracy.avg:
                self.best_acc = self.test_accuracy.avg
                self.best_model = self.model

            SEN = con_matrix[0][0] / (con_matrix[0][0] + con_matrix[1][0]) if (con_matrix[0][0] + con_matrix[1][
                0]) != 0 else 0
            SPE = con_matrix[1][1] / (con_matrix[1][1] + con_matrix[0][1]) if (con_matrix[1][1] + con_matrix[0][
                1]) != 0 else 0

            if val_result[0] > best_val_auc:
                best_val_accuracy = val_result[0]
            if val_result[1] > best_val_f1:
                best_val_f1 = val_result[1]
            if val_result[2] > best_val_precision:
                best_val_precision = val_result[2]
            if val_result[3] < best_val_error_rate:
                best_val_error_rate = val_result[3]
            if SEN > best_val_sen:
                best_val_sen = SEN
            if SPE > best_val_spe:
                best_val_spe = SPE

            self.logger.info(" | ".join([
                f'Epoch[{epoch}/{self.epochs}]',
                f'Train Loss:{self.train_loss.avg:.3f}',
                f'Train Accuracy:{self.train_accuracy.avg:.3f}%',
                f'Test Loss:{self.test_loss.avg:.3f}',
                f'Val Accuracy:{self.val_accuracy.avg:.3f}%',
                f'Test Accuracy:{self.test_accuracy.avg:.3f}%',
                f'Val AUC:{val_result[0]:.2f}',
                f'Test AUC:{test_result[0]:.4f}',
                f'Test SEN:{SEN:.4f}',
                f'Test SPE:{SPE:.4f}'
            ]))

            txt += f'Epoch[{epoch}/{self.epochs}] ' + f'Train Loss:{self.train_loss.avg:.3f} ' + \
                   f'Train Loss:{self.test_loss.avg:.3f} ' + f'Train Accuracy:{self.train_accuracy.avg:.3f}% ' + \
                   f'Val Accuracy:{self.val_accuracy.avg:.3f}% ' + f'Test Accuracy:{self.test_accuracy.avg:.3f}% ' + \
                   f'Val AUC:{val_result[0]:.3f} ' + f'Test AUC:{test_result[0]:.4f}' + \
                   f'Test SEN:{SEN:.4f}' + f'Test SPE:{SPE:.4f}' + '\n'

            training_process.append([self.train_accuracy.avg, self.train_loss.avg,
                                     self.val_loss.avg, self.test_loss.avg, self.val_accuracy.avg,
                                     self.test_accuracy.avg]
                                    + val_result + test_result)
            train_loss.append(self.train_loss.avg)
            test_loss.append(self.test_loss.avg)

        now = datetime.now()
        date_time = now.strftime("%m-%d-%H-%M-%S")
        self.save_path = self.save_path / Path(f"{self.best_acc:.3f}%_{date_time}")

        import os

        # Ensure save path exists before saving best validation metrics
        save_file_path = self.save_path / 'best_validation_metrics.txt'
        os.makedirs(os.path.dirname(save_file_path), exist_ok=True)

        # Save best validation metrics
        with open(str(save_file_path), 'w') as file:
            file.write(f'Best Validation Accuracy: {best_val_accuracy}\n')
            file.write(f'Best Validation F1: {best_val_f1}\n')
            file.write(f'Best Validation Precision: {best_val_precision}\n')
            file.write(f'Best Validation Error Rate: {best_val_error_rate}\n')
            file.write(f'Best Validation SEN: {best_val_sen}\n')
            file.write(f'Best Validation SPE: {best_val_spe}\n')

        if self.save_learnable_graph:
            self.generate_save_learnable_matrix()
        self.save_result(training_process, txt, train_loss, test_loss)

        import matplotlib.pyplot as plt

        # Plot loss curves
        plt.figure()
        plt.plot(range(1, len(train_loss) + 1), train_loss, label='Train Loss')
        plt.plot(range(1, len(test_loss) + 1), test_loss, label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Test Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(str(self.save_path) + '/loss_curve.png')  # Save loss curve image
        plt.show()

        # Plot accuracy curves
        train_accuracy = [item[0] for item in training_process]
        val_accuracy = [item[4] for item in training_process]
        test_accuracy = [item[5] for item in training_process]

        plt.figure()
        plt.plot(range(1, len(train_accuracy) + 1), train_accuracy, label='Train Accuracy')
        plt.plot(range(1, len(val_accuracy) + 1), val_accuracy, label='Val Accuracy')
        plt.plot(range(1, len(test_accuracy) + 1), test_accuracy, label='Test Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title('Training, Validation, and Test Accuracy')
        plt.legend()
        plt.grid(True)
        plt.savefig(str(self.save_path) + '/accuracy_curve.png')  # Save accuracy curve image
        plt.show()

        # Completion message
        print("Write completed!")
