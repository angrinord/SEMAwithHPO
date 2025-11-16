import logging
import math

import numpy as np
import torch
from ConfigSpace import ConfigurationSpace, UniformIntegerHyperparameter
from smac import Scenario, AlgorithmConfigurationFacade
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from backbone.sema_block import SEMAModules
from models.base import BaseLearner
from utils.inc_net import SEMAVitNet
from utils.toolkit import tensor2numpy

num_workers = 8


class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self.backup_state = None
        self.bn_default = 16
        self.bn_min = 8
        self.bn_max = 64
        bn_stride = 16
        self.hpo_trials = (self.bn_max-self.bn_min)//bn_stride + 1
        self.hpo_epochs = 1
        self.reg_lambda = 0.001
        self.last_test_acc = 0.0
        self.subset_size = 64   # -1 means use the whole training set
        self.hpo_train_loader = None
        self.hpo_val_loader = None
        self._network = SEMAVitNet(args, True)
        self. batch_size = args["batch_size"]
        self. init_lr = args["init_lr"]
        
        self.weight_decay = args["weight_decay"] if args["weight_decay"] is not None else 0.0005
        self.min_lr = args['min_lr'] if args['min_lr'] is not None else 1e-8
        self.args = args

    def after_task(self):
        self._known_classes = self._total_classes

    def incremental_train(self, data_manager):
        self._cur_task += 1
        if self._cur_task == 0:
            self._network.fc = nn.Linear(768, data_manager.nb_classes)
            nn.init.kaiming_uniform_(self._network.fc.weight, a=math.sqrt(5))
            nn.init.zeros_(self._network.fc.bias)
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))

        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),source="train", mode="train", )
        self.train_dataset = train_dataset
        self.data_manager = data_manager
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test")
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)

        # do hpo on a subset of next partition of dataset
        if self.subset_size > 0:
            indices = torch.randperm(len(train_dataset))[:self.subset_size]
            subset = torch.utils.data.Subset(train_dataset, indices)
            val_size = max(1, int(0.2 * self.subset_size))
            train_size = self.subset_size - val_size
            hpo_train_subset, hpo_val_subset = torch.utils.data.random_split(
                subset, [train_size, val_size]
            )
            self.hpo_train_loader = DataLoader(
                hpo_train_subset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=num_workers,
            )
            self.hpo_val_loader = DataLoader(
                hpo_val_subset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=num_workers,
            )
        # do hpo on entire partition
        else:
            self.hpo_train_loader = self.train_loader
            self.hpo_val_loader = self.test_loader

        train_dataset_for_protonet=data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),source="train", mode="test", )
        self.train_loader_for_protonet = DataLoader(train_dataset_for_protonet, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)

        if len(self._multiple_gpus) > 1:
            print('Multiple GPUs')
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader):
        
        self._network.to(self._device)
        
        if self._cur_task == 0:
            # show total parameters and trainable parameters
            total_params = sum(p.numel() for p in self._network.parameters())
            print(f'{total_params:,} total parameters.')
            total_trainable_params = sum(
                p.numel() for p in self._network.parameters() if p.requires_grad)
            print(f'{total_trainable_params:,} training parameters.')
            self._train_new(train_loader, test_loader)
        else:
            for module in self._network.backbone.modules():
                if isinstance(module, SEMAModules):
                    module.detecting_outlier = True
            detect_loader = DataLoader(train_loader.dataset, batch_size=self.args["detect_batch_size"], shuffle=True, num_workers=num_workers)     
            added = self._detect_outlier(detect_loader, train_loader, test_loader, 0)

            for module in self._network.backbone.modules():
                if isinstance(module, SEMAModules):
                    module.detecting_outlier = False
            if added == 0:
                self.update_optimizer_and_scheduler(num_epoch=self.args['func_epoch'], lr=self.init_lr)
                self._init_train(self.args['func_epoch'], train_loader, test_loader, self.optimizer, self.scheduler, phase='func')
            
        for module in self._network.backbone.modules():
            if isinstance(module, SEMAModules):
                module.end_of_task_training()

    def _train_new(self, train_loader, test_loader):
        self.update_optimizer_and_scheduler(num_epoch=self.args['func_epoch'], lr=self.init_lr)
        self._init_train(self.args['func_epoch'], train_loader, test_loader, self.optimizer, self.scheduler, phase='func')
        self.update_rd_optimizer_and_scheduler(num_epoch=self.args['rd_epoch'], lr=self.args['rd_lr'])
        self._init_train(self.args['rd_epoch'], train_loader, test_loader, self.rd_optimizer, self.rd_scheduler, phase='rd')

    def _detect_outlier(self, detect_loader, train_loader, test_loader, added):
        is_added = False
        for i, (_, inputs, targets) in enumerate(detect_loader):
            inputs, targets = inputs.to(self._device), targets.to(self._device)
            model_outcome = self._network(inputs)
            added_record = model_outcome["added_record"]

            if sum(added_record) > 0:
                # TODO: Backing up model, performing hpo on it, resetting after each run of hpo, and then resetting after hpo is finished is currently broken.
                # 1. backup adapters
                self.backup_state = {name: p.detach().clone() for name, p in self._network.named_parameters() if p.requires_grad}

                # 2. do hpo
                best_ffn = self.hpo()
                logging.info(f"[HPO] Best bottleneck for next adapter: {best_ffn}")

                # # 3. restore adapters
                sd = self._network.state_dict()
                for name, value in self.backup_state.items():
                    sd[name].copy_(value)

                # 4. reinitialize newest adapter with hpo-selected bottleneck
                for module in self._network.backbone.modules():
                    if isinstance(module, SEMAModules):
                        module.set_next_bottleneck(best_ffn)
                        module.reinitialize_latest_adapter(best_ffn)

                added += 1
                is_added = True
                for module in self._network.backbone.modules():
                    if isinstance(module, SEMAModules):
                        module.detecting_outlier = False

                self._train_new(train_loader, test_loader)

                for module in self._network.backbone.modules():
                    if isinstance(module, SEMAModules):
                        module.detecting_outlier = True
                        module.freeze_functional()
                        module.freeze_rd()
                        module.reset_newly_added_status()
        if is_added:
            return self._detect_outlier(detect_loader, train_loader, test_loader, added)
        return added

    def _init_train(self, total_epoch, train_loader, test_loader, optimizer, scheduler, phase='func'):
        prog_bar = tqdm(range(total_epoch))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                outcome = self._network(inputs)

                logits = outcome["logits"]
                logits = logits[:, :self._total_classes]
                if self._cur_task > 0:
                    logits[:, :self._known_classes] = -float('inf')

                if phase == "func":
                    loss = F.cross_entropy(logits, targets)
                elif phase == "rd":
                    logits = outcome["logits"]
                    loss = outcome["rd_loss"]

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            test_acc = self._compute_accuracy(self._network, test_loader)
            info = "{} Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                phase,
                self._cur_task,
                epoch + 1,
                total_epoch,
                losses / len(train_loader),
                train_acc,
                test_acc,
            )
            prog_bar.set_description(info)
        logging.info(info)
        if phase == "func":
            self.last_test_acc = float(test_acc)

    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outcome = self._network(inputs)
                logits = outcome["logits"]
                outputs = logits[:, :self._total_classes]
            predicts = torch.topk(
                outputs, k=self.topk, dim=1, largest=True, sorted=True
            )[
                1
            ]  # [bs, topk]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]
    
    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outcome = self._network(inputs)
                logits = outcome["logits"]
                outputs = logits[:, :self._total_classes]
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)

        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)

    def update_optimizer_and_scheduler(self, num_epoch=20, lr=None):
        lr = self.args["init_lr"] if lr is None else lr
        func_params = [p for n,p in self._network.named_parameters() if ('functional' in n or 'router' in n or 'fc' in n) and p.requires_grad]
        if self.args['optimizer']=='sgd':
            self.optimizer = optim.SGD(func_params, momentum=0.9, lr=lr,weight_decay=self.args["weight_decay"])
        elif self.args['optimizer']=='adam':
            self.optimizer = optim.AdamW(func_params, lr=lr, weight_decay=self.args["weight_decay"])

        min_lr = self.args['min_lr'] if self.args['min_lr'] is not None else 1e-8
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=num_epoch, eta_min=min_lr)    

    def update_rd_optimizer_and_scheduler(self, num_epoch=20, lr=None):
        lr = self.args["rd_lr"] if lr is None else lr
        rd_params = [p for n,p in self._network.named_parameters() if 'rd' in n and p.requires_grad]
        if self.args['optimizer']=='sgd':
            self.rd_optimizer = optim.SGD(rd_params, momentum=0.9, lr=lr,weight_decay=self.args["weight_decay"])
        elif self.args['optimizer']=='adam':
            self.rd_optimizer = optim.AdamW(rd_params, lr=lr, weight_decay=self.args["weight_decay"])
        
        min_lr = self.args['min_lr'] if self.args['min_lr'] is not None else 1e-8
        self.rd_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.rd_optimizer, T_max=num_epoch, eta_min=min_lr) if self.rd_optimizer else None
        
    def save_checkpoint(self, filename):
        state_dict = self._network.state_dict()
        save_dict = {}
        for k, v in state_dict.items():
            if 'adapter' in k or ('fc' in k and 'block' not in k):
                save_dict[k] = v
        torch.save(save_dict, "{}.pth".format(filename))

    def load_checkpoint(self, filename):
        self._network.load_state_dict(torch.load(filename), strict=False)

    def _objective(self, config, seed):
        sd = self._network.state_dict()
        for name, value in self.backup_state.items():
            sd[name].copy_(value)

        ffn_num = int(config["ffn_num"])
        torch.manual_seed(seed)
        np.random.seed(seed)

        # 1. Apply candidate ffn_num for this trial
        for module in self._network.backbone.modules():
            if isinstance(module, SEMAModules):
                module.set_next_bottleneck(ffn_num)
                if module.adapters[-1].newly_added:
                    module.reinitialize_latest_adapter(ffn_num)

        # 2. Run the miniature training loop on HPO subset
        self.update_optimizer_and_scheduler(self.hpo_epochs, lr=self.init_lr)
        self._init_train(self.hpo_epochs,
                         self.hpo_train_loader,
                         self.hpo_val_loader,
                         self.optimizer,
                         self.scheduler,
                         phase="func")

        acc = self.last_test_acc
        return (1 - acc / 100.0) + self.reg_lambda * ffn_num

    def hpo(self):
        cs = ConfigurationSpace()
        cs.add_hyperparameter(UniformIntegerHyperparameter("ffn_num", lower=self.bn_min, upper=self.bn_max, default_value=self.bn_default))
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        od = f"./smac_output/task_{timestamp}"
        scenario = Scenario(
            configspace=cs,
            n_trials=self.hpo_trials,
            deterministic=True,
            output_directory=od,
            seed=0
        )
        smac = AlgorithmConfigurationFacade(
            scenario,
            target_function=self._objective
        )
        incumbent = smac.optimize()
        best_ffn = incumbent["ffn_num"]
        for module in self._network.backbone.modules():
            if isinstance(module, SEMAModules):
                module.set_next_bottleneck(best_ffn)
        return best_ffn
