import numpy as np
import os
from tqdm import tqdm
import sys
from collections import OrderedDict
from collections import defaultdict
import torch


class ExperimentIO(object):

    def __init__(self):
        pass

    @staticmethod
    def load_model(model_class, filename):
        state = torch.load(f=filename)
        init_params = state['init_params']
        model = model_class(**init_params)
        model.load_state_dict(state_dict=state['network'])
        return model

    @staticmethod
    def load_checkpoint(model, optimizer, filename):
        state = torch.load(f=filename)
        model.load_state_dict(state_dict=state['network'])
        optimizer.load_state_dict(state_dict=state['optimizer'])
        return model, optimizer

    @staticmethod
    def save_checkpoint(model, optimizer, current_epoch, dirname):
        state = dict()
        state['network'] = model.state_dict()  # save network parameter and other variables.
        state['init_params'] = model.init_params
        state['optimizer'] = optimizer.state_dict()

        filename = os.path.join(dirname, 'epoch_{}'.format(current_epoch))
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        torch.save(state, f=filename)

    @staticmethod
    def save_epoch_stats(epoch_stats, filename, first_line=False):
        """
        tasks:
        - if directory does not exist it will be created.
        - if file already exists then content get's
        params:
            epoch_stats: dict {}
        remarks:
            (1) use mode = +w to overwrite.
            (2) such that column names are in a desired order.
        """

        if type(epoch_stats) is not OrderedDict:  # (2)
            raise Exception('epoch_stats must be an ordered dict. got: {}'.format(type(epoch_stats)))

        if first_line:
            mode = '+w'
        else:
            mode = '+a'

        dirname = os.path.dirname(filename)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        with open(filename, mode) as f:
            if first_line:
                header = ','.join([k for k in epoch_stats.keys()])
                f.write(header + '\n')

            line = ','.join(['{:.4f}'.format(value) for value in epoch_stats.values()])
            f.write(line + '\n')


class Tester(object):
    def __init__(self, sampler, device):
        self.sampler = sampler
        self.device = device

    @staticmethod
    def _valid_iter(model, batch):
        model.eval()
        forward_params, targets = model.process_batch(batch)
        scores = model.forward(**forward_params)
        output = model.stat_computer.compute_test_stats(scores, targets)

        return output

    def __call__(self, model, current_epoch):
        batch_stats = defaultdict(lambda: [])
        results = OrderedDict({})

        with tqdm(total=len(self.sampler)) as pbar:
            for i, batch in enumerate(self.sampler):
                output = self._valid_iter(model, batch)
                for k, v in output.items():
                    batch_stats[k].append(output[k])
                description = 'epoch: {} '.format(current_epoch) + \
                              ' '.join(["{}: {:.4f}".format(k, np.mean(v)) for k, v in batch_stats.items()])
                pbar.update(1)
                pbar.set_description(description)

        for k, v in batch_stats.items():
            results[k] = np.around(np.mean(v), decimals=4)

        return results


class Trainer(object):
    def __init__(self, sampler, device):
        self.sampler = sampler
        self.device = device

    @staticmethod
    def _train_iter(batch, model, optimizer):
        model.train()
        forward_params, targets = model.process_batch(batch)
        scores = model.forward(**forward_params)
        output = model.stat_computer.compute_train_stats(scores, targets)
        loss = model.stat_computer.compute_loss(scores, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return output, model, optimizer

    def __call__(self, model, optimizer, current_epoch):
        batch_stats = defaultdict(lambda: [])
        epoch_stats = OrderedDict({})

        with tqdm(total=len(self.sampler)) as pbar_train:
            for i, batch in enumerate(self.sampler):
                output, model, optimizer = self._train_iter(batch, model, optimizer)

                for k, v in output.items():
                    batch_stats[k].append(output[k])
                description = \
                    'epoch: {} '.format(current_epoch) + \
                    ' '.join(["{}: {:.4f}".format(k, np.mean(v)) for k, v in batch_stats.items()])
                pbar_train.update(1)
                pbar_train.set_description(description)

        for k, v in batch_stats.items():
            epoch_stats[k] = np.around(np.mean(v), decimals=4)

        return model, optimizer, epoch_stats


class Experiment(object):

    def __init__(self,
                 model,
                 model_params,
                 optimizer,
                 num_epochs,
                 trainer,
                 tester,
                 experiment_dir,
                 use_gpu=True):

        """Performs a training and validation experiment.

        Args:
            model: The model to perform the experiment with.
            model_params: The parameters to initialize the model with. These are saved when creating a checkpoint.
            optimizer: The optimizer used during training.
            num_epochs: The number of epochs to train for.
            trainer: The <Trainer>. Responsible for performing a training epoch.
            tester: The <Tester>. Responsible for performing a testing/ validation epoch.
            experiment_dir: This directory contains the checkpoints and the results of an experiment.
        """

        self.model = model
        self.model_params = model_params
        self.optimizer = optimizer
        self.experiment_dirname = experiment_dir
        self.num_epochs = num_epochs
        self.trainer = trainer
        self.tester = tester

        self.results_path = os.path.join(experiment_dir, 'results.txt')
        self.checkpoints_dir = os.path.join(experiment_dir, 'checkpoints')
        self.device = torch.device('cpu')  # default device is cpu.

        device_name = 'cpu'
        if use_gpu:
            if not torch.cuda.is_available():
                print("GPU IS NOT AVAILABLE")
            else:
                self.device = torch.device('cuda:{}'.format(0))
                device_name = torch.cuda.get_device_name(self.device)
                self.model.to(device=self.device)

        print('initialized experiment with device: {}'.format(device_name))

    def run(self):
        for current_epoch in range(self.num_epochs):
            # valid_results = self.tester_module(self.model, current_epoch)
            self.model, self.optimizer, train_results = self.trainer(self.model, self.optimizer, current_epoch)
            if current_epoch % 1 == 0 and current_epoch > -1:
                valid_results = self.tester(self.model, current_epoch)
            else:
                valid_results = {}

            sys.stderr.write('\n')
            results = OrderedDict({})
            results['current_epoch'] = current_epoch
            for k in train_results.keys():
                results[k] = train_results[k]
            for k in valid_results.keys():
                results[k] = valid_results[k]

            ExperimentIO.save_checkpoint(self.model, self.optimizer, current_epoch, dirname=self.checkpoints_dir)
            ExperimentIO.save_epoch_stats(results, self.results_path, first_line=(current_epoch == 0))