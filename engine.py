import torch
from torch import optim
import os
from setup_fm import DPMallzee
from setup_fm import DPMovieLens100k
from setup_fm import Sampler
from trainer import Experiment
from trainer import Trainer
from trainer import Tester
from models.fm import FMBaseline

# todo: improve loading/ saving models functionality.

BUILTIN_DATAPROVIDERS = {
    'mallzee': DPMallzee,
    'ml-100k': DPMovieLens100k
}


class _BaseEngine(object):
    def __init__(self, config):
        trial_parent_dir = config['trial_parent_dir']
        trial = config['trial']
        self.experiment_dir = os.path.join(trial_parent_dir, 'trial_{}'.format(trial))
        self.experiment = None

        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)

        config_path = os.path.join(self.experiment_dir, 'config.txt')
        with open(config_path, 'w+') as f:
            for k, v in config.items():
                f.write('{}: {}\n'.format(k, v))

    def add_experiment(self, experiment):
        self.experiment = experiment

    def run(self):
        results_path = os.path.join(self.experiment.experiment_dir, 'results.txt')
        if not os.path.exists(results_path):
            self.experiment.run()
        else:
            print('{} exists. delete folder to re-run or resume.'.format(results_path))


class FMEngine(_BaseEngine):

    def __init__(self, config):
        super(FMEngine, self).__init__(config)

        train_dp = BUILTIN_DATAPROVIDERS[config['data_name']]('train_0')
        test_dp = BUILTIN_DATAPROVIDERS[config['data_name']]('valid_0')
        train_sampler = Sampler(train_dp, config['batch_size'])
        test_sampler = Sampler(test_dp, config['batch_size'])
        trainer = Trainer(train_sampler, device=torch.device(0))
        tester = Tester(test_sampler, device=torch.device(0))

        model_class = FMBaseline
        model_params = {
            'x_dim': train_dp.item_num_features,
            'u_dim': train_dp.num_users,
            'n_factors': config['n_factors'],
            'device': torch.device(0),
            'eval_acc': config['eval_acc']}

        model = model_class(**model_params)
        optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

        experiment = Experiment(
            model=model,
            model_params=model_params,
            optimizer=optimizer,
            num_epochs=config['num_epochs'],
            trainer=trainer,
            tester=tester,
            experiment_dir=self.experiment_dir,
            use_gpu=True)

        self.add_experiment(experiment)