from engine import FMEngine

config = {
    'data_name': 'mallzee',
    'n_factors': 4,
    'eval_acc': True,
    'learning_rate': 0.001,
    'batch_size': 256,
    'num_epochs': 100
}

engine = FMEngine(config)
engine.run()

