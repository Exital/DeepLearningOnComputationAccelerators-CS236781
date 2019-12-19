import hw2.experiments as experiments
import torch

seed = 42
torch.manual_seed(seed)


# Experiments 1_1
for n_filters in [[32], [64]]:
    for n_layers in [2, 4, 8, 16]:
        experiments.run_experiment(
            'exp1_1', seed=seed, bs_train=100, batches=10, epochs=60, early_stopping=5,
            filters_per_layer=n_filters, layers_per_block=n_layers, pool_every=3, hidden_dims=[100],
            model_type='resnet',
        )


# Experiments 1_2
for n_layers in [2, 4, 8]:
    for n_filters in [[32], [64], [128], [256]]:
        experiments.run_experiment(
            'exp1_2', seed=seed, bs_train=100, batches=10, epochs=60, early_stopping=5,
            filters_per_layer=n_filters, layers_per_block=n_layers, pool_every=2, hidden_dims=[100],
            model_type='resnet',
        )


# Experiments 1_3
for n_layers in [1, 2, 3, 4]:
    experiments.run_experiment(
        'exp1_3', seed=seed, bs_train=100, batches=10, epochs=60, early_stopping=5,
        filters_per_layer=[64, 128, 256], layers_per_block=n_layers, pool_every=3, hidden_dims=[100],
        model_type='resnet',
    )

 # Experiments 1_4
for n_layers in [8, 16, 32]:
    pool_every = int(n_layers // 4)
    experiments.run_experiment(
        'exp1_4', seed=seed, bs_train=100, batches=10, epochs=60, early_stopping=5,
        filters_per_layer=[32], layers_per_block=n_layers, pool_every=pool_every, hidden_dims=[100],
        model_type='resnet',
    )

for n_layers in [2, 4, 8]:
    if n_layers == 8:
        pool_every = 8
    else:
        pool_every = 3
    experiments.run_experiment(
        'exp1_4', seed=seed, bs_train=100, batches=10, epochs=60, early_stopping=5,
        filters_per_layer=[64, 128, 256], layers_per_block=n_layers, pool_every=pool_every, hidden_dims=[100],
        model_type='resnet',
    )


# Experiments 2
for n_layers in [3, 6, 9, 12]:
    experiments.run_experiment(
        'exp2', seed=seed, bs_train=100, batches=10, epochs=60, early_stopping=5,
        filters_per_layer=[32, 64, 128], layers_per_block=n_layers, pool_every=n_layers, hidden_dims=[200, 100],
        model_type='ycn',
    )