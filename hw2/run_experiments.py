import hw2.experiments as experiments
import torch

seed = 42
torch.manual_seed(seed)


# Experiments 1.1
for n_filters in [[32], [64]]:
    for n_layers in [2, 4, 8, 16]:
        experiments.run_experiment(
            'test_run', seed=seed, bs_train=50, batches=10, epochs=10, early_stopping=5,
            filters_per_layer=n_filters, layers_per_block=n_layers, pool_every=3, hidden_dims=[100],
            model_type='resnet',
        )


# Experiments 1.2
for n_layers in [2, 4, 8]:
    for n_filters in [[32], [64], [128], [256]]:
        experiments.run_experiment(
            'test_run', seed=seed, bs_train=50, batches=10, epochs=10, early_stopping=5,
            filters_per_layer=n_filters, layers_per_block=n_layers, pool_every=3, hidden_dims=[100],
            model_type='resnet',
        )


# Experiments 1.3
for n_layers in [1, 2, 3, 4]:
    experiments.run_experiment(
        'test_run', seed=seed, bs_train=50, batches=10, epochs=10, early_stopping=5,
        filters_per_layer=[64, 128, 256], layers_per_block=n_layers, pool_every=3, hidden_dims=[100],
        model_type='resnet',
    )

 # Experiments 1.4
for n_layers in [8, 16, 32]:
    experiments.run_experiment(
        'test_run', seed=seed, bs_train=50, batches=10, epochs=10, early_stopping=5,
        filters_per_layer=[32], layers_per_block=n_layers, pool_every=3, hidden_dims=[100],
        model_type='resnet',
    )

for n_layers in [2, 4, 8]:
    experiments.run_experiment(
        'test_run', seed=seed, bs_train=50, batches=10, epochs=10, early_stopping=5,
        filters_per_layer=[64, 128, 256], layers_per_block=n_layers, pool_every=3, hidden_dims=[100],
        model_type='resnet',
    )
