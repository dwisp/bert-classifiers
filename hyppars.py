rseed = 451
nnseed = 300

grid_rf = {
    'max_depth': [None, 20],
    'max_features': [20, 30],
    'max_leaf_nodes': [None, 1135]
}

par_mlp = {
    'activation': 'relu',
    'hidden_layer_sizes': (100, 200, 300, 200, 100),
    'random_state': rseed,
    'max_iter': 1000
}

par_cat = {
    'depth': 5,
    'learning_rate': 0.13,
    'iterations': 2000,
    'bagging_temperature': 0.35,
    'l2_leaf_reg': 1.5,
    'loss_function': 'MultiClass',
    'eval_metric': 'TotalF1',
    'random_state': rseed,
    'thread_count': -1,
    'task_type': 'GPU',
}