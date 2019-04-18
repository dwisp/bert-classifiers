# contains models trained on the whole of the training data
# not every one is as good as all the others, though
from os.path import join
import pickle
import helpers as hlp
import hyppars as pr
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from catboost import CatBoostClassifier

X, y = hlp.load_train()

def save_clf(clf, filename):
    with open(join(hlp.path_models, filename), 'wb') as dest:
        pickle.dump(clf, dest)

# RANDOM FOREST, ~0.94110
pars_rf = pd.read_csv(join(hlp.path_models, 'pars_rf.csv')).\
    to_dict(orient='records')[0]

pars_rf['n_estimators'] = 200
pars_rf['random_state'] = pr.rseed
pars_rf['class_weight'] = 'balanced'
pars_rf['min_samples_split'] = 3
pars_rf['n_jobs'] = -1

rf_final = RandomForestClassifier(**pars_rf)
rf_final.fit(X, y)
save_clf(rf_final, 'rf.pkl')

# MLP, 0.96286
mlp_final = MLPClassifier(**pr.par_mlp)
mlp_final.fit(X, y)
save_clf(mlp_final, 'mlp.pkl')

# CATBOOST ~0.963
cat_final = CatBoostClassifier(**pr.par_cat)
cat_final.fit(X, y)
cat_final.save_model(join(hlp.path_models, 'catboost.meow'))

# final nn is trained in train_models.py