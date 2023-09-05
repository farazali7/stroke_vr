import os
import time

import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedGroupKFold, RandomizedSearchCV
from sklearn.metrics import f1_score
from sklearn.utils.fixes import loguniform

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier

from scipy.stats import uniform, randint

from src.utils.data.io import slice_features_and_labels
from src.utils.metrics import compute_metrics

from config import cfg

if __name__ == '__main__':
    dataset = pd.read_pickle(cfg['PATHS']['FEATURE_DATA'] + '/feature_data.pkl')
    label_cols = cfg['RESPONSE_VARS']

    gt_metric = 'perceived_exertion'  # One of {'perceived_exertion', 'perceived_enjoyment', 'game_performance'}
    feature_data_type = 'PCA'  # One of {'kinematic', 'PCA'}

    for gt_metric in label_cols:
        X_kin, X_pca, y_all, groups = slice_features_and_labels(dataset, label_cols)
        X = X_kin if feature_data_type == 'kinematic' else X_pca
        y = y_all[gt_metric].values

        # Split off trainval & test set
        X_train, X_test, y_train, y_test, groups_train, groups_test = train_test_split(X, y, groups,
                                                                                       shuffle=True,
                                                                                       random_state=cfg['SEED'],
                                                                                       stratify=y, test_size=0.2)
        # sgkf_holdout = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=cfg['SEED'])
        # train_idxs, test_idxs = next(iter(sgkf_holdout.split(X=X, y=y, groups=groups)))
        # X_train, y_train, X_test, y_test = X[train_idxs], y[train_idxs], X[test_idxs], y[test_idxs]
        groups = groups_train

        # Create CV splitter that is grouped by subject
        sgkf = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=cfg['SEED'])

        results_base_path = cfg['PATHS']['RESULTS']
        if not os.path.exists(results_base_path):
            os.makedirs(results_base_path, exist_ok=True)
        exp_base = f'{feature_data_type}_{gt_metric}'
        results_file_name = f'{exp_base}_results.csv'
        results_path = os.path.join(results_base_path, results_file_name)
        results = pd.DataFrame(columns=['Model', 'Accuracy', 'AUC', 'F1-score', 'Recall', 'Precision'])

        # LOGISTIC REGRESSION
        splits = sgkf.split(X=X_train, y=y_train, groups=groups)
        print(f"\n--- STARTING TRAINING LOGISTIC REGRESSION ---\n")
        s = time.time()
        log_reg = LogisticRegression(verbose=1, class_weight='balanced', n_jobs=os.cpu_count())
        dist = {
            'C': uniform(loc=0, scale=4)
        }
        log_reg = RandomizedSearchCV(log_reg, dist, n_iter=5, scoring='f1',
                                     n_jobs=os.cpu_count(), cv=splits, verbose=1,
                                     random_state=cfg['SEED'])
        log_reg.fit(X_train, y_train)
        if log_reg.best_score_ == 0.0:
            print(f'NO GOOD HPARAMS FOUND - FITTING ON DEFAULT...')
            log_reg = LogisticRegression(verbose=1, class_weight='balanced', n_jobs=os.cpu_count()).fit(X_train, y_train)
        e = time.time()
        print(f"\n--- DONE TRAINING LOGISTIC REGRESSION, TOOK: {round(e - s, 2)}s ---\n")
        log_reg_preds = log_reg.predict(X_test)
        log_reg_metrics = compute_metrics(y_test, log_reg_preds)
        results.loc[len(results)] = ['Logistic Regression', *log_reg_metrics]
        results.to_csv(results_path, index=False)

        hparam_path = os.path.join(results_base_path, exp_base + '_log_reg_hparam.csv')
        hparam_res = pd.DataFrame(log_reg.cv_results_)
        hparam_res.to_csv(hparam_path, index=False)

        # KNN
        splits = sgkf.split(X=X_train, y=y_train, groups=groups)
        print(f"\n--- STARTING TRAINING KNN ---\n")
        s = time.time()
        knn = KNeighborsClassifier(n_jobs=os.cpu_count())
        dist = {
            'n_neighbors': randint(low=1, high=20),
            'weights': ['uniform', 'distance']
        }
        knn = RandomizedSearchCV(knn, dist, n_iter=5, scoring='f1',
                                 n_jobs=os.cpu_count(), cv=splits, verbose=1,
                                 random_state=cfg['SEED'])
        knn.fit(X_train, y_train)
        e = time.time()
        print(f"\n--- DONE TRAINING KNN, TOOK: {round(e - s, 2)}s ---\n")
        knn_preds = knn.predict(X_test)
        knn_metrics = compute_metrics(y_test, knn_preds)
        results.loc[len(results)] = ['KNN', *knn_metrics]
        results.to_csv(results_path, index=False)

        hparam_path = os.path.join(results_base_path, exp_base + '_knn_hparam.csv')
        hparam_res = pd.DataFrame(knn.cv_results_)
        hparam_res.to_csv(hparam_path, index=False)

        # RANDOM FOREST
        splits = sgkf.split(X=X_train, y=y_train, groups=groups)
        print(f"\n--- STARTING TRAINING RANDOM FOREST ---\n")
        s = time.time()
        rf = RandomForestClassifier(verbose=1, class_weight='balanced', n_jobs=os.cpu_count())
        dist = {
            'criterion': ['gini', 'entropy', 'log_loss'],
            'n_estimators': randint(low=100, high=1000),
            'max_depth': [None] + list(range(5, 20)),
            'min_samples_split': randint(low=2, high=10),
            'min_samples_leaf': randint(low=1, high=10)
        }
        rf = RandomizedSearchCV(rf, dist, n_iter=5, scoring='f1',
                                n_jobs=os.cpu_count(), cv=splits, verbose=1,
                                random_state=cfg['SEED'])
        rf.fit(X_train, y_train)
        e = time.time()
        print(f"\n--- DONE TRAINING RANDOM FOREST, TOOK: {round(e - s, 2)}s ---\n")
        rf_preds = rf.predict(X_test)
        rf_metrics = compute_metrics(y_test, rf_preds)
        results.loc[len(results)] = ['Random Forest', *rf_metrics]
        results.to_csv(results_path, index=False)

        hparam_path = os.path.join(results_base_path, exp_base + '_rf_hparam.csv')
        hparam_res = pd.DataFrame(rf.cv_results_)
        hparam_res.to_csv(hparam_path, index=False)

        # SVC
        splits = sgkf.split(X=X_train, y=y_train, groups=groups)
        print(f"\n--- STARTING TRAINING SVC ---\n")
        s = time.time()
        svc = LinearSVC(verbose=1, class_weight='balanced')
        dist = {
            'C': uniform(loc=0, scale=4),
            'penalty': ['l2']
        }
        svc = RandomizedSearchCV(svc, dist, n_iter=5, scoring='f1',
                                 n_jobs=os.cpu_count(), cv=splits, verbose=1,
                                 random_state=cfg['SEED'])
        svc.fit(X_train, y_train)
        e = time.time()
        print(f"\n--- DONE TRAINING SVC, TOOK: {round(e - s, 2)}s ---\n")
        svc_preds = svc.predict(X_test)
        svc_metrics = compute_metrics(y_test, svc_preds)
        results.loc[len(results)] = ['Support Vector Machine', *svc_metrics]
        results.to_csv(results_path, index=False)

        hparam_path = os.path.join(results_base_path, exp_base + '_svc_hparam.csv')
        hparam_res = pd.DataFrame(svc.cv_results_)
        hparam_res.to_csv(hparam_path, index=False)

        # NAIVE BAYES
        splits = sgkf.split(X=X_train, y=y_train, groups=groups)
        print(f"\n--- STARTING TRAINING NAIVE BAYES ---\n")
        s = time.time()
        nb = GaussianNB()
        dist = {}
        nb = RandomizedSearchCV(nb, dist, n_iter=5, scoring='f1',
                                n_jobs=os.cpu_count(), cv=splits, verbose=1,
                                random_state=cfg['SEED'])
        nb.fit(X_train, y_train)
        e = time.time()
        print(f"\n--- DONE TRAINING NAIVE BAYES, TOOK: {round(e - s, 2)}s ---\n")
        nb_preds = nb.predict(X_test)
        nb_metrics = compute_metrics(y_test, nb_preds)
        results.loc[len(results)] = ['Naive Bayes', *nb_metrics]
        results.to_csv(results_path, index=False)

        hparam_path = os.path.join(results_base_path, exp_base + '_nb_hparam.csv')
        hparam_res = pd.DataFrame(nb.cv_results_)
        hparam_res.to_csv(hparam_path, index=False)

        # MLP
        splits = sgkf.split(X=X_train, y=y_train, groups=groups)
        print(f"\n--- STARTING TRAINING MLP ---\n")
        s = time.time()
        mlp = MLPClassifier(verbose=1, early_stopping=True)
        dist = {'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 100)],
                'activation': ['relu', 'tanh'],
                'alpha': uniform(loc=0, scale=0.1),
                'learning_rate_init': loguniform(1e-5, 1e-3)}
        mlp = RandomizedSearchCV(mlp, dist, n_iter=5, scoring='f1',
                                 n_jobs=os.cpu_count(), cv=splits, verbose=1,
                                 random_state=cfg['SEED'])
        mlp.fit(X_train, y_train)
        e = time.time()
        print(f"\n--- DONE TRAINING MLP, TOOK: {round(e - s, 2)}s ---\n")
        mlp_preds = mlp.predict(X_test)
        mlp_metrics = compute_metrics(y_test, mlp_preds)
        results.loc[len(results)] = ['Multi-layer Perceptron', *mlp_metrics]
        results.to_csv(results_path, index=False)

        hparam_path = os.path.join(results_base_path, exp_base + '_mlp_hparam.csv')
        hparam_res = pd.DataFrame(mlp.cv_results_)
        hparam_res.to_csv(hparam_path, index=False)

        # AdaBoost
        splits = sgkf.split(X=X_train, y=y_train, groups=groups)
        print(f"\n--- STARTING TRAINING ADABOOST ---\n")
        s = time.time()
        ab = AdaBoostClassifier()
        dist = {'n_estimators': randint(low=50, high=200),
                'learning_rate': uniform(loc=0.01, scale=1.0)}
        ab = RandomizedSearchCV(ab, dist, n_iter=5, scoring='f1',
                                n_jobs=os.cpu_count(), cv=splits, verbose=1,
                                random_state=cfg['SEED'])
        ab.fit(X_train, y_train)
        e = time.time()
        print(f"\n--- DONE TRAINING ADABOOST, TOOK: {round(e - s, 2)}s ---\n")
        ab_preds = ab.predict(X_test)
        ab_metrics = compute_metrics(y_test, ab_preds)
        results.loc[len(results)] = ['AdaBoost', *ab_metrics]
        results.to_csv(results_path, index=False)

        hparam_path = os.path.join(results_base_path, exp_base + '_ab_hparam.csv')
        hparam_res = pd.DataFrame(ab.cv_results_)
        hparam_res.to_csv(hparam_path, index=False)

        print('Done')
