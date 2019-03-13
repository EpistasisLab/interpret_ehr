import multiprocessing
 
if __name__ == '__main__':
    multiprocessing.set_start_method('forkserver')
    import sys
    from xgboost import XGBClassifier # Assumes XGBoost v0.6
    import pdb
    from evaluate_model import evaluate_model
    import numpy as np

    dataset = sys.argv[1]
    save_file = sys.argv[2]
    random_seed = int(sys.argv[3])
    rare = eval(sys.argv[4])

    # Read the data set into meory
    # parameter variation
    hyper_params = {
            'n_estimators': [500],
            'gamma': [0] + list(np.logspace(-4,2,3)),
            'learning_rate':[0.001, 0.01, 0.1, 0.3]
        }
    # hyper_params = {
    #         'n_estimators': (500,),
    #     }
    # create the classifier
    clf = XGBClassifier(n_jobs=1)

    # evaluate the model
    evaluate_model(dataset, save_file, random_seed, clf, 'XGB', hyper_params,False,rare=rare)
