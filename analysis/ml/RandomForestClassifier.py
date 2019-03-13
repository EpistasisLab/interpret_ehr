import multiprocessing
 
if __name__ == '__main__':
    multiprocessing.set_start_method('forkserver')
    import sys
    from sklearn.ensemble import RandomForestClassifier
    import pdb
    from evaluate_model import evaluate_model
    import numpy as np

    dataset = sys.argv[1]
    save_file = sys.argv[2]
    random_seed = int(sys.argv[3])
    rare = eval(sys.argv[4])

    # Read the data set into meory
    # parameter variation
    hyper_params = [{
       'n_estimators': [500],
       'min_samples_leaf': np.logspace(-4,-1,4),
       'max_features': ('sqrt','log2',None),
    }]

    # hyper_params = {
    #    'n_estimators': [100,500],
    #    'criterion': ('gini',)
    # }

    # create the classifier
    clf = RandomForestClassifier(class_weight='balanced',n_jobs=1)

    # evaluate the model
    evaluate_model(dataset, save_file, random_seed, clf, 'RF', hyper_params,False,rare=rare)
