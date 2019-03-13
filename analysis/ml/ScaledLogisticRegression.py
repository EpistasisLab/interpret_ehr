import multiprocessing

if __name__ == '__main__':
    multiprocessing.set_start_method('forkserver')
    import sys
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    import pdb
    from evaluate_model import evaluate_model
    import numpy as np

    dataset = sys.argv[1]
    save_file = sys.argv[2]
    random_seed = int(sys.argv[3])
    rare = eval(sys.argv[4])

    # create the classifier
    clf = Pipeline([('scale',StandardScaler()),
                     ('lr', LogisticRegression(solver='saga',
                                               max_iter=1000,
                                               random_state=random_seed))
                     ])

    hyper_params = {
            'lr__C': np.logspace(-2,1,20),
            'lr__penalty': ['l1','l2'] 
            }
    # evaluate the model
    evaluate_model(dataset, save_file, random_seed, clf, 'ScaleLR', hyper_params, False,rare=rare)
