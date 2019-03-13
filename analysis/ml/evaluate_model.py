import sys
import itertools
import pandas as pd
from sklearn.model_selection import (StratifiedKFold, cross_val_predict, 
                                     GridSearchCV, ParameterGrid, train_test_split)
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, make_scorer
from imblearn.pipeline import Pipeline,make_pipeline
from metrics import balanced_accuracy
#from imblearn.under_sampling import NearMiss
from quartile_exact_match import QuartileExactMatch
import warnings
import time
from tempfile import mkdtemp
from shutil import rmtree
from sklearn.externals.joblib import Memory
from read_file import read_file
from utils import feature_importance, compute_imp_score, roc
import pdb
import numpy as np

def evaluate_model(dataset, save_file, random_state, clf, clf_name, hyper_params, 
                   longitudinal=False,rare=True):

    print('reading data...',end='')
    features, labels, pt_ids, feature_names, zfile = read_file(dataset,longitudinal,rare)
    print('done.',len(labels),'samples,',np.sum(labels==1),'cases,',features.shape[1],'features')
    if 'Feat' in clf_name:
        #set feature names
        clf.feature_names = ','.join(feature_names).encode()
    n_splits=10
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True,random_state=random_state)

    scoring = make_scorer(balanced_accuracy)
 
    ### 
    # controls matching on age and sex
    ###
    idx_age = np.argmax(feature_names == 'age')
    idx_sex = np.argmax(feature_names == 'SEX')

    #sampler = NearMiss(random_state=random_state, return_indices=True)
    sampler = QuartileExactMatch(quart_locs=[idx_age],exact_locs = [idx_sex],
                                 random_state=random_state)
     
    print('sampling data...',end='')
    X,y,sidx = sampler.fit_sample(features,labels)
    print('sampled data contains',np.sum(y==1),'cases',np.sum(y==0),'controls')
    ### 
    # split into train/test 
    ###
    X_train, X_test, y_train, y_test, sidx_train, sidx_test = (
            train_test_split(X, y, sidx,
                             train_size=0.5,
                             test_size=0.5,
                             random_state=random_state))

    # X,y,sidx = sampler.fit_sample(features[train_idx],labels[train_idx])
    if len(hyper_params) > 0:
        param_grid = list(ParameterGrid(hyper_params))
        #clone estimators
        Clfs = [clone(clf).set_params(**p) for p in param_grid]
        # fit with hyperparameter optimization 
        cv_scores = np.zeros((len(param_grid),10))              # cross validated scores
        cv_preds = np.zeros((len(param_grid),len(y_train)))      # cross validated predictions
        cv_probs = np.zeros((len(param_grid),len(y_train)))      # cross validated probabilities
        FI = np.zeros((len(param_grid),features.shape[1]))     # cross validated, permuted feature importance
        FI_internal = np.zeros((len(param_grid),features.shape[1]))     # cross validated feature importance

        ###########
        # this is a manual version of 10-fold cross validation with hyperparameter tuning
        t0 = time.process_time()
        for j,(train_idx, val_idx) in enumerate(cv.split(X_train,y_train)):
            print('fold',j)

            for i,est in enumerate(Clfs):
                print('training',type(est).__name__,i+1,'of',len(Clfs))
                if 'Feat' in clf_name:
                    est.logfile = (est.logfile.decode().split('.log')[0] + '.log.param' + str(i)
                                   + '.cv' + str(j)).encode()           
                ##########
                # fit model
                ##########
                if longitudinal:
                    est.fit(X_train[train_idx],y_train[train_idx],
                            zfile,pt_ids[sidx_train[train_idx]])
                else:
                    est.fit(X_train[train_idx],y_train[train_idx])
                
                ##########
                # get predictions
                ##########
                print('getting validation predictions...')
                if longitudinal:
                    # cv_preds[i,val_idx] = est.predict(X_train[val_idx], 
                    #                                    zfile,pt_ids[sidx_train[train_idx]])
                    if getattr(clf, "predict_proba", None):
                        cv_probs[i,val_idx] = est.predict_proba(X_train[val_idx],
                                                                 zfile,
                                                                 pt_ids[sidx_train[train_idx]])[:,1]
                    elif getattr(clf, "decision_function", None):
                        cv_probs[i,val_idx] = est.decision_function(X_train[val_idx],
                                                                 zfile,
                                                                 pt_ids[sidx_train[train_idx]])
                else:
                    # cv_preds[i,val_idx] = est.predict(X_train[val_idx])
                    if getattr(clf, "predict_proba", None):
                        cv_probs[i,val_idx] = est.predict_proba(X_train[val_idx])[:,1]
                    elif getattr(clf, "decision_function", None):
                        cv_probs[i,val_idx] = est.decision_function(X_train[val_idx])
                
                ##########
                # scores
                ##########
                cv_scores[i,j] = roc_auc_score(y_train[val_idx], cv_probs[i,val_idx])

        runtime = time.process_time() - t0
        ###########
        
        print('gridsearch finished in',runtime,'seconds') 
       
        ##########
        # get best model and its information
        mean_cv_scores = [np.mean(s) for s in cv_scores]
        best_clf = Clfs[np.argmax(mean_cv_scores)]
        ##########
    else:
        print('skipping hyperparameter tuning')
        best_clf = clf  # this option is for skipping model tuning
        t0 = time.process_time()


    print('fitting tuned model to all training data...')
    if longitudinal:
        best_clf.fit(X_train, y_train, zfile, pt_ids[sidx_train])
    else:
        best_clf.fit(X_train,y_train)

    if len(hyper_params)== 0: 
        runtime = time.process_time() - t0
    # cv_predictions = cv_preds[np.argmax(mean_cv_scores)]
    # cv_probabilities = cv_probs[np.argmax(mean_cv_scores)]
    if not longitudinal:
        # internal feature importances
        cv_FI_int = compute_imp_score(best_clf,clf_name,X_train, y_train,random_state,perm=False)
        # cv_FI_int = FI_internal[np.argmax(mean_cv_scores)]
        # permutation importances
        FI = compute_imp_score(best_clf, clf_name, X_test, y_test, random_state, perm=True)
        
    ##########
    # metrics: test the best classifier on the held-out test set 
    print('getting test predictions...')
    if longitudinal:

        print('best_clf.predict(X_test, zfile, pt_ids[sidx_test])')
        test_predictions = best_clf.predict(X_test, zfile, pt_ids[sidx_test])
        if getattr(clf, "predict_proba", None):
            print('best_clf.predict_proba(X_test, zfile, pt_ids[sidx_test])')
            test_probabilities = best_clf.predict_proba(X_test,
                                                 zfile,
                                                 pt_ids[sidx_test])[:,1]
        elif getattr(clf, "decision_function", None):
            test_probabilities = best_clf.decision_function(X_test,
                                                     zfile,
                                                     pt_ids[sidx_test])
    else:
        test_predictions = best_clf.predict(X_test)
        if getattr(clf, "predict_proba", None):
            test_probabilities = best_clf.predict_proba(X_test)[:,1]
        elif getattr(clf, "decision_function", None):
            test_probabilities = best_clf.decision_function(X_test)

    # # write cv_pred and cv_prob to file
    # df = pd.DataFrame({'cv_prediction':cv_predictions,'cv_probability':cv_probabilities,
    #                    'pt_id':pt_ids})
    # df.to_csv(save_file.split('.csv')[0] + '_' + str(random_state) + '.cv_predictions',index=None)
    accuracy = accuracy_score(y_test, test_predictions)
    macro_f1 = f1_score(y_test, test_predictions, average='macro')
    bal_acc = balanced_accuracy(y_test, test_predictions)
    roc_auc = roc_auc_score(y_test, test_probabilities)

    ##########
    # save results to file
    print('saving results...')
    param_string = ','.join(['{}={}'.format(p, v) 
                             for p,v in best_clf.get_params().items() 
                             if p!='feature_names']).replace('\n','').replace(' ','')

    out_text = '\t'.join([dataset.split('/')[-1],
                          clf_name,
                          param_string,
                          str(random_state), 
                          str(accuracy),
                          str(macro_f1),
                          str(bal_acc),
                          str(roc_auc),
                          str(runtime)])
    print(out_text)
    with open(save_file, 'a') as out:
        out.write(out_text+'\n')
    sys.stdout.flush()

    print('saving feature importance') 
    # write feature importances
    if not longitudinal:
        feature_importance(save_file, best_clf, feature_names, X_test, y_test, random_state, 
                           clf_name, param_string, cv_FI_int,perm=False)
        feature_importance(save_file, best_clf, feature_names, X_test, y_test, random_state, 
                           clf_name, param_string, FI,perm=True)
    # write roc curves
    print('saving roc') 
    roc(save_file, y_test, test_probabilities, random_state, clf_name,param_string)

    return best_clf
