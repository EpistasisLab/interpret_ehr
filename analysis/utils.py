import numpy as np
from read_file import read_file
from eli5.sklearn import PermutationImportance
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import pdb

def feature_importance(save_file, clf, feature_names, training_features, training_classes, 
                       random_state, clf_name, clf_params, coefs=None, perm=True):
    """ prints feature importance information for a trained estimator (clf)"""
    if coefs is None:
        coefs = compute_imp_score(clf, clf_name, training_features, training_classes, 
                                  random_state, perm)
    
    assert(len(coefs)==len(feature_names))

    out_text=''
    # algorithm seed    feature score
    for i,c in enumerate(coefs):
        out_text += '\t'.join([clf_name,
                               clf_params,
                               str(random_state),
                               feature_names[i],
                               str(c)])+'\n'
       
    ext = '.perm_score' if perm else '.imp_score'
    with open(save_file.split('.csv')[0] + ext,'a') as out:
        out.write(out_text)

def compute_imp_score(pipe, clf_name, training_features, training_classes, random_state, 
                      perm):
    # clf = pipe.named_steps[clf_name]  
    clf = pipe
    # pdb.set_trace()
    if hasattr(clf, 'coef_') :
        coefs = np.abs(clf.coef_.flatten())
        coefs = coefs/np.sum(coefs)
    elif clf_name == 'ScaleLR':
        coefs = np.abs(clf.named_steps['lr'].coef_.flatten())
        coefs = coefs/np.sum(coefs)
    else:
        coefs = getattr(clf, 'feature_importances_', None)
    # print('coefs:',coefs)
   
    if coefs is None or perm:
        perm = PermutationImportance(
                                    estimator=clf,
                                    n_iter=5,
                                    random_state=random_state,
                                    refit=False
                                    )
        perm.fit(training_features, training_classes)
        coefs = perm.feature_importances_

    
    #return (coefs-np.min(coefs))/(np.max(coefs)-np.min(coefs))
    # return coefs/np.sum(coefs)
    return coefs

# def plot_imp_score(save_file, coefs, feature_names, seed):
#     # plot bar charts for top 10 importanct features
#     num_bar = min(10, len(coefs))
#     indices = np.argsort(coefs)[-num_bar:]
#     h=plt.figure()
#     plt.title("Feature importances")
#     plt.barh(range(num_bar), coefs[indices], color="r", align="center")
#     plt.yticks(range(num_bar), feature_names[indices])
#     plt.ylim([-1, num_bar])
#     h.tight_layout()
#     plt.savefig(save_file.split('.')[0] + '_imp_score_' + str(seed) + '.pdf')

######################################################################################### ROC Curve

def roc(save_file, y_true, probabilities, random_state, clf_name, clf_params):
    """prints receiver operator chacteristic curve data"""

    # pdb.set_trace()
    fpr,tpr,_ = roc_curve(y_true, probabilities)

    AUC = auc(fpr,tpr)
    # print results
    out_text=''
    for f,t in zip(fpr,tpr):
        out_text += '\t'.join([clf_name,
                               clf_params,
                               str(random_state),
                               str(f),
                               str(t),
                               str(AUC)])+'\n'

    with open(save_file.split('.csv')[0] + '.roc','a') as out:
        out.write(out_text)


