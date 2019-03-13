import multiprocessing

if __name__ == '__main__':
    multiprocessing.set_start_method('forkserver')
    import sys
    from sklearn.feature_selection import f_classif
    # from sklearn.linear_model import LogisticRegression
    from p_values_for_logreg import LogisticReg
    from sklearn.preprocessing import StandardScaler
    import pdb
    import numpy as np
    from read_file import read_file
    dataset = sys.argv[1]
    save_file = sys.argv[2]
    random_state = int(sys.argv[3])
    rare = eval(sys.argv[4])
    longitudinal=False

    features, labels, pt_ids, feature_names, zfile = read_file(dataset,longitudinal,rare)
    scores=[]
    # scale the data
    features = StandardScaler().fit_transform(features)
    # create the classifier
    for i in np.arange(features.shape[1]):
        print('fitting feature',feature_names[i],i,'of',features.shape[1])
        est = LogisticReg(solver='saga',
                                 C = 1000, 
                                 random_state=random_state)
        est.fit(features[:,i].reshape(-1,1),labels)
        print('pvalue:',est.p_values[0],
              'weight:',np.abs(est.model.coef_.flatten()[0]))
        if est.p_values[0] < 0.05:
            scores.append(np.abs(est.model.coef_.flatten()[0]))
        else:
            scores.append(0)
    # save file
    out_text=''
    param_string = ','.join(['{}={}'.format(p, v) 
                             for p,v in est.model.get_params().items()])

    # algorithm seed    feature score
    for i,c in enumerate(scores):
        out_text += '\t'.join(['Univariate LR',
                               param_string,
                               str(random_state),
                               feature_names[i],
                               str(c)])+'\n'
    import os
    if os.path.exists(save_file.split('.csv')[0]+'.imp_score'):
        os.remove(save_file.split('.csv')[0]+'.imp_score')
    if os.path.exists(save_file.split('.csv')[0]+'.roc'):
        os.remove(save_file.split('.csv')[0]+'.roc')

    ext = '.univariate_score'
    with open(save_file.split('.csv')[0] + ext,'w') as out:
        out.write('algorithm\talg-parameters\tseed\tfeature\tscore\n')
        out.write(out_text)
