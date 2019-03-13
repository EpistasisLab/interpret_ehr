import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import seaborn as sns
sns.set_style("whitegrid")
import math
import argparse
from glob import glob
import pdb
def main():
    """Analyzes results and generates figures."""
 
    parser = argparse.ArgumentParser(description="An analyst for quick ML applications.",
                                     add_help=False)
  
    parser.add_argument('RUN_DIR', action='store',  type=str, help='Path to results from analysis.')    
    parser.add_argument('-max_feat',action='store',dest='MAX_FEAT',default=10,type=int,
                        help = 'Max features to show in importance plots.')
    args = parser.parse_args()
   
    # dataset = args.NAME
    # dataset = args.NAME.split('/')[-1].split('.')[0] 
    # run_dir = 'results/' + dataset + '/' 
    run_dir = args.RUN_DIR
    if run_dir[-1] != '/': 
        run_dir += '/'
    dataset = run_dir.split('/')[-2]
    print('dataset:',dataset)
    print('loading data from',run_dir)

    frames = []     # data frames to combine
    count = 0
    for f in glob(run_dir + '*.csv'):
        if 'imp_score' not in f:
            frames.append(pd.read_csv(f,sep='\t',index_col=False))
            count = count + 1

    df = pd.concat(frames, join='outer', ignore_index=True)
    print('loaded',count,'result files with results from these learners:',df['algorithm'].unique())

    restricted_cols = ['prep_alg','preprocessor', 'prep-parameters', 'algorithm', 'alg-parameters','dataset',
                      'trial','seed','parameters']
    columns_to_plot = [c for c in df.columns if c not in restricted_cols ] 
    #['accuracy','f1_macro','bal_accuracy']
    print('generating boxplots for these columns:',columns_to_plot)

    for col in columns_to_plot:
        fig = plt.figure()
        # for i, prep in enumerate(unique_preps):
            # fig.add_subplot(math.ceil(len(unique_preps)), 2,i+1)
        # pdb.set_trace()
        df[col] = df[col].astype(np.float)
        sns.boxplot(data=df,x="algorithm",y=col)
        # plt.title(prep,size=16)
        plt.gca().set_xticklabels(df.algorithm.unique(),size=14,rotation=45)
        plt.ylabel(col,size=16)
        plt.ylim(0.5,1.0)
        plt.xlabel('')
        fig.tight_layout() 
        plt.savefig(run_dir + '_'.join([ dataset, col,'boxplots.pdf']))

    ####################################################################### feature importance plots
    frames = []     # data frames to combine
    count = 0
    for f in glob(run_dir + '*.imp_score'):
        frames.append(pd.read_csv(f,sep='\t',index_col=False))
        count = count + 1

    df = pd.concat(frames, join='outer', ignore_index=True)
    print('loaded',count,'feature importance files with results from these learners:',df['algorithm'].unique())
    
    dfp =  df.groupby(['algorithm','feature']).median().unstack(['algorithm'])
    dfpn = df.groupby(['feature','algorithm']).median().groupby('feature').sum().unstack()
    
    dfpn.sort_values(ascending=False, inplace=True)
    # sort by median feature importance
    nf = min(args.MAX_FEAT, dfpn.index.labels[1].shape[0])
    dfpw = dfp.iloc[dfpn.index.labels[1][:nf]]
    h = dfpw['score'].plot(kind='bar', stacked=True)
    leg = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.ylabel('Importance Score')
    plt.savefig(run_dir + '_'.join([ dataset, 'importance_scores.pdf']),bbox_extra_artists=(leg,h), bbox_inches='tight')

    ############################################################# roc curves
    frames = []     # data frames to combine
    count = 0
    for f in glob(run_dir + '*.roc'):
        frames.append(pd.read_csv(f,sep='\t',index_col=False))
        count = count + 1

    df = pd.concat(frames, join='outer', ignore_index=True)
    
    print('loaded',count,'roc files with results from these learners:',df['algorithm'].unique())

    h, ax = plt.subplots()
    ax.plot([0, 1],[0, 1],'--k',label='_nolegend_')
    colors = ('r','y','b','g','c','k')
    colors = plt.cm.Blues(np.linspace(0.1, 0.9, len(df['algorithm'].unique())))

    n_algs = len(df['algorithm'].unique())
    markers = ['o','v','^','<','>','8','s',
               'p','P','*','h','H','+','x','X','D','d','|','_']
    for i, (alg,df_g) in enumerate(df.groupby('algorithm')):
   
        aucs = df_g.auc.values
        seed_max = df_g.loc[df_g.auc.idxmax()]['seed']
        seed_min = df_g.loc[df_g.auc.idxmin()]['seed']
        seed_med = df_g.loc[np.abs(df_g.auc - df_g.auc.median()) == np.min(np.abs(df_g.auc - df_g.auc.median()))]['seed']
        seed_med = seed_med.iloc[0]
         
        auc = df_g.auc.median()
        # fpr = df_g['fpr'].unique()
        tprs,fprs=[],[]
        fpr_min = df_g.loc[df_g.seed == seed_min,:]['fpr']
        fpr_max = df_g.loc[df_g.seed == seed_max,:]['fpr']
        tpr_min = df_g.loc[df_g.seed == seed_min,:]['tpr']
        tpr_max = df_g.loc[df_g.seed == seed_max,:]['tpr']
        tpr_med = df_g.loc[df_g.seed == seed_med,:]['tpr']
        fpr_med = df_g.loc[df_g.seed == seed_med,:]['fpr']
 
        ax.plot(fpr_med,tpr_med, color=colors[i % n_algs], marker=markers[i], 
                linestyle='--', linewidth=1, label='{:s} (AUC = {:0.2f})'.format(alg,auc))
      
        # ax.plot(fpr_max,tpr_max, color=colors[i % n_algs],  
        #         linestyle='--', linewidth=1, label='_nolegend_', alpha=0.1)
        # ax.plot(fpr_min,tpr_min,color=colors[i % n_algs],  
        #         linestyle='--', linewidth=1, label='_nolegend_', alpha=0.1)

    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    leg = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.ylim(0,1)
    plt.xlim(0,1)
    plt.savefig(run_dir + '_'.join([ dataset, 'roc_curves.pdf']), bbox_inches='tight')

    print('done!')    

if __name__ == '__main__':
    main()
