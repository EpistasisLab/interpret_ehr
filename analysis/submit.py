import argparse
import os
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="An analyst for quick ML applications.",
                                     add_help=False)

    parser.add_argument('-icd', action='store', dest='CODES',default=
                    '780.93,733.00,530.85,530.81,493.90,333.94,327.23,288.60,278.00,275.42'  
                    ',250.00,250.40,729.1,585.9,571.8,564.1,473.9,472.0,428.9,331.0,285.9,277.7'
                    ',276.1,272.4,272.1,256.4,311',type=str, 
            help='Comma-separated list of icd-9 codes to run')
    parser.add_argument('-ml',action='store',dest='MLS', type=str, 
            default='LogisticRegression')
    parser.add_argument('--short',action='store_false',dest='LONG', default=True)
    parser.add_argument('--local',action='store_true',dest='LOCAL', default=False)
    parser.add_argument('--norare',action='store_true',dest='NORARE', default=False)
    parser.add_argument('-n_trials',action='store',dest='NTRIALS', default=1)
    parser.add_argument('-m',action='store',dest='M', default=16000)
    parser.add_argument('-cutoffs',action='store',dest='CUTOFFS', type=str, default='')
    parser.add_argument('-trials',action='store',dest='TRIALS', type=str, default='')
    parser.add_argument('-results',action='store',dest='RDIR', type=str, default='/project/moore/users/lacava/geis-ehr/results')
    parser.add_argument('-dir',action='store',dest='DIR', type=str, 
                        default='/project/moore/users/lacava/geis-ehr/phased/')
   
    args = parser.parse_args()
  
    if len(args.TRIALS)>0:
        args.NTRIALS = len(args.TRIALS.split(','))

    codes = args.CODES.split(',')
    cutoffs = args.CUTOFFS.split(',')

    q = 'moore_long' if args.LONG else 'moore_normal'

    if args.LOCAL:
        lpc_options = ''
    else:
        lpc_options = '--lsf -q {Q} -m {M} -n_jobs 1'.format(Q=q,M=args.M)
    
    for c in codes:
        for delay in cutoffs:
            if delay != '':
                dataset = args.DIR + 'icd9_' + c + '_cutoff' + delay
            else:
                dataset = args.DIR + 'icd9_' + c 

            if len(args.TRIALS)>0: 
                cmd = ('python analyze.py {DATA} -ml {ML} -n_trials {NTR} {RARE} '
                        '-trials {TR} {LPC}').format(DATA=dataset,
                                                    ML=args.MLS,
                                                    TR=args.TRIALS,
                                                    NTR=args.NTRIALS,
                                                    RARE='--norare' if args.NORARE else '',
                                                    LPC=lpc_options)
            else:
                cmd = ('python analyze.py {DATA} -ml {ML} -n_trials {NTR} {RARE} '
                        '-results {RDIR} {LPC}').format(DATA=dataset,
                                        ML=args.MLS,
                                        NTR=args.NTRIALS,
                                        RARE='--norare' if args.NORARE else '',
                                        RDIR=args.RDIR,
                                        LPC=lpc_options)
            print(cmd)
            os.system(cmd)
