import numpy as np
from sklearn.utils import check_random_state

class QuartileExactMatch():
    """Matches cases by sampling controls from the same quartile bins as each control sample or by matching exactly on given feature indices."""
    def __init__(self, quart_locs, exact_locs, random_state=None):
        # features to match on
        self.quart_locs = quart_locs
        self.exact_locs = exact_locs
        if random_state:
            self.random_state = check_random_state(random_state)

    def fit(self, features, labels):
        """Calculates and bins quartiles"""
        quarts = np.zeros((len(self.quart_locs),3))
        for i in self.quart_locs:
            quarts[i] = [np.percentile(features[:,i],25),
                         np.percentile(features[:,i],50),
                         np.percentile(features[:,i],75),
                        ]
        
        # bin samples according to quartiles
        self.quart_bin = np.zeros((len(features),len(self.quart_locs)))

        for i,f in enumerate(features):
            for j,q in enumerate(quarts):
                if f[self.quart_locs[j]] <= q[0]:
                    self.quart_bin[i,j] = 1
                elif f[self.quart_locs[j]] <= q[1]:
                    self.quart_bin[i,j] = 2
                elif f[self.quart_locs[j]] <= q[2]:
                    self.quart_bin[i,j] = 3
                else:
                    self.quart_bin[i,j] = 4
                    
#         print('quart_bin:',self.quart_bin,self.quart_bin.shape)
    def sample(self, features, labels): 
#         print('labels:',labels)
#         print('features:',features)
        iscase = labels == 1
        cases = np.where(iscase)[0]
        controls = []
#         print('cases:',cases)
        for c in cases:
            # matches are where quartile bins match the bins for sample c
            quart_match = np.array(self.quart_bin[:] == self.quart_bin[c,:]).all(axis=1)
            exact_match = np.array([features[:,j] == features[c,j] 
                                    for j in self.exact_locs]).all(axis=0)
#             print('c:',c) 
#             print('quart_bin[c]:',self.quart_bin[c])
#             print('exact[c]:',features[c,self.exact_locs[0]])
            
#             print('~iscase:',~iscase)
#             print('quartile match:',quart_match)
#             print('exact match:',exact_match)
                        
            matches = np.where(np.array(~iscase & 
                               quart_match &
                               exact_match
                               ))[0]
#             print('matches:',matches)
            if len(matches)>0:
                # pick randomly from matching bins
                controls.append(self.random_state.choice(matches,replace=False))
            else:
                # if there are no matches, just pick a random control
                controls.append(self.random_state.choice(np.where(~iscase)[0]))
#             print('controls:',controls)
        assert(len(controls) == len(cases))
        assert((controls != cases).all())
        if len(controls) > len(np.unique(controls)):
            print("WARNING: controls are not unique")

        X_sample = np.vstack((features[cases],features[controls]))
        y_sample = np.vstack((labels[cases],labels[controls])).flatten()
        idx_sample = np.hstack((cases,controls))
#         print('X_sample:',X)
#         print('y_sample:',y_sample)
#         print('idx_sample:',idx_sample)
        return X_sample, y_sample, idx_sample

    def fit_sample(self, features, labels):
        self.fit(features,labels)
        return self.sample(features, labels)

