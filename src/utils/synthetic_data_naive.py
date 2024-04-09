import numpy as np
import pickle


class synthetic_data_naive():
    def __init__(self,N,M,K,p):
        self.N = N # Samples
        self.M = M # Number of features
        self.K = K # Number of archetypes
        self.p = p
        self.alpha = 0.5
        self.eta = self.generate_probArc(self.alpha)
        self.columns = ["SQ"+str(i) for i in range(1, M+1)]


    def generate_probArc(self, alpha):


        self.eta = np.random.dirichlet(alpha*np.ones(self.K), size=(self.M))

        return self.eta

    def generate_data(self):

        x = np.ones([self.M,self.N])
        h = np.ones([self.K,self.N])
        tmp = np.ones([self.M,self.N])

        for j in range(self.N):
            
            h[:,j] = np.random.uniform(low=0, high=self.p, size=self.K)
            tmp[:,j]=self.eta@h[:,j]
            x[:,j] = np.round(tmp[:,j])

        x = x.astype(int)
        return x


X = synthetic_data_naive(100,50,3,5).generate_data()

with open('Data/synthetic_data_naive.pkl', 'wb') as f:
    pickle.dump(X, f)

