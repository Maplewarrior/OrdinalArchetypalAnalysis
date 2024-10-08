import numpy as np
from scipy.stats import norm
import pickle
import torch
import pdb

########## CLASS FOR CREATING SYNTHETIC DATA  ##########
class _synthetic_data:
    
    ########## INITIALIZATION - CREATES THE DATA ##########
    def __init__(self, N, M ,K, p, sigma, rb, a_param, b_param, sigma_std = 0):
        
        self.N = N
        self.M = M
        self.K = K
        self.p = p
        self.columns = ["SQ"+str(i) for i in range(1, M+1)]
        self.X, self.Z, self.Z_alpha, self.A, self.betas = self.X(N=N, M=M, K=K, p=p, sigma=sigma, rb=rb, a_param=a_param, b_param=b_param, sigma_std = sigma_std)
        
    ########## IF THERE IS RESPONSE BIAS IN THE DATA ##########
    def biasedBetas(self, N, p, b_param):
        b = np.array([b_param]*p)
        return np.random.dirichlet(b, size=N)
    
    ########## CONTSTRAINTS ON THE RESPONSE BIAS BETAS ##########
    def betaConstraintsBias(self, betas):
        N, J = betas.shape
        new_betas = np.ones((N,J+1))
        betas = np.concatenate((np.zeros((N, 1)), betas), axis=1)
        denoms = np.sum(betas,axis=1)
        
        for i in range(N):
            for j in range(J+1):
                new_betas[i,j] = np.sum(betas[i,:j+1])/denoms[i]
                
        ### perturb tail boundaries with a small value
        new_betas[:, 0] += np.random.uniform(-0.05, np.min(new_betas[:, 1]), size=(N,))
        new_betas[:, -1] += np.random.uniform(-(1-np.max(new_betas[:, -2])), 0.05, size=(N,))
        
        assert all(new_betas[:, 0] < new_betas[:, 1]), 'Ordinal scale not preserved in synthetic data!'
        assert all(new_betas[:, -1] > new_betas[:, -2]), 'Ordinal scale not preserved in synthetic data!'
        return new_betas
    
    ########## CONSTRAINTS ON THE NON RESOPNS BIAS BETAS ##########
    def betaConstraints(self, betas):
   
       new_betas = np.empty(len(betas))
       denom = sum(betas)

       for i in range(len(new_betas)):
           new_betas[i] = np.sum(betas[:i+1]) / denom
           
       return new_betas[:-1]
   
    ########## SOFTPLUS HELPER FUNCTION ##########
    def softplus(self, sigma, sigma_std):
        if sigma_std == 0:
            return np.log(1 + np.exp(sigma))
        else:
            sigmas = []
            for n in range(self.N):
                sigmas.append(np.log(1 + np.exp(sigma + np.random.uniform(-1,1,1)*sigma_std)))
            sigmasMatrix  = np.repeat(sigmas, self.M, axis=1)
            return sigmasMatrix

    ########## HELPER FUNCTION, CALCULATES THE Z ARCEHTYPE MATRIX ##########
    def get_Z(self, N, M, K, p, rb, b_param):
        # Ensure reproducibility
        np.random.seed(42)

        # Check to ensure that there are no NaN's
        if b_param < 0.01:
            b_param = 0.01
        
        if rb == True:
            betas = self.biasedBetas(N=N, p=p, b_param=b_param)
            betas = self.betaConstraintsBias(betas)
            mu_betas = betas.mean(axis=0)
            alphas = (mu_betas[1:] + mu_betas[:-1]) / 2


        else:
            betas = np.array([b_param]*p)
            betas = self.betaConstraints(betas)
            
            alphas = np.empty(p)
            alphas[0] = (0 + betas[0]) / 2
            alphas[-1] = (1+ betas[-1]) / 2
            for i in range(len(betas)-1):
                alphas[i+1] = (betas[i] + betas[i+1]) / 2
        
        Z_ordinal = np.ceil(np.random.uniform(0, 1, size = (M,K))*p).astype(int)
        Z_alpha = alphas[Z_ordinal-1]
        
        return Z_ordinal, Z_alpha, betas

    ########## HELPER FUNCTION, CALCULATES THE A - LINEAR COMBINATION MATRIX ##########
    def get_A(self, N, K, a_param):
        np.random.seed(42) # set another seed :)
        
        # Constrain a_param to avoid NaN's
        if a_param < 0.01:
            a_param = 0.01
        
        alpha = np.array([a_param]*K)
        return np.random.dirichlet(alpha, size=N).transpose()
    
    ########## HELPER FUNCTION, CALCULATES THE D DENSITY MATRIX ##########
    def get_D(self, X_rec, betas, sigma, rb):
        M, N = X_rec.shape
        
        if rb == False:
        
            J = len(betas)    
            D = np.empty((J+2, M, N))
            
            for j in range(J+2):
                # Left-most tail
                if j == 0:
                    D[j] = np.ones((M,N))*(np.inf*(-1))
                # Right-most tail
                elif j == J+1:
                    D[j] = np.ones((M,N))*(np.inf)
                else:
                    D[j] = (betas[j-1] - X_rec)/(sigma.T+1e-16) ## Add softplus(sigma)
                    
        else: # betas are already padded
            J = len(betas[0,:])
            D = np.empty((J, M, N))
            
            # D = torch.rand(len(betas[0,:])+2,M,N)
            # D[0] = torch.tensor(np.matrix(np.ones((N)) * (-np.inf)))
            # D[-1] = torch.tensor(np.matrix(np.ones((N)) * (np.inf)))
            # D[1:-1] = torch.div(torch.unsqueeze(betas.T, 2).repeat(1,1,N)-X_rec.T,torch.unsqueeze(sigma+1e-16, 1).repeat(1,N))
            
            for j in range(J):
                if j == 0:
                    D[j] = np.ones((M,N))*(np.inf*(-1))
                elif j == J-1:
                    D[j] = np.ones((M,N))*(np.inf)
                else:
                    D[j] = (betas[:,j] - X_rec)/((sigma.T+1e-16)) ## Add softplus(sigma)
                    # D[j] = torch.div((b[:,j-1] - X_hat[:, None]),sigma)[:,0,:].T
        
        return D - np.mean(D[1:-1])

    ########## HELPER FUNCTION, CALCULATES THE PROBABILITY FROM THE DENSITY MATRIX ##########
    def Probs(self, D):
        
        J, M, N = D.shape
        
        probs = np.empty((J-1, M, N)) 
        for i in range(J):
            if i != J-1:
                probs[i,:,:] = norm.cdf(D[i+1], loc=0, scale=1) - norm.cdf(D[i], loc=0, scale=1)
                
        return probs

    ########## HELPER FUNCTION, SAMPLES FROM PROBABILITY MATRIX TO GET CATEGORICAL ##########
    def toCategorical(self, probs):
        
        categories = np.arange(1, len(probs)+1)    
        J, M, N = probs.shape
        X_cat = np.empty((M,N))
        
        
        for m in range(M):
            for n in range(N):
                X_cat[m,n] = int(np.random.choice(categories, p = list(probs[:,m,n])))
                
        X_cat = X_cat.astype(int)
        return X_cat
    
    ########## CALUCLATES DATA WITH HELP OF ALL OTHER FUNCTIONS ##########
    def X(self, M, N, K, p, sigma, rb=False, a_param=1, b_param=100, sigma_std = 0):
        
        Z_ordinal, Z_alpha, betas = self.get_Z(N=N,M=M, K=K, p=p, rb=rb, b_param=b_param)
        A = self.get_A(N, K, a_param=a_param)
        X_rec = Z_alpha@A
        
        D = self.get_D(X_rec, betas, self.softplus(sigma, sigma_std), rb=rb)
    
        probs = self.Probs(D)
        
        X_final = self.toCategorical(probs)
        
        
        return X_final, Z_ordinal, Z_alpha, A, betas

    ########## SAVES THE DATA LOCALLY ON PC ########## 
    def _save(self,type,filename):
        file = open("synthetic_results/" + type + "_" + filename + '_metadata' + '.obj','wb')
        pickle.dump(self, file)
        file.close()
