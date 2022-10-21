########## IMPORTS ##########
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from timeit import default_timer as timer
from AA_result_class import _OAA_result
from loading_bar_class import _loading_bar
from OAA_class_adj import _OAA



########## ORDINAL ARCHETYPAL ANALYSIS CLASS ##########
class _RBOAA:

    ########## HELPER FUNCTION // EARLY STOPPING ##########
    def _early_stopping(self):
        next_imp = self.loss[-round(len(self.loss)/100)]-self.loss[-1]
        prev_imp = (self.loss[0]-self.loss[-1])*1e-5
        return next_imp < prev_imp
    
    ########## HELPER FUNCTION // A AND B ##########
    def _apply_constraints_AB(self,A):
        m = nn.Softmax(dim=1)
        return m(A)

    ########## HELPER FUNCTION // BETAS ##########
    def _apply_constraints_beta(self,b, c1, c2): 

        betas = torch.empty((self.N,self.p+1))
        betas[:, 0:self.p+1] = self._softplus(c1) * torch.cumsum(torch.nn.functional.softmax(b.clone(),dim=1),dim=1) + c2
        return betas

    ########## HELPER FUNCTION // SIGMA ##########
    def _apply_constraints_sigma(self,sigma):
        m = nn.Softplus()
        return m(sigma)
    
    def _softplus(self, s):
        m = nn.Softplus()
        return m(s)
    

    ########## HELPER FUNCTION // ALPHA ##########
    def _calculate_alpha(self,b):
        alphas = (b[:,0:self.p] + b[:,1:self.p+1]) / 2
        return alphas

    ########## HELPER FUNCTION // X_tilde ##########
    def _calculate_X_tilde(self,X,alphas):
        X_tilde = torch.gather(alphas,1,X-1)
        return X_tilde

    ########## HELPER FUNCTION // X_hat ##########
    def _calculate_X_hat(self,X_tilde,A,B):
        Z = B @ X_tilde
        X_hat = A @ Z
        return X_hat
    
    ########## HELPER FUNCTION // LOSS ##########
    def _calculate_loss(self,Xt, X_hat, b, sigma, global_sigma):
        
        if global_sigma:
            sigma = sigma.expand(self.N, 1)
        

        z_next = (torch.gather(b,1,Xt)-X_hat)/sigma
        z_prev = (torch.gather(b,1,Xt-1)-X_hat)/sigma
        z_next[Xt == len(b[0,:])+1] = np.inf
        z_prev[Xt == 1] = -np.inf

        P_next = torch.distributions.normal.Normal(0, 1).cdf(z_next)
        P_prev = torch.distributions.normal.Normal(0, 1).cdf(z_prev)
        neg_logP = -torch.log(( P_next - P_prev ) + 1e-10)
        loss = torch.sum(neg_logP)

        return loss

    ########## HELPER FUNCTION // ERROR ##########
    def _error(self,Xt,A_non_constraint,B_non_constraint,b_non_constraint,sigma_non_constraint, c1_non_constraint, c2, global_sigma):

        A = self._apply_constraints_AB(A_non_constraint)
        B = self._apply_constraints_AB(B_non_constraint)
        b = self._apply_constraints_beta(b_non_constraint, c1_non_constraint, c2)
        sigma = self._apply_constraints_sigma(sigma_non_constraint)
        alphas = self._calculate_alpha(b)

        X_tilde = self._calculate_X_tilde(Xt,alphas)
        X_hat = self._calculate_X_hat(X_tilde,A,B)

        loss = self._calculate_loss(Xt, X_hat, b, sigma, global_sigma)
        
        return loss

    ########## COMPUTE ARCHETYPES FUNCTION OF OAA ##########
    def _compute_archetypes(self,
        X, K, p, n_iter, lr, mute, columns, 
        with_synthetic_data = False, 
        early_stopping = False, 
        with_OAA_initialization = False, for_hotstart_usage = False):


        ########## INITIALIZATION ##########
        self.N, self.M = len(X.T), len(X.T[0,:])
        Xt = torch.tensor(X.T, dtype = torch.long)
        self.N_arange = [m for m in range(self.M) for n in range(self.N)]
        self.M_arange = [n for n in range(self.N) for m in range(self.M)]
        self.p = p
        self.loss = []
        start = timer()
        

        if with_OAA_initialization:
            # if not mute:
            #     print("\nPerforming OAA for initialization of RBOAA.")
            OAA = _OAA()
            A_hot, B_hot, sigma_hot, b_hot, c1_hot, c2_hot = OAA._compute_archetypes_alternating(X, K, p, n_iter=3000, lr=0.05, mute=mute, columns=columns, with_synthetic_data = with_synthetic_data, early_stopping = early_stopping, for_hotstart_usage=True)
            A_non_constraint = torch.autograd.Variable(torch.tensor(A_hot), requires_grad=True)
            B_non_constraint = torch.autograd.Variable(torch.tensor(B_hot), requires_grad=True)
            # sigma_non_constraint = torch.autograd.Variable(torch.tensor(sigma_hot).repeat(self.N,1), requires_grad=True)
            b_non_constraint = torch.autograd.Variable(torch.tensor(b_hot).repeat(self.N,1), requires_grad=True)
            c1_non_constraint = torch.autograd.Variable(torch.tensor(c1_hot), requires_grad=True)
            c2 = torch.autograd.Variable(torch.tensor(c2_hot), requires_grad=True)
            
            # Optimize on a single global sigma if True
            if for_hotstart_usage:
                sigma_non_constraint = torch.autograd.Variable(torch.tensor(sigma_hot), requires_grad=True)
            # Optimize on one sigma per subject
            else:
                sigma_non_constraint = torch.autograd.Variable(torch.tensor(sigma_hot).repeat(self.N,1), requires_grad=True)
        else:
            A_non_constraint = torch.autograd.Variable(torch.randn(self.N, K), requires_grad=True)
            B_non_constraint = torch.autograd.Variable(torch.randn(K, self.N), requires_grad=True)
            sigma_non_constraint = torch.autograd.Variable(torch.randn(1).repeat(self.N,1), requires_grad=True)
            b_non_constraint = torch.autograd.Variable(torch.rand(self.N,p+1), requires_grad=True)
            c1_non_constraint = torch.autograd.Variable(torch.rand(1), requires_grad=True)
            c2 = torch.autograd.Variable(torch.rand(1), requires_grad=True)

        optimizer = optim.Adam([A_non_constraint, 
                                B_non_constraint, 
                                b_non_constraint, 
                                sigma_non_constraint,
                                c1_non_constraint,
                                c2], amsgrad = True, lr = lr)

        if not mute:
            loading_bar = _loading_bar(n_iter, "Response Bias Ordinal Archetypal Analysis")

        
        ########## ANALYSIS ##########
        for i in range(n_iter):
            if not mute:
                loading_bar._update()
            optimizer.zero_grad()
            L = self._error(Xt,A_non_constraint,B_non_constraint,b_non_constraint,sigma_non_constraint, c1_non_constraint, c2, for_hotstart_usage)
            self.loss.append(L.detach().numpy())
            L.backward()
            optimizer.step()


            ########## EARLY STOPPING ##########
            if i % 25 == 0 and early_stopping:
                if len(self.loss) > 200 and self._early_stopping():
                    if not mute:
                        loading_bar._kill()
                        print("Analysis ended due to early stopping.\n")
                    break
            
        ########## POST ANALYSIS ##########
        A_f = self._apply_constraints_AB(A_non_constraint).detach().numpy()
        B_f = self._apply_constraints_AB(B_non_constraint).detach().numpy()
        b_f = self._apply_constraints_beta(b_non_constraint, c1_non_constraint, c2)
        alphas_f = self._calculate_alpha(b_f)
        X_tilde_f = self._calculate_X_tilde(Xt,alphas_f).detach().numpy()
        Z_tilde_f = (self._apply_constraints_AB(B_non_constraint).detach().numpy() @ X_tilde_f)
        sigma_f = self._apply_constraints_sigma(sigma_non_constraint).detach().numpy()
        X_hat_f = self._calculate_X_hat(X_tilde_f,A_f,B_f)
        end = timer()
        time = round(end-start,2)
        Z_f = B_f @ X_tilde_f
        
        c1_f = self._softplus(c1_non_constraint).detach().numpy()
        c2_f = c2.detach().numpy()


        ########## CREATE RESULT INSTANCE ##########
        result = _OAA_result(
            A_f.T,
            B_f.T,
            X,
            n_iter,
            b_f.detach().numpy()[:,1:-1],
            Z_f.T,
            X_tilde_f.T,
            Z_tilde_f.T,
            X_hat_f.T,
            self.loss,
            K,
            p,
            time,
            columns,
            "RBOAA",
            sigma_f,
            with_synthetic_data=with_synthetic_data)

        if not mute:
            result._print()
        
        if for_hotstart_usage:
            A_np = A_non_constraint.detach().numpy()
            B_np = B_non_constraint.detach().numpy()
            b_np = b_non_constraint.detach().numpy()
            sigma_np = sigma_non_constraint.detach().numpy()
            c1_np = c1_non_constraint.detach().numpy()
            c2_np = c2.detach().numpy()
            return A_np, B_np, b_np, sigma_np, c1_np, c2_np
            
        else:
        
            return result
    
     ########## COMPUTE ARCHETYPES FUNCTION OF OAA ##########
    def _compute_archetypes_alternating(self,
        X, K, p, n_iter, lr, mute, columns, 
        with_synthetic_data = False, 
        early_stopping = False, 
        with_OAA_initialization = False,
        ):


        ########## INITIALIZATION ##########
        self.N, self.M = len(X.T), len(X.T[0,:])
        Xt = torch.tensor(X.T, dtype = torch.long)
        self.N_arange = [m for m in range(self.M) for n in range(self.N)]
        self.M_arange = [n for n in range(self.N) for m in range(self.M)]
        self.p = p
        self.loss = []
        start = timer()

        #### CHANGED TO ALSO INCLUDE SIGMA GLOBAL/LOCAL ####
        if with_OAA_initialization:
            if not mute:
                print("\n Performing initialization...")
            
            A_hot, B_hot, b_hot, sigma_hot, c1_hot, c2_hot = self._compute_archetypes(X, K, p, n_iter, lr, mute, columns, early_stopping = True, with_OAA_initialization=True, for_hotstart_usage=True)
            #self(X, K, p, n_iter, 0.01, mute, columns, with_synthetic_data = with_synthetic_data, early_stopping = early_stopping, for_hotstart_usage=True)
            
            print("----------------------------------------------------")
            print("FINAL VARIABLE INIT IN PROGRESS")
            A_non_constraint = torch.autograd.Variable(torch.tensor(A_hot), requires_grad=True)
            B_non_constraint = torch.autograd.Variable(torch.tensor(B_hot), requires_grad=True)
            b_non_constraint = torch.autograd.Variable(torch.tensor(b_hot), requires_grad=True)
            sigma_non_constraint = torch.autograd.Variable(torch.tensor(sigma_hot).repeat(self.N, 1), requires_grad=True)
            c1_non_constraint = torch.autograd.Variable(torch.tensor(c1_hot), requires_grad=True)
            c2 = torch.autograd.Variable(torch.tensor(c2_hot), requires_grad=True)
            
            # if global_sigma:
            #     sigma_non_constraint = torch.autograd.Variable(torch.tensor(sigma_hot), requires_grad=True)
            # else:
            #     sigma_non_constraint = torch.autograd.Variable(torch.tensor(sigma_hot).repeat(self.N,1), requires_grad=True)
            
        else:
            A_non_constraint = torch.autograd.Variable(torch.randn(self.N, K), requires_grad=True)
            B_non_constraint = torch.autograd.Variable(torch.randn(K, self.N), requires_grad=True)
            sigma_non_constraint = torch.autograd.Variable(torch.randn(1).repeat(self.N,1), requires_grad=True)
            b_non_constraint = torch.autograd.Variable(torch.rand(self.N,p+1), requires_grad=True)
            c1_non_constraint = torch.autograd.Variable(torch.rand(1), requires_grad=True)
            c2 = torch.autograd.Variable(torch.rand(1), requires_grad=True)

        optimizer = optim.Adam([A_non_constraint, 
                                B_non_constraint, 
                                b_non_constraint, 
                                sigma_non_constraint,
                                c1_non_constraint,
                                c2], amsgrad = True, lr = lr)

        if not mute:
            loading_bar = _loading_bar(n_iter, "Response Bias Ordinal Archetypal Analysis")

        
        ########## ANALYSIS ##########
        print("SUCCESFULL VARIABLE INITIALIZATION")
        for i in range(n_iter):
            if not mute:
                loading_bar._update()
            optimizer.zero_grad()
            L = self._error(Xt,A_non_constraint.clone(),B_non_constraint, b_non_constraint,sigma_non_constraint, c1_non_constraint, c2, global_sigma=False)
            self.loss.append(L.detach().numpy())
            L.backward()
            optimizer.step()


            ########## EARLY STOPPING ##########
            if i % 25 == 0 and early_stopping:
                if len(self.loss) > 200 and self._early_stopping():
                    if not mute:
                        loading_bar._kill()
                        print("Analysis ended due to early stopping.\n")
                    break
            
        ########## POST ANALYSIS ##########
        A_f = self._apply_constraints_AB(A_non_constraint).detach().numpy()
        B_f = self._apply_constraints_AB(B_non_constraint).detach().numpy()
        b_f = self._apply_constraints_beta(b_non_constraint, c1_non_constraint, c2)
        alphas_f = self._calculate_alpha(b_f)
        X_tilde_f = self._calculate_X_tilde(Xt,alphas_f).detach().numpy()
        Z_tilde_f = (self._apply_constraints_AB(B_non_constraint).detach().numpy() @ X_tilde_f)
        sigma_f = self._apply_constraints_sigma(sigma_non_constraint).detach().numpy()
        X_hat_f = self._calculate_X_hat(X_tilde_f,A_f,B_f)
        end = timer()
        time = round(end-start,2)
        Z_f = B_f @ X_tilde_f


        ########## CREATE RESULT INSTANCE ##########
        result = _OAA_result(
            A_f.T,
            B_f.T,
            X,
            n_iter,
            b_f.detach().numpy()[:,1:-1],
            Z_f.T,
            X_tilde_f.T,
            Z_tilde_f.T,
            X_hat_f.T,
            self.loss,
            K,
            p,
            time,
            columns,
            "RBOAA",
            sigma_f,
            with_synthetic_data=with_synthetic_data)

        if not mute:
            result._print()
        
        return result
