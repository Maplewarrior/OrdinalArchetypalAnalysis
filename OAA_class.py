########## IMPORTS ##########
from re import T
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from timeit import default_timer as timer
from AA_result_class import _OAA_result
from loading_bar_class import _loading_bar
from CAA_class import _CAA



########## ORDINAL ARCHETYPAL ANALYSIS CLASS ##########
class _OAA:

    ########## INITIALIZE NN HELPER FUNCTIONS ##########
    def __init__(self):
        self.softplus = nn.Softplus()
        self.softmax_dim0 = nn.Softmax(dim=0)
        self.softmax_dim1 = nn.Softmax(dim=1)
        self.constantPad = nn.ConstantPad1d(1, 0)
        self.result_backup = {"loss": np.inf, "itteration": 0, "result": None}

    ########## HELPER FUNCTION // BACKUP ##########
    def backup_itteration(self,Xt,A_non_constraint,B_non_constraint,b_non_constraint,sigma_non_constraint,c1_non_constraint,c2,beta_regulators,start,lr,X,n_iter,K,p,columns,with_synthetic_data,sigma_cap,loss,i):
        if loss < self.result_backup["loss"]:
            result = self.get_result(Xt,A_non_constraint,B_non_constraint,b_non_constraint,sigma_non_constraint,c1_non_constraint,c2,beta_regulators,start,X,n_iter,K,p,columns,with_synthetic_data,sigma_cap)
            self.result_backup["loss"] = loss
            self.result_backup["itteration"] = i
            self.result_backup["result"] = result
    
    ########## HELPER FUNCTION // INFORCE BACKUP ##########
    def get_backup(self):
        return self.result_backup["result"]

    ########## HELPER FUNCTION // CREATE INSTANCE OF RESULT ##########
    def get_result(self,Xt,A_non_constraint,B_non_constraint,b_non_constraint,sigma_non_constraint,c1_non_constraint,c2,beta_regulators,start,X,n_iter,K,p,columns,with_synthetic_data,sigma_cap):
        A_f = self._apply_constraints_AB(A_non_constraint).detach().numpy()
        B_f = self._apply_constraints_AB(B_non_constraint).detach().numpy()
        b_f = self._apply_constraints_beta(b_non_constraint,c1_non_constraint,c2,beta_regulators)
        alphas_f = self._calculate_alpha(b_f,beta_regulators)
        X_tilde_f = self._calculate_X_tilde(Xt,alphas_f).detach().numpy()
        Z_tilde_f = (self._apply_constraints_AB(B_non_constraint).detach().numpy() @ X_tilde_f)
        sigma_f = self._apply_constraints_sigma(sigma_non_constraint, sigma_cap).detach().numpy()
        X_hat_f = self._calculate_X_hat(X_tilde_f,A_f,B_f)
        end = timer()
        time = round(end-start,2)
        Z_f = B_f @ X_tilde_f
        result = _OAA_result(
            A_f.T,
            B_f.T,
            X,
            n_iter,
            b_f.detach().numpy(),
            Z_f.T,
            X_tilde_f.T,
            Z_tilde_f.T,
            X_hat_f.T,
            np.copy(self.loss),
            K,
            p,
            time,
            columns,
            "OAA",
            sigma_f,
            with_synthetic_data=with_synthetic_data)
        return result

    ########## HELPER FUNCTION // EARLY STOPPING ##########
    def _early_stopping(self,i):
        last_avg = np.mean(self.loss[-200:-100])
        current_avg = np.mean(self.loss[-100:])
        total_imp = (self.loss[-(i-1)]-self.loss[-1])
        return (last_avg-current_avg) < total_imp*1e-5

    ########## HELPER FUNCTION // A AND B ##########
    def _apply_constraints_AB(self,A):
        return self.softmax_dim1(A)

    ########## HELPER FUNCTION // BETAS ##########
    def _apply_constraints_beta(self,b,c1_non_constraint,c2,beta_regulators): 
        if beta_regulators:
            return self.softplus(c1_non_constraint) * torch.cumsum(self.softmax_dim0(b), dim=0)[:len(b)] + c2
        else:
            return torch.cumsum(self.softmax_dim0(b), dim=0)[:len(b)]

    ########## HELPER FUNCTION // SIGMA ##########
    def _apply_constraints_sigma(self,sigma,sigma_cap):
        if sigma_cap:
            return self.softplus(sigma.clamp(min=-9.21, max=-9.21))
        return self.softplus(sigma)

    ########## HELPER FUNCTION // ALPHA ##########
    def _calculate_alpha(self,b,beta_regulators):
        b_j = torch.cat((torch.tensor([0.0]),b),0)
        b_j_plus1 = torch.cat((b,torch.tensor([1.0])),0)
        alphas = (b_j_plus1+b_j)/2
        if beta_regulators:
            alphas[torch.gt(alphas, 1)] = 1.0
            alphas[torch.lt(alphas, 0)] = 0.0
        return alphas

    ########## HELPER FUNCTION // X_tilde ##########
    def _calculate_X_tilde(self,X,alphas):
        X_tilde = alphas[X.long()-1]
        return X_tilde

    ########## HELPER FUNCTION // X_hat ##########
    def _calculate_X_hat(self,X_tilde,A,B):
        Z = B @ X_tilde
        X_hat = A @ Z
        return X_hat
    
    ########## HELPER FUNCTION // LOSS ##########
    def _calculate_loss(self,Xt, X_hat, b, sigma):
        b = self.constantPad(b)
        b[-1] = 1.0
        z_next = (b[Xt] - X_hat)/sigma
        z_prev = (b[Xt-1] - X_hat)/sigma
        z_next[Xt == len(b)+1] = np.inf
        z_prev[Xt == 1] = -np.inf
        P_next = torch.distributions.normal.Normal(0, 1).cdf(z_next)
        P_prev = torch.distributions.normal.Normal(0, 1).cdf(z_prev)
        neg_logP = -torch.log(( P_next - P_prev ) +1e-10)
        loss = torch.sum(neg_logP)
        return loss

    ########## HELPER FUNCTION // ERROR ##########
    def _error(self,Xt,A_non_constraint,B_non_constraint,b_non_constraint,sigma_non_constraint,c1_non_constraint,c2,sigma_cap,beta_regulators):
        A = self._apply_constraints_AB(A_non_constraint)
        B = self._apply_constraints_AB(B_non_constraint)
        b = self._apply_constraints_beta(b_non_constraint,c1_non_constraint,c2,beta_regulators)
        sigma = self._apply_constraints_sigma(sigma_non_constraint,sigma_cap)
        alphas = self._calculate_alpha(b,beta_regulators)
        X_tilde = self._calculate_X_tilde(Xt,alphas)
        X_hat = self._calculate_X_hat(X_tilde,A,B)
        loss = self._calculate_loss(Xt, X_hat, b, sigma)
        return loss
        

    ########## COMPUTE ARCHETYPES FUNCTION OF OAA ##########
    def _compute_archetypes(
        self, 
        X, 
        K, 
        p, 
        n_iter, 
        lr, 
        mute, 
        columns, 
        with_CAA_initialization = False,
        with_synthetic_data = False, 
        early_stopping = False, 
        backup_itterations = False,
        for_hotstart_usage = False,
        sigma_cap = False,
        beta_regulators = False,
        alternating = False):


        ########## INITIALIZATION OF GENERAL VARIABLES ##########
        self.N, self.M = len(X.T), len(X.T[0,:])
        Xt = torch.tensor(X.T, dtype = torch.long)
        self.loss = []


        ########## INITIALIZATION OF OPTIMIZED VARIABLES ##########
        ########## ALTERNATING ##########
        if alternating:
            if not mute:
                print("\nPerforming alternating analysis with sigma cap.")
            optimizer, A_non_constraint, B_non_constraint, sigma_non_constraint, b_non_constraint, c1_non_constraint, c2 = self._compute_archetypes(X, 
                                                                                                                                                    K, 
                                                                                                                                                    p, 
                                                                                                                                                    n_iter=n_iter, 
                                                                                                                                                    lr=lr, 
                                                                                                                                                    mute=mute, 
                                                                                                                                                    columns=columns, 
                                                                                                                                                    with_CAA_initialization=with_CAA_initialization,
                                                                                                                                                    with_synthetic_data=with_synthetic_data, 
                                                                                                                                                    early_stopping=early_stopping, 
                                                                                                                                                    for_hotstart_usage=True, 
                                                                                                                                                    sigma_cap=True, 
                                                                                                                                                    beta_regulators=beta_regulators, 
                                                                                                                                                    alternating=False, 
                                                                                                                                                    backup_itterations=backup_itterations)
            sigma_non_constraint.requires_grad_(True)
            

        ########## NON ALTERNATING ##########
        else:
            if with_CAA_initialization:
                CAA = _CAA()
                A_non_constraint_np, B_non_constraint_np = CAA._compute_archetypes(X=X,K=K,p=p,n_iter=n_iter,lr=lr,mute=mute,columns=columns,with_synthetic_data=with_synthetic_data,early_stopping=early_stopping,for_hotstart_usage=True)
                A_non_constraint = torch.autograd.Variable(torch.tensor(A_non_constraint_np.T), requires_grad=True)
                B_non_constraint = torch.autograd.Variable(torch.tensor(B_non_constraint_np.T), requires_grad=True)
            else:
                A_non_constraint = torch.autograd.Variable(torch.randn(self.N, K), requires_grad=True)
                B_non_constraint = torch.autograd.Variable(torch.randn(K, self.N), requires_grad=True)
            b_non_constraint = torch.autograd.Variable(torch.rand(p), requires_grad=True)
            if sigma_cap:
                sigma_non_constraint = torch.tensor(-9.21, requires_grad=False)
            else:
                sigma_non_constraint = torch.autograd.Variable(torch.rand(1), requires_grad=True)
            c1_non_constraint = torch.autograd.Variable(torch.rand(1), requires_grad=True)
            c2 = torch.autograd.Variable(torch.rand(1), requires_grad=True)
            optimizer = optim.Adam([A_non_constraint, 
                                    B_non_constraint, 
                                    b_non_constraint, 
                                    sigma_non_constraint,
                                    c1_non_constraint,
                                    c2], amsgrad = True, lr = lr)
        
        
        ########## TIMER AND PRINT INITIALIZATION ##########
        if not mute:
            loading_bar = _loading_bar(n_iter, "Ordinal Archetypal Analysis")
        start = timer()

        
        ########## ANALYSIS ##########
        for i in range(n_iter):
            if not mute:
                loading_bar._update()
            optimizer.zero_grad()
            L = self._error(Xt,A_non_constraint,B_non_constraint,b_non_constraint,sigma_non_constraint,c1_non_constraint,c2,sigma_cap,beta_regulators)
            self.loss.append(L.detach().numpy())
            L.backward()
            optimizer.step()


            ########## CHECKS ##########
            if i % 25 == 0:
                ########## ITTERATION BACKUP ##########
                if backup_itterations:
                    self.backup_itteration(Xt,A_non_constraint,B_non_constraint,b_non_constraint,sigma_non_constraint,c1_non_constraint,c2,beta_regulators,start,lr,X,n_iter,K,p,columns,with_synthetic_data,sigma_cap,L.detach().numpy(),i)
                ########## EARLY STOPPING ##########
                if early_stopping:
                    if i > 200 and self._early_stopping(i):
                        if not mute:
                            loading_bar._kill()
                            print("Analysis ended due to early stopping.\n")
                        break
        
        ########## CREATE FINAL VERSION OF RESULT ##########
        self.backup_itteration(Xt,A_non_constraint,B_non_constraint,b_non_constraint,sigma_non_constraint,c1_non_constraint,c2,beta_regulators,start,lr,X,n_iter,K,p,columns,with_synthetic_data,sigma_cap,L.detach().numpy(),i)
        ########## RETURN MATRICIES IF HOTSTART USAGE ##########
        if not for_hotstart_usage:
            ########## GET INSTANCE OF RESULT ##########
            result = self.get_backup()
            if not mute:
                result._print()
            return result
        

        ########## RETURN MATRICIES IF HOTSTART USAGE ##########
        else:
            return optimizer, A_non_constraint, B_non_constraint, sigma_non_constraint, b_non_constraint, c1_non_constraint, c2

