########## IMPORTS ##########
from turtle import back
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from timeit import default_timer as timer
from AA_result_class import _OAA_result
from loading_bar_class import _loading_bar
from OAA_class import _OAA



########## ORDINAL ARCHETYPAL ANALYSIS CLASS ##########
class _RBOAA:

    def __init__(self):
        self.result_backup = {"loss": np.inf, "itteration": 0, "result": None}

    ########## HELPER FUNCTION // BACKUP ##########
    def backup_itteration(self,Xt,A_non_constraint,B_non_constraint,b_non_constraint,sigma_non_constraint,c1_non_constraint,c2,beta_regulators,start,X,n_iter,K,p,columns,with_synthetic_data,loss,i,global_sigma):
        if loss < self.result_backup["loss"]:
            result = self.get_result(Xt,A_non_constraint,B_non_constraint,b_non_constraint,sigma_non_constraint,c1_non_constraint,c2,beta_regulators,start,X,n_iter,K,p,columns,with_synthetic_data,global_sigma)
            self.result_backup["loss"] = loss
            self.result_backup["itteration"] = i
            self.result_backup["result"] = result

    ########## HELPER FUNCTION // GET BACKUP ##########
    def get_backup(self):
        return self.result_backup["result"]
    
    ########## HELPER FUNCTION // CREATE INSTANCE OF RESULT ##########
    def get_result(self,Xt,A_non_constraint,B_non_constraint,b_non_constraint,sigma_non_constraint,c1_non_constraint,c2,beta_regulators,start,X,n_iter,K,p,columns,with_synthetic_data,global_sigma):
        A_f = self._apply_constraints_AB(A_non_constraint).detach().numpy()
        B_f = self._apply_constraints_AB(B_non_constraint).detach().numpy()
        b_f = self._apply_constraints_beta(b_non_constraint,c1_non_constraint,c2,beta_regulators)
        alphas_f = self._calculate_alpha(b_f,beta_regulators)
        X_tilde_f = self._calculate_X_tilde(Xt,alphas_f).detach().numpy()
        Z_tilde_f = (self._apply_constraints_AB(B_non_constraint).detach().numpy() @ X_tilde_f)
        sigma_f = self._apply_constraints_sigma(sigma_non_constraint,global_sigma).detach().numpy()
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
                "RBOAA",
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
        m = nn.Softmax(dim=1)
        return m(A)

    ########## HELPER FUNCTION // BETAS ##########
    def _apply_constraints_beta(self,b,c1_non_constraint,c2,beta_regulators): 
        betas = torch.empty((self.N,self.p+1))
        if beta_regulators:
            betas[:, 1:self.p+1] = torch.nn.functional.softplus(c1_non_constraint) * torch.cumsum(torch.nn.functional.softmax(b.clone(),dim=1),dim=1) + c2
            return betas
        else:
            betas[:,0] = 0
            betas[:, 1:self.p+1] = torch.cumsum(torch.nn.functional.softmax(b.clone(),dim=1),dim=1)
            return betas

    ########## HELPER FUNCTION // SIGMA ##########
    def _apply_constraints_sigma(self,sigma,global_sigma):
        m = nn.Softplus()
        if global_sigma:
            return m(torch.mean(sigma).repeat(self.N,1))
        return m(sigma)

    ########## HELPER FUNCTION // ALPHA ##########
    def _calculate_alpha(self,b,beta_regulators):
        alphas = (b[:,0:self.p] + b[:,1:self.p+1]) / 2
        if beta_regulators:
            alphas[torch.gt(alphas, 1)] = 1.0
            alphas[torch.lt(alphas, 0)] = 0.0
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
    def _calculate_loss(self,Xt,X_hat,b,sigma):
        z_next = (torch.gather(b,1,Xt)-X_hat)/sigma
        z_prev = (torch.gather(b,1,Xt-1)-X_hat)/sigma
        z_next[Xt == len(b[0,:])+1] = np.inf
        z_prev[Xt == 1] = -np.inf
        P_next = torch.distributions.normal.Normal(0, 1).cdf(z_next)
        P_prev = torch.distributions.normal.Normal(0, 1).cdf(z_prev)
        neg_logP = -torch.log(( P_next - P_prev ) +1e-10)
        loss = torch.sum(neg_logP)
        return loss

    ########## HELPER FUNCTION // ERROR ##########
    def _error(self,Xt,A_non_constraint,B_non_constraint,b_non_constraint,sigma_non_constraint,c1_non_constraint,c2,global_sigma,beta_regulators):
        A = self._apply_constraints_AB(A_non_constraint)
        B = self._apply_constraints_AB(B_non_constraint)
        b = self._apply_constraints_beta(b_non_constraint,c1_non_constraint,c2,beta_regulators)
        sigma = self._apply_constraints_sigma(sigma_non_constraint,global_sigma)
        alphas = self._calculate_alpha(b,beta_regulators)
        X_tilde = self._calculate_X_tilde(Xt,alphas)
        X_hat = self._calculate_X_hat(X_tilde,A,B)
        loss = self._calculate_loss(Xt,X_hat,b,sigma)
        return loss

    ########## COMPUTE ARCHETYPES FUNCTION OF OAA ##########
    def _compute_archetypes(self,
        X, K, p, n_iter, lr, mute, columns, 
        with_synthetic_data = False, 
        early_stopping = False, 
        backup_itterations = False,
        with_OAA_initialization = False, 
        for_hotstart_usage = False,
        hotstart_alternating = False,
        alternating = False,
        beta_regulators = False):


        ########## INITIALIZATION ##########
        self.N, self.M = len(X.T), len(X.T[0,:])
        Xt = torch.tensor(X.T, dtype = torch.long)
        self.N_arange = [m for m in range(self.M) for n in range(self.N)]
        self.M_arange = [n for n in range(self.N) for m in range(self.M)]
        self.p = p
        self.loss = []
        

        ########## INITIALIZATION OF OPTIMIZED VARIABLES // ALTERNATING ##########
        if alternating:
            optimizer, A_non_constraint, B_non_constraint, b_non_constraint, sigma_non_constraint, c1_non_constraint, c2 = self._compute_archetypes(X, K, p, n_iter, lr, mute, columns, early_stopping = early_stopping, backup_itterations=backup_itterations, with_OAA_initialization=with_OAA_initialization, for_hotstart_usage=True, alternating=False, hotstart_alternating=alternating, beta_regulators=beta_regulators)


        ########## INITIALIZATION OF OPTIMIZED VARIABLES // OAA INITALIZATION ##########
        elif with_OAA_initialization:
            if not mute:
                print("\nPerforming OAA for initialization of RBOAA.")
            OAA = _OAA()
            _, A_hot, B_hot, sigma_hot, b_hot, c1_hot, c2_hot = OAA._compute_archetypes(X, K, p, n_iter=n_iter, lr=lr, mute=mute, columns=columns, with_synthetic_data = with_synthetic_data, with_CAA_initialization=with_OAA_initialization, early_stopping = early_stopping, backup_itterations=backup_itterations, for_hotstart_usage=True,alternating=hotstart_alternating,beta_regulators=beta_regulators)
            A_non_constraint = A_hot.clone().detach().requires_grad_(True)
            B_non_constraint = B_hot.clone().detach().requires_grad_(True)
            b_non_constraint = b_hot.clone().detach().repeat(self.N,1).requires_grad_(True)
            c1_non_constraint = c1_hot.clone().detach().repeat(self.N,1).requires_grad_(True)
            c2 = c2_hot.clone().detach().repeat(self.N,1).requires_grad_(True)
            sigma_non_constraint = sigma_hot.clone().detach().repeat(self.N,1).requires_grad_(True)
            if for_hotstart_usage:
                if not mute:
                    print("\nPerforming RBOAA with global sigma.")
            optimizer = optim.Adam([A_non_constraint, 
                                    B_non_constraint, 
                                    b_non_constraint, 
                                    sigma_non_constraint,
                                    c1_non_constraint,
                                    c2], amsgrad = True, lr = lr)
        

        ########## INITIALIZATION OF OPTIMIZED VARIABLES // REGULAR ##########
        else:
            A_non_constraint = torch.autograd.Variable(torch.randn(self.N, K), requires_grad=True)
            B_non_constraint = torch.autograd.Variable(torch.randn(K, self.N), requires_grad=True)
            b_non_constraint = torch.autograd.Variable(torch.rand(self.N,p+1), requires_grad=True)
            c1_non_constraint = torch.autograd.Variable(torch.rand(1).repeat(self.N,1), requires_grad=True)
            c2 = torch.autograd.Variable(torch.rand(1).repeat(self.N,1), requires_grad=True)
            if for_hotstart_usage:
                if not mute:
                    print("\nPerforming RBOAA with global sigma.")
                sigma_non_constraint = torch.autograd.Variable(torch.randn(1), requires_grad=True)
            else:
                sigma_non_constraint = torch.autograd.Variable(torch.randn(1).repeat(self.N,1), requires_grad=True)
            optimizer = optim.Adam([A_non_constraint, 
                                    B_non_constraint, 
                                    b_non_constraint, 
                                    sigma_non_constraint,
                                    c1_non_constraint,
                                    c2], amsgrad = True, lr = lr)


        ########## TIME AND PRINT INITALIZATION ##########
        if not mute:
            loading_bar = _loading_bar(n_iter, "Response Bias Ordinal Archetypal Analysis")
        start = timer()

        
        ########## ANALYSIS ##########
        for i in range(n_iter):
            if not mute:
                loading_bar._update()
            optimizer.zero_grad()
            L = self._error(Xt,A_non_constraint,B_non_constraint,b_non_constraint,sigma_non_constraint,c1_non_constraint,c2,for_hotstart_usage,beta_regulators)
            self.loss.append(L.detach().numpy())
            L.backward()
            optimizer.step()


            ########## CHECKS ##########
            if i % 25 == 0:
                ########## ITTERATION BACKUP ##########
                if backup_itterations:
                    self.backup_itteration(Xt,A_non_constraint,B_non_constraint,b_non_constraint,sigma_non_constraint,c1_non_constraint,c2,beta_regulators,start,X,n_iter,K,p,columns,with_synthetic_data,L.detach().numpy(),i,for_hotstart_usage)
                ########## EARLY STOPPING ##########
                if early_stopping:
                    if i > 200 and self._early_stopping(i):
                        self.backup_itteration(Xt,A_non_constraint,B_non_constraint,b_non_constraint,sigma_non_constraint,c1_non_constraint,c2,beta_regulators,start,X,n_iter,K,p,columns,with_synthetic_data,L.detach().numpy(),i,for_hotstart_usage)
                        if not mute:
                            loading_bar._kill()
                            print("\nAnalysis ended due to early stopping.\n")
                        break
            
        ########## RETURN MATRICIES IF HOTSTART USAGE ##########
        if not for_hotstart_usage:
            ########## GET INSTANCE OF RESULT ##########
            result = self.get_backup()
            if not mute:
                result._print()
            return result
        
        ########## RETURN MATRICIES IF HOTSTART USAGE ##########
        else:
            return optimizer, A_non_constraint, B_non_constraint, b_non_constraint, sigma_non_constraint, c1_non_constraint, c2