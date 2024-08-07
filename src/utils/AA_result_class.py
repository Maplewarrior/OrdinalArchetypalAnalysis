
########## CONVENTIONAL ARCHETYPAL ANALYSIS RESULT ##########
class _CAA_result:
    
    def __init__(self, A, B, X, X_hat, n_iter, RSS, Z, K, p, time, columns,type, with_synthetic_data = False):
        self.A = A
        self.B = B
        self.X = X
        self.X_hat  = X_hat
        self.n_iter = len(RSS)
        self.loss = RSS
        self.Z = Z
        self.K = K
        self.p = p
        self.time = time
        self.columns = columns
        self.type = type
        self.with_synthetic_data = with_synthetic_data
        self.N = len(self.X[0,:])

    def _print(self):
        if self.type == "CAA":
            type_name = "Conventional Archetypal"
        else:
            type_name = "Two Step Archetypal"
        print("/////////////// INFORMATION ABOUT " + type_name.upper() + " ANALYSIS \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\")
        print(f"▣ The " + type_name + " Analysis was computed using " + str(self.K) + " archetypes.")
        print(f"▣ The " + type_name + " Analysis was computed on " + str(len(self.X)) + " attributes.")
        print(f"▣ The " + type_name + " Analysis was computed on " + str(self.N) + " subjects.")
        print(f"▣ The " + type_name + " Analysis ran for " + str(self.n_iter) + " iterations.")
        print(f"▣ The " + type_name + " Analysis took " + str(self.time) + " seconds to complete.")
        print(f"▣ The final RSS was: {self.loss[-1]}.")

########## ORDINAL ARCHETYPAL ANALYSIS RESULT ##########
class _OAA_result:

    def __init__(self, A, B, X, n_iter, b, Z, X_tilde, Z_tilde, X_hat, loss, K, p, time, columns,type,sigma, with_synthetic_data = False):
        self.A = A
        self.B = B
        self.X = X
        self.n_iter = len(loss)
        self.b = b
        self.sigma = sigma
        self.X_tilde = X_tilde
        self.Z_tilde = Z_tilde
        self.X_hat = X_hat
        self.loss = loss
        self.Z = Z
        self.K = K
        self.p = p
        self.time = time
        self.columns = columns
        self.type = type
        self.with_synthetic_data = with_synthetic_data
        self.N = len(self.X[0,:])

    def _print(self):
        if self.type == "RBOAA":
            type_name = "Response Bias Ordinal Archetypal"
        else:
            type_name = "Ordinal Archetypal"
        
        print("/////////////// INFORMATION ABOUT " + type_name.upper() + " ANALYSIS \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\")
        print(f"▣ The " + type_name + " Analysis was computed using " + str(self.K) + " archetypes.")
        print(f"▣ The " + type_name + " Analysis was computed on " + str(len(self.X)) + " attributes.")
        print(f"▣ The " + type_name + " Analysis was computed on " + str(self.N) + " subjects.")
        print(f"▣ The " + type_name + " Analysis ran for " + str(self.n_iter) + " iterations.")
        print(f"▣ The " + type_name + " Analysis took " + str(self.time) + " seconds to complete.")
        print(f"▣ The final loss was: {self.loss[-1]}.")