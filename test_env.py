
from AAM import AA

N, M = 3000, 10
K = 3
p = 6
a_param = 1
b_param = 10
sigma = -6.9
sigma_dev = 0.1
rb = True
n_iter = 1000
lr = 0.1
sigma_cap = False
alternating = True
AA = AA()
AA.create_synthetic_data(N, M, K, p, sigma, rb, b_param, a_param, sigma_dev)


AA.analyse(K=K, p=p, n_iter=n_iter, early_stopping=True, model_type="OAA", lr=lr, with_synthetic_data=True, with_hot_start=False, alternating=alternating, sigma_cap=sigma_cap)
AA.plot(model_type="OAA", with_synthetic_data=True) # "PCA_scatter_plot")
AA.plot(model_type="OAA", with_synthetic_data=True, plot_type="loss_plot")


# sigma_cap=True:   loss = 15414
# sigma_cap=False   loss = 19000

# old OAA           loss = 14337

