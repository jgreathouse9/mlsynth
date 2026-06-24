import numpy as np
import torch

def gen_cov(d, noise_min=0.05, noise_max=0.2, random_seed=None, diag=True):
    rng = np.random.default_rng(random_seed)
    if diag:
        Q_diag_values = rng.standard_normal(size=d)  # random values for the diagonal
        Q = np.diag(np.abs(Q_diag_values))
    else:
        Q = rng.uniform(noise_min, noise_max, size=(d, d)) / np.sqrt(d)
        Q = Q @ Q.T  # make it symmetric positive definite
        Q += 1e-6 * np.eye(d)
        sign = np.triu(rng.choice([-1,1], size=(d,d)),1)
        sign = sign + sign.T + np.eye(d)
        Q = Q*sign

        eigvals, eigvecs = np.linalg.eigh(Q)
        eigvals[eigvals < 1e-6] = 1e-6
        Q = eigvecs @ np.diag(eigvals) @ eigvecs.T
    return Q


def gen_A(d=15, method="QR", random_seed=None):
    rng = np.random.default_rng(random_seed)

    if method=="QR":
        X = rng.normal(size=(d, d))
        A, R = np.linalg.qr(X)
    
    elif method=="dirichlet":
        A = rng.dirichlet(alpha=rng.random(d), size=d)

    elif method == "noisy_dirichlet":
        A = np.zeros((d, d))
        for i in range(d):
            a = rng.dirichlet(alpha=rng.random(d) + np.abs(rng.standard_normal(d)) * 0.1, size=1)
            # a = rng.dirichlet(alpha=np.random.rand(d))
            A[i,:] = a
    else :
        raise ValueError("method not recognized")
    return A

def gen_H(d=15, N=15, method="dirichlet", random_seed=None):
    rng = np.random.default_rng(random_seed)

    if method=="dirichlet":
        H = rng.dirichlet(alpha=rng.random(d), size=N)

    elif method=="noisy_dirichlet":
        H = np.zeros((N, d))
        for i in range(N):
            h = rng.dirichlet(alpha=rng.random(d) +np.abs(rng.standard_normal(d)) * 0.1, size=1)
            # h = rng.dirichlet(alpha=np.random.rand(d), size=1)
            H[i,:] = h
    else:
        print(method)
        raise ValueError("method not recognized")
    
    return H

def gen_dirchelet_params(d=5, N=15,
                         noise_min_q=0.01, noise_max_q=0.1,
                         noise_min_r=0.01, noise_max_r=0.1,
                         noise_min_p0=0.01, noise_max_p0=0.1,
                         random_seed=None, Q_diag=True, R_diag=True, Q_zero=False,
                         A_method="QR", H_method="dirichlet"):
    
    rng = np.random.default_rng(random_seed)
    A = gen_A(d=d, method=A_method, random_seed=random_seed)
    H = gen_H(d=d, N=N, method=H_method, random_seed=random_seed)

    if Q_zero:
        Q = np.zeros((d,d))
    else:
        Q = gen_cov(d=d, noise_min=noise_min_q, noise_max=noise_max_q, random_seed=random_seed, diag=Q_diag)

    R = gen_cov(d=N, noise_min=noise_min_r, noise_max=noise_max_r, random_seed=random_seed, diag=R_diag)

    m0 = rng.random(d)

    P0 = gen_cov(d=d, noise_min=noise_min_p0, noise_max=noise_max_p0, random_seed=random_seed, diag=False)

    return [A, H, Q, R, m0, P0]



def generate_model_data(theta, T, random_seed=None, burn_time=3, return_signal=False):
    A, H, Q, R, m0, P0 = theta

    rng = np.random.default_rng(random_seed)
    d = A.shape[0]
    N = H.shape[0]
    
    W = np.zeros((T+burn_time, d)) # hidden states
    Y_sig = np.zeros((T+burn_time, N)) # signals
    Y = np.zeros((T+burn_time, N)) # observations
    
    q = rng.multivariate_normal(mean=np.zeros(d), cov=Q)
    w = rng.multivariate_normal(mean=m0, cov=P0) + q
    
    for t in range(T+burn_time):
        r = rng.multivariate_normal(mean=np.zeros(N), cov=R)
        y = H @ w
        Y_sig[t] = y
        Y[t] = y + r
        W[t] = w 
        
        q = rng.multivariate_normal(mean=np.zeros(d), cov=Q)
        w = A @ w + q        

    if return_signal:
        return Y_sig[burn_time:,:], Y[burn_time:,:]
    else:
        return Y[burn_time:,:]
    

def generate_multiple_layers(d_true, N, T, k=1, noise_min=0, noise_max=1, random_seed=None, burn_time=3):
    rng = np.random.default_rng(random_seed)
    theta_list = []
    for _ in range(k):
        theta_true = gen_dirchelet_params(d=d_true, N=N, noise_min = noise_min, noise_max=noise_max, 
                                    Q_diag=False, R_diag=False) #[A, H, Q, R, m0, P0]
        theta_list.append(theta_true)
    
    mb = rng.random(d_true) # d-dimensional mean for b
    Pb = gen_cov(d=d_true, noise_min=noise_min, noise_max=noise_max, random_seed=random_seed, diag=False)
    b_list = [rng.multivariate_normal(mean=mb, cov=Pb) for _ in range(N)]

    mc = rng.random(d_true) # d-dimensional mean for c
    Pc = gen_cov(d=d_true, noise_min=noise_min, noise_max=noise_max, random_seed=random_seed, diag=False)
    c_list = [rng.multivariate_normal(mean=mc, cov=Pc) for _ in range(k)]

    for layer in range(k):
        A, H, Q, R, m0, P0 = theta_list[layer]
        c_k = c_list[layer]
        H_new = np.array(b_list) @ np.diag(c_k) # N by d
        theta_list[layer] = [A, H_new, Q, R, m0, P0]

    Y_sig_list = []
    Y_list = []
    for layer in range(k):
        theta_true = theta_list[layer]
        Y_sig, Y = generate_model_data(theta_true, T=T, random_seed=random_seed, burn_time=burn_time, return_signal=True) 
        Y_sig_list.append(Y_sig)
        Y_list.append(Y)
    
    return torch.tensor(Y_sig_list), torch.tensor(Y_list)

def data_flatten(Y, method="time"):
    k, T, N = Y.shape
    if method == "time":
        Y = Y.permute(1, 0, 2).reshape(T, k*N)  # (T, k*N)
    elif method == "unit":
        Y = Y.permute(2, 0, 1).reshape(N, k*T)  # (N, k*T)
    else:
        raise ValueError("method should be 'time' or 'unit'")
    return Y

if __name__ == "__main__":

    d_true=10
    N=5
    T=3
    T0 = 39
    theta_true = gen_dirchelet_params(d=d_true, N=N, noise_min = 0.1, noise_max=1,
                                    Q_diag=False, R_diag=False, ) #[A, H, Q, R, m0, P0]
    Y_sig, Y = generate_model_data(theta_true, T=T, return_signal=True) # T by N
    print(theta_true[2])
    print(theta_true[3])

    Q_true = theta_true[2]
    R_true = theta_true[3]
    print(Q_true)
    # print(R_true)

    Y_sig, Y = generate_model_data(theta_true, T=T, return_signal=True) # T by N

    print(Y)
    
    print("hello")