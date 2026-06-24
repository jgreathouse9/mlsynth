import torch
import random
from torch import nn
from matrix import Matrix
from genData.SSM import *


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class TimeAwareSC:
    def __init__(self, Y, d,
                 T0=None,
                 Q_diag=True, R_diag=True,
                 learn_Q=True, learn_R=True,
                 require_thresh=False,
                 device="cpu", dtype=torch.float32):
        self.eps = 1e-6
        self.device = device
        self.dtype = dtype

        if not torch.is_tensor(Y):
            Y = torch.tensor(Y, dtype=dtype, device=device)
        self.Y = Y  # (N, T)

        self.N, self.T = Y.shape
        self.T0 = T0
        self.d = d

        zeros_row = torch.zeros(1, self.N, dtype=dtype, device=device)
        self.y_seq = torch.cat((zeros_row, Y.T), dim=0)  # (T+1, N)

        # Parameters (initialized later)
        self.A = None
        self.H = None
        self.Lq = None
        self.Lr = None
        self.Lp0 = None
        self.m0 = None

        self.Q_diag = Q_diag
        self.R_diag = R_diag
        self.learn_Q = learn_Q
        self.learn_R = learn_R
        self.require_thresh = require_thresh

        # Cache identities
        self.I_d = torch.eye(self.d, dtype=dtype, device=device)
        self.I_N = torch.eye(self.N, dtype=dtype, device=device)

        # Kalman sequences
        self.m_seq = [None] * (self.T + 1)
        self.P_seq = [None] * (self.T + 1)

        # RTS sequences
        self.m_s_seq = [None] * (self.T + 1)
        self.P_s_seq = [None] * (self.T + 1)
        self.G_seq = [None] * (self.T)

        self.total_em_rounds = 0
        self.converged = None
        self.random_seed = 0

    # ---------- SPD helpers & properties ----------
    def _make_spd(self, L, out_dim):
        """Build SPD from unconstrained lower-triangular param."""
        L = torch.tril(L)
        diag = torch.nn.functional.softplus(torch.diagonal(L, 0)) + 1e-6
        L = L - torch.diag_embed(torch.diagonal(L)) + torch.diag_embed(diag)
        M = L @ L.T
        if M.shape[-1] != out_dim:
            M = M[:out_dim, :out_dim]
        M = M + self.eps * (self.I_d if out_dim == self.d else self.I_N)
        return M

    @property
    def Q(self):
        return self._make_spd(self.Lq, self.d)

    @Q.setter
    def Q(self, Q_new):
        with torch.no_grad():
            try:
                Lq_new = torch.linalg.cholesky(Q_new + self.eps * self.I_d)
            except RuntimeError:
                Lq_new = torch.linalg.cholesky(Q_new + 1e-4 * self.I_d)
            self.Lq.copy_(Lq_new)

    @property
    def R(self):
        return self._make_spd(self.Lr, self.N)

    @R.setter
    def R(self, R_new):
        with torch.no_grad():
            try:
                Lr_new = torch.linalg.cholesky(R_new + self.eps * self.I_N)
            except RuntimeError:
                Lr_new = torch.linalg.cholesky(R_new + 1e-4 * self.I_N)
            self.Lr.copy_(Lr_new)

    @property
    def P0(self):
        return self._make_spd(self.Lp0, self.d)

    @P0.setter
    def P0(self, P0_new):
        with torch.no_grad():
            try:
                Lp0_new = torch.linalg.cholesky(P0_new + self.eps * self.I_d)
            except RuntimeError:
                Lp0_new = torch.linalg.cholesky(P0_new + 1e-4 * self.I_d)
            self.Lp0.copy_(Lp0_new)

    # ---------- Initialization ----------
    def initialize_theta(self, method='dirichlet', A_method="QR", H_method="dirichlet", Q_zero=False, random_seed=None):
        if self.d is None:
            raise ValueError("Dimension 'd' must be set before initializing parameters.")

        if method == 'naive':
            A = torch.eye(self.d, dtype=self.dtype, device=self.device)
            H = torch.randn(self.N, self.d, dtype=self.dtype, device=self.device)
            Q = torch.eye(self.d, dtype=self.dtype, device=self.device) * 0.1
            R = torch.eye(self.N, dtype=self.dtype, device=self.device) * 0.1
            m0 = torch.zeros(self.d, dtype=self.dtype, device=self.device)
            P0 = torch.eye(self.d, dtype=self.dtype, device=self.device)

        elif method == 'dirichlet':
            A_np, H_np, Q_np, R_np, m0_np, P0_np = gen_dirchelet_params(
                d=self.d, N=self.N, Q_diag=self.Q_diag, R_diag=self.R_diag,
                Q_zero=Q_zero, A_method=A_method, H_method=H_method, random_seed=random_seed
            )
            A = torch.tensor(A_np, dtype=self.dtype, device=self.device)
            H = torch.tensor(H_np, dtype=self.dtype, device=self.device)
            Q = torch.tensor(Q_np, dtype=self.dtype, device=self.device)
            R = torch.tensor(R_np, dtype=self.dtype, device=self.device)
            m0 = torch.tensor(m0_np, dtype=self.dtype, device=self.device)
            P0 = torch.tensor(P0_np, dtype=self.dtype, device=self.device)

        elif method == 'pca':
            T_for_pca = self.T0 if self.T0 is not None else self.T
            u, s, vh = torch.linalg.svd(self.Y[:, :T_for_pca].double(), full_matrices=False)
            full_rank = min(self.N, T_for_pca)
            if self.d <= full_rank:
                u = u[:, :self.d]
                s = s[:self.d]
                vh = vh[:self.d, :]
                H = (u @ torch.diag(s)).to(dtype=self.dtype, device=self.device)
                A = torch.zeros((self.d, self.d), dtype=self.dtype, device=self.device)
                for t in range(vh.shape[1] - 1):
                    A += torch.outer(vh[:, t + 1], vh[:, t])
                A /= max(1, (vh.shape[1] - 1))
                m0 = torch.zeros(self.d, dtype=self.dtype, device=self.device)
            else:
                H = torch.randn(self.N, self.d, dtype=self.dtype, device=self.device)
                H[:, :full_rank] = (u[:, :full_rank] @ torch.diag(s[:full_rank])).to(dtype=self.dtype, device=self.device)
                A = torch.randn(self.d, self.d, dtype=self.dtype, device=self.device) * 0.1
                m0 = torch.zeros(self.d, dtype=self.dtype, device=self.device)

            P0 = torch.eye(self.d, dtype=self.dtype, device=self.device)
            Q = torch.eye(self.d, dtype=self.dtype, device=self.device) * 0.1
            R = torch.eye(self.N, dtype=self.dtype, device=self.device) * 0.1

        else:
            raise ValueError(f"Unknown init method: {method}")

        # Core parameters
        self.A = nn.Parameter(A, requires_grad=True)
        self.H = nn.Parameter(H, requires_grad=True)
        self.m0 = nn.Parameter(m0, requires_grad=True)

        # Cholesky factors as the learnable params
        self.Lq = nn.Parameter(torch.linalg.cholesky(Q + self.eps * self.I_d), requires_grad=True)
        self.Lr = nn.Parameter(torch.linalg.cholesky(R + self.eps * self.I_N), requires_grad=True)
        self.Lp0 = nn.Parameter(torch.linalg.cholesky(P0 + self.eps * self.I_d), requires_grad=True)

    # ---------- Kalman filter & smoother ----------
    def kalman_filter_step(self, y_k, m_prev, P_prev, post=False):
        A, H = self.A, self.H
        Q, R = self.Q, self.R
        if post:
            R = R.clone()
            R[0, 0] = 1e15

        m_pred = A @ m_prev
        P_pred = A @ P_prev @ A.T + Q + self.eps * self.I_d

        v = y_k - (H @ m_pred)
        S = H @ P_pred @ H.T + R + self.eps * self.I_N

        # Solve instead of inv
        K = P_pred @ H.T @ torch.linalg.inv(S)
        # K = torch.linalg.solve(S, (H @ P_pred).T).T

        m_filt = m_pred + K @ v
        P_filt = P_pred - K @ S @ K.T
        return m_filt, P_filt

    def rts_smoother_step(self, m_filt, P_filt, m_smooth_next, P_smooth_next):
        A, Q = self.A, self.Q
        P_pred = A @ P_filt @ A.T + Q + self.eps * self.I_d
        # Gk = P_filt A^T P_pred^{-1} (pinv for safety)
        Gk = P_filt @ A.T @ torch.linalg.pinv(P_pred)
        m_smooth = m_filt + Gk @ (m_smooth_next - (A @ m_filt))
        P_smooth = P_filt + Gk @ (P_smooth_next - P_pred) @ Gk.T
        return m_smooth, P_smooth, Gk

    def kalman_filter(self, T=None):
        if T is None:
            T = self.T0

        y_seq = self.y_seq.clone()
        m_seq = [None] * (T + 1)
        P_seq = [None] * (T + 1)
        m_seq[0] = self.m0
        P_seq[0] = self.P0

        for k in range(1, T + 1):
            post = (k > self.T0)
            y_k = y_seq[k].clone()
            if post:
                y_k[0] = 0.0
            m_seq[k], P_seq[k] = self.kalman_filter_step(y_k, m_seq[k - 1], P_seq[k - 1], post=post)

        self.m_seq[:T + 1] = m_seq
        self.P_seq[:T + 1] = P_seq

    def rts_smoother(self, T=None):
        if T is None:
            T = self.T0

        m_s_seq = [None] * (T + 1)
        P_s_seq = [None] * (T + 1)
        G_seq = [None] * (T)

        m_s_seq[T] = self.m_seq[T]
        P_s_seq[T] = self.P_seq[T]

        for k in range(T - 1, -1, -1):
            m_s_seq[k], P_s_seq[k], G_seq[k] = self.rts_smoother_step(
                self.m_seq[k], self.P_seq[k], m_s_seq[k + 1], P_s_seq[k + 1]
            )

        self.m_s_seq[:T + 1] = m_s_seq
        self.P_s_seq[:T + 1] = P_s_seq
        self.G_seq[:T] = G_seq

    # ---------- Intermediate values ----------
    def get_intermediate_values(self, T):
        m_s = torch.stack(self.m_s_seq[:T+1])          # (T+1, d)
        P_s = torch.stack(self.P_s_seq[:T+1])          # (T+1, d, d)
        y_seq = self.y_seq[:T+1]                       # (T+1, N)

        m1, P1 = m_s[1:], P_s[1:]                      # k = 1..T
        m0, P0 = m_s[:-1], P_s[:-1]                    # k = 0..T-1

        Sigma = torch.mean(P1 + m1.unsqueeze(-1) @ m1.unsqueeze(1), dim=0)  # (d,d)
        Phi   = torch.mean(P0 + m0.unsqueeze(-1) @ m0.unsqueeze(1), dim=0)  # (d,d)

        B = torch.mean(y_seq[1:].unsqueeze(-1) @ m1.unsqueeze(1), dim=0)    # (N,d)

        G_stack = torch.stack(self.G_seq[:T])                                 # (T, d, d)
        C = torch.mean(P1 @ G_stack.transpose(-1, -2) + m1.unsqueeze(-1) @ m0.unsqueeze(1), dim=0)  # (d,d)

        D = torch.mean(y_seq[1:].unsqueeze(-1) @ y_seq[1:].unsqueeze(1), dim=0)  # (N,N)

        Sigma = Sigma + self.eps * self.I_d
        Phi   = Phi + self.eps * self.I_d
        D     = D + self.eps * self.I_N

        return Sigma, Phi, B, C, D

    # ---------- EM ----------
    def m_step(self, T, thresh=1e-4, modified_m_step=False):
        if self.T0 is None:
            raise ValueError("T0 must be set before running M-step.")

        Sigma, Phi, B, C, D = self.get_intermediate_values(T)

        # Optional tweak carried over
        if T > self.T0 and modified_m_step:
            with torch.no_grad():
                b1 = torch.mean(torch.stack([torch.outer(self.y_seq[k], self.m_s_seq[k]) for k in range(1, self.T0 + 1)], dim=0), dim=0)[0, :]
                d1 = torch.mean(torch.stack([torch.outer(self.y_seq[k], self.y_seq[k]) for k in range(1, self.T0 + 1)], dim=0), dim=0)[0, 0]
                B[0, :] = b1
                D[0, 0] = d1

        A_new = C @ torch.linalg.inv(Phi)
        H_new = B @ torch.linalg.inv(Sigma)

        Q_new, R_new = None, None
        if self.learn_Q:
            Q_new = Sigma - C @ A_new.T - A_new @ C.T + A_new @ Phi @ A_new.T
            if self.Q_diag:
                diag = torch.diag(Q_new)
                if self.require_thresh:
                    diag = torch.maximum(diag, torch.full_like(diag, thresh))
                Q_new = torch.diag(diag)
            Q_new = Q_new + self.eps * self.I_d

        if self.learn_R:
            R_new = D - B @ H_new.T - H_new @ B.T + H_new @ Sigma @ H_new.T
            if self.R_diag:
                diag = torch.diag(R_new)
                if self.require_thresh:
                    diag = torch.maximum(diag, torch.full_like(diag, thresh))
                R_new = torch.diag(diag)
            R_new = R_new + self.eps * self.I_N

        m0_new = self.m_s_seq[0]
        P0_new = self.P_s_seq[0] + self.eps * self.I_d

        with torch.no_grad():
            self.A.copy_(A_new)
            self.H.copy_(H_new)
            self.m0.copy_(m0_new)
            if Q_new is not None: self.Q = Q_new   # property setter -> Lq
            if R_new is not None: self.R = R_new   # property setter -> Lr
            self.P0 = P0_new                        # property setter -> Lp0

    def get_stopping_condition(self, theta, theta_pre, tol=1e-4):
        A, H, Lq, Lr, Lp0, m0 = theta
        A_pre, H_pre, Lq_pre, Lr_pre, Lp0_pre, m0_pre = theta_pre
        return (torch.linalg.norm(A - A_pre) > tol or
                torch.linalg.norm(H - H_pre) > tol or
                torch.linalg.norm(Lq - Lq_pre) > tol or
                torch.linalg.norm(Lr - Lr_pre) > tol or
                torch.linalg.norm(Lp0 - Lp0_pre) > tol or
                torch.linalg.norm(m0 - m0_pre) > tol)

    def em_algorithm(self, T, N_ITER=1000, modified_m_step=False):
        if self.T0 is None:
            raise ValueError("T0 must be set before running EM algorithm.")

        i = 0
        stopping_condition = True
        while (stopping_condition) and (i < N_ITER):
            self.total_em_rounds += 1
            i += 1

            self.kalman_filter(T)
            self.rts_smoother(T)

            theta_pre = [t.clone() for t in [self.A, self.H, self.Lq, self.Lr, self.Lp0, self.m0]]
            self.m_step(T, modified_m_step=modified_m_step)

            try:
                _ = self.log_likelihood(T=T)
            except Exception:
                pass

            stopping_condition = self.get_stopping_condition(
                [self.A, self.H, self.Lq, self.Lr, self.Lp0, self.m0], theta_pre
            )
            if stopping_condition == False:
                self.total_em_rounds -= i
                self.converged = True
                print("Stopping condition met, exiting loop at i =", i)

        return (self.A, self.H, self.Q, self.R, self.P0,
                self.m0, self.m_seq, self.P_seq, self.m_s_seq, self.P_s_seq)

    def em_pre(self, T0, N1):
        with torch.no_grad():
            self.T0 = T0
            if self.A is None:
                raise ValueError("Parameters must be initialized before running EM algorithm.")
            self.total_em_rounds = 0
        self.em_algorithm(T=T0, N_ITER=N1)

    def em_post(self, N2, modified_m_step=False):
        with torch.no_grad():
            if self.T0 is None:
                raise ValueError("Did you run em_pre first? T0 must be set before running post-intervention EM algorithm.")
            if self.A is None:
                raise ValueError("Did you run em_pre first? Parameters must be initialized before running post-intervention EM algorithm.")
        self.em_algorithm(T=self.T, N_ITER=N2, modified_m_step=modified_m_step)

    def em_full(self, T0, N1, N2):
        self.em_pre(T0, N1)
        self.em_post(N2)

    # ---------- Likelihood ----------
    def log_likelihood(self, T=None):
        if T is None:
            T = self.T0

        d, N = self.d, self.N
        Sigma, Phi, B, C, D = self.get_intermediate_values(T)
        Q, R, P0 = self.Q, self.R, self.P0

        def slogdet_logZ(A, dim):
            sign, logabs = torch.linalg.slogdet(A)
            return dim * torch.log(torch.tensor(2 * torch.pi, dtype=self.dtype, device=self.device)) + logabs

        Q_func = 0.0
        Q_func = Q_func + slogdet_logZ(P0, d)
        Q_func = Q_func + T * slogdet_logZ(Q, d)
        Q_func = Q_func + T * slogdet_logZ(R, N)

        Lp0 = torch.linalg.cholesky(P0)
        Lq = torch.linalg.cholesky(Q)
        Lr = torch.linalg.cholesky(R)

        Q_func = Q_func + torch.trace(torch.cholesky_solve(
            self.P_s_seq[0] + torch.outer(self.m_s_seq[0] - self.m0, self.m_s_seq[0] - self.m0), Lp0))
        Q_func = Q_func + T * torch.trace(torch.cholesky_solve(
            Sigma - C @ self.A.T - self.A @ C.T + self.A @ Phi @ self.A.T, Lq))
        Q_func = Q_func + T * torch.trace(torch.cholesky_solve(
            D - B @ self.H.T - self.H @ B.T + self.H @ Sigma @ self.H.T, Lr))

        return -0.5 * Q_func

    # ---------- Training ----------
    def forward(self, T):
        self.kalman_filter(T)
        self.rts_smoother(T)

    def gradient_ascent(self, optimizer, scheduler, n_steps=100, loss_type="loglikelihood"):
        """
        Run gradient ascent on log likelihood (or other loss).
        """
        for step in range(n_steps):
            optimizer.zero_grad()

            # One forward (KF + RTS) per step
            self.kalman_filter(self.T0)
            self.rts_smoother(self.T0)

            if loss_type == "loglikelihood":
                ll = self.log_likelihood(T=self.T0)
                loss = -ll
            elif loss_type == "norm":
                Y_filt_pre = (self.H @ torch.stack(self.m_s_seq[1:self.T0 + 1]).T)  # N x T0
                target_pred = Y_filt_pre[0, :]
                donors_pred = Y_filt_pre[1:, :]
                loss = torch.norm(target_pred - self.Y[0, :self.T0]) + torch.norm(donors_pred - self.Y[1:, :self.T0])
            elif loss_type == "mixed":
                ll = self.log_likelihood(T=self.T0)
                Y_filt_pre = (self.H @ torch.stack(self.m_s_seq[1:self.T0 + 1]).T)
                target_pred = Y_filt_pre[0, :]
                donors_pred = Y_filt_pre[1:, :]
                loss = torch.norm(target_pred - self.Y[0, :self.T0]) + torch.norm(donors_pred - self.Y[1:, :self.T0]) - 0.1 * ll
            else:
                raise ValueError(f"Unknown loss_type: {loss_type}")

            loss.backward()
            optimizer.step()
            scheduler.step()

            if step % 100 == 0 or step == n_steps - 1:
                with torch.no_grad():
                    print(f"[Step {step}] log likelihood: {self.log_likelihood(T=self.T0).item():.6f}")

    # ---------- Prediction utilities ----------
    def predict_post_intervention(self):
        """with the fixed parameters, predict the target observations"""
        A = self.A.detach()
        H = self.H.detach()
        T0 = self.T0
        T = self.T

        post_pred = []
        x_pred = A @ self.m_s_seq[T0]
        for i in range(T - T0):
            if i > 0:
                x_pred = A @ x_pred
            y_pred = H @ x_pred
            post_pred.append(y_pred.unsqueeze(0))   # (1, N)

        post_pred = torch.cat(post_pred, dim=0)  # (T-T0, N)
        return post_pred

    def augment_target(self, ys):
        """augment the target observations with the prediction based on current parameters"""
        post_pred = self.predict_post_intervention()
        target_pred = post_pred[:, 0]  # assuming target is the first dimension
        ys_augmented = ys.clone()
        ys_augmented[self.T0:, 0] = target_pred

        zeros_row = torch.zeros(1, self.N, dtype=self.dtype, device=self.device)
        y_seq_augmented = torch.cat((zeros_row, ys_augmented), dim=0)  # (T+1, N)
        return y_seq_augmented

    def make_prediction(self, T0=None):
        y_seq_augmented = self.augment_target(self.Y.T)    # T x N
        T0 = self.T0 if T0 is None else T0
        T = self.T

        # Temporarily swap y_seq (no in-place writes)
        original = self.y_seq
        try:
            self.y_seq = y_seq_augmented
            self.kalman_filter(T)
            self.rts_smoother(T)
        finally:
            self.y_seq = original

        # prediction for y
        Y_filtered = self.H @ torch.stack(self.m_s_seq[1:]).T  # N x T
        target_prediction = Y_filtered[0, :]                   # length T
        donor_prediction = Y_filtered[1:, :]                   # (N-1) x T

        # target variance estimates
        h1 = self.H[0]
        target_var_estimates = torch.tensor([h1 @ P_s @ h1.T for P_s in self.P_s_seq[1:]], dtype=self.dtype, device=self.device)  # length T

        return target_prediction, donor_prediction, target_var_estimates


if __name__ == "__main__":
    seed = 20
    set_seed(seed)

    d_true = 5
    N = 10
    T = 100
    theta_true = gen_dirchelet_params(d=d_true, N=N, random_seed=seed)
    Y = generate_model_data(theta_true, T=T, random_seed=seed)
    Y = torch.tensor(Y, dtype=torch.float32)    # (T, N)

    d = 5
    T0 = 50
    N1 = 100
    N2 = 10
    lr = 1e-3

    # PCA baseline
    u, s, vh = torch.linalg.svd(Y)
    Y_pca = (u[:, :d] @ torch.diag(s[:d]) @ vh[:d, :])
    pca_error = torch.norm(Y_pca[T0:, 0] - Y[T0:, 0])
    print("Frobenius norm between PCA and true signal     :", pca_error.item())

    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(seed=100)
    model = TimeAwareSC(Y=Y.T.to(device), d=d, device=device, dtype=torch.float32)
    model.initialize_theta(method='pca', random_seed=seed)
    model.T0 = T0
\

    # EM
    model.em_full(T0=T0, N1=N1, N2=N2)
    # EM performance
    print("After EM:", model.log_likelihood(T=T0).item())
    with torch.no_grad():
        target_pred, donor_pred, target_var_estimates = model.make_prediction()
    pred_error = torch.norm(target_pred[T0:] - Y[T0:, 0].to(device))
    print("Prediction error (L2 norm):", pred_error.item())

    # for _ in range(10):
    #     print("--------------------------")

        # # Gradient ascent
        # optimizer = torch.optim.Adam([model.A, model.H, model.Lq, model.Lr, model.Lp0, model.m0], lr=lr)
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.995)
        # model.gradient_ascent(optimizer, scheduler, loss_type="loglikelihood", n_steps=100)
        # # Gradient ascent performance
        # print("After gradient ascent:", model.log_likelihood(T=T0).item())
        # with torch.no_grad():
        #     target_pred, donor_pred, target_var_estimates = model.make_prediction()
        # pred_error = torch.norm(target_pred[T0:] - Y[T0:, 0].to(device))
        # print("Prediction error (L2 norm):", pred_error.item())

        # decay
        # N1 = int(N1 * 0.7)
        # N2 = int(N2 * 0.7)
        # lr = lr * 0.7

    print("finished")
