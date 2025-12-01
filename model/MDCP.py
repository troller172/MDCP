import numpy as np
from sklearn.preprocessing import SplineTransformer
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.base import clone
from const import *

np.random.seed(RANDOM_SEED)

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(RANDOM_SEED)
        torch.cuda.manual_seed_all(RANDOM_SEED)
except Exception:
    TORCH_AVAILABLE = False


def ensure_2d(X):
    """Ensure input is a 2D array by reshaping 1D inputs into column vectors."""

    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    return X


def ensure_2d_query(X):
    """Ensure query inputs are represented as 2D row vectors."""

    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    return X


def softplus_np(z):
    """Numerically stable softplus implementation."""

    z = np.asarray(z)
    large_mask = z > 10
    small_mask = z < -10
    medium_mask = ~(large_mask | small_mask)

    result = np.zeros_like(z)
    result[large_mask] = z[large_mask]
    result[small_mask] = 0.0
    result[medium_mask] = np.log1p(np.exp(z[medium_mask]))
    return result


class SourceModelClassification:
    """Conditional probability estimator with cross-validated calibration."""

    def __init__(
        self,
        X,
        Y,
        learner='gbm',
        calibration_method='isotonic',
        n_splits=5,
        probability_floor=1e-6,
    ):
        self.X = ensure_2d(X)
        self.Y = np.asarray(Y).astype(int)
        if self.X.shape[0] != self.Y.shape[0]:
            raise ValueError("X and Y must contain the same number of samples.")
        self.classes, class_counts = np.unique(self.Y, return_counts=True)
        if self.classes.size < 2:
            raise ValueError("Classification source requires at least two classes.")

        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.learner = learner
        self.calibration_method = calibration_method
        self.prob_floor = probability_floor
        self.n_splits = self._resolve_cv_splits(class_counts, n_splits)

        base_estimator = self._build_base_estimator()
        cv = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=RANDOM_SEED,
        )
        try:
            self.model = CalibratedClassifierCV(
                base_estimator=base_estimator,
                method=self.calibration_method,
                cv=cv,
            )
        except TypeError:
            # scikit-learn >=1.4 renamed argument to ``estimator``
            self.model = CalibratedClassifierCV(
                estimator=base_estimator,
                method=self.calibration_method,
                cv=cv,
            )
        self.model.fit(self.X, self.Y)
        self.classes_ = self.model.classes_

    def _resolve_cv_splits(self, class_counts, requested):
        min_per_class = int(np.min(class_counts))
        return max(2, min(requested, min_per_class))

    def _build_base_estimator(self):
        learner = (self.learner or 'gbm').lower()
        if learner == 'rf':
            return RandomForestClassifier(
                n_estimators=500,
                max_features='sqrt',
                min_samples_leaf=2,
                n_jobs=-1,
                random_state=RANDOM_SEED,
            )
        return HistGradientBoostingClassifier(
            random_state=RANDOM_SEED,
            learning_rate=0.1,
        )

    def f_x(self, x_query):
        Xq = ensure_2d(x_query)
        return np.ones(Xq.shape[0])

    def predict_proba(self, x_query):
        Xq = ensure_2d(x_query)
        probs = self.model.predict_proba(Xq)
        probs = np.clip(probs, self.prob_floor, 1.0)
        probs_sum = np.maximum(probs.sum(axis=1, keepdims=True), 1e-12)
        return probs / probs_sum

    def f_y_given_x(self, x_query):
        return self.predict_proba(x_query)

    def joint_prob(self, x_query, y_vals):
        Xq = ensure_2d(x_query)
        probs = self.predict_proba(Xq)
        y_vals = np.asarray(y_vals)
        out = np.full((Xq.shape[0], len(y_vals)), self.prob_floor)

        for idx, y in enumerate(y_vals):
            if y in self.class_to_idx:
                out[:, idx] = probs[:, self.class_to_idx[y]]
        return out

    def joint_prob_at_pairs(self, Xpairs, Ypairs):
        Xpairs = ensure_2d(Xpairs)
        probs = self.predict_proba(Xpairs)
        Ypairs = np.asarray(Ypairs).reshape(-1)
        result = np.full(Ypairs.shape[0], self.prob_floor)
        idx = np.arange(Ypairs.shape[0])
        y_indices = np.array([self.class_to_idx.get(y, -1) for y in Ypairs])
        mask = y_indices >= 0
        if np.any(mask):
            result[mask] = probs[idx[mask], y_indices[mask]]
        return result


class PrecomputedProbabilitySource:
    """Wrap precomputed class probabilities as a source model interface."""

    def __init__(
        self,
        X_reference,
        Y_reference,
        probability_matrix,
        *,
        scale=None,
        prob_floor=1e-12,
    ):
        self.X_reference = ensure_2d(X_reference) if X_reference is not None else None
        self.Y_reference = (
            np.asarray(Y_reference).astype(int) if Y_reference is not None else None
        )
        self._probability = np.asarray(probability_matrix, dtype=float)
        if self._probability.ndim != 2:
            raise ValueError("probability_matrix must be 2-D")

        self._max_index = max(self._probability.shape[0] - 1, 0)
        self._scale = float(scale) if scale is not None else float(max(self._max_index, 1))
        if self._scale <= 0:
            self._scale = 1.0

        self.prob_floor = float(prob_floor)
        n_classes = self._probability.shape[1]
        self.classes = np.arange(n_classes, dtype=int)
        self.class_to_idx = {int(c): int(idx) for idx, c in enumerate(self.classes)}

    def _indices_from_features(self, X_query):
        X_arr = ensure_2d(X_query)
        feature_col = np.asarray(X_arr[:, 0], dtype=float)
        scaled = np.clip(feature_col, 0.0, 1.0) * self._scale
        idx = np.rint(scaled).astype(int)
        return np.clip(idx, 0, self._max_index)

    def f_x(self, x_query):
        Xq = ensure_2d(x_query)
        return np.ones(Xq.shape[0])

    def marginal_pdf_x(self, x_query):
        Xq = ensure_2d(x_query)
        return np.ones(Xq.shape[0])

    def predict_proba(self, x_query):
        idx = self._indices_from_features(x_query)
        probs = self._probability[idx]
        return np.clip(probs, self.prob_floor, 1.0)

    def f_y_given_x(self, x_query):
        return self.predict_proba(x_query)

    def joint_prob(self, x_query, y_values):
        idx = self._indices_from_features(x_query)
        classes = np.asarray(y_values, dtype=int)
        matrix = self._probability[np.ix_(idx, classes)]
        return np.clip(matrix, self.prob_floor, 1.0)

    def joint_prob_at_pairs(self, Xpairs, Ypairs):
        idx = self._indices_from_features(Xpairs)
        labels = np.asarray(Ypairs, dtype=int)
        probs = self._probability[idx, labels]
        return np.clip(probs, self.prob_floor, 1.0)

    # For regression compatibility hooks used elsewhere
    joint_pdf = joint_prob
    joint_pdf_at_pairs = joint_prob_at_pairs


class SourceModelRegressionGaussian:
    """Gaussian plug-in conditional density obtained via OOF mean and log-variance models."""

    def __init__(
        self,
        X,
        Y,
        learner='gbm',
        n_splits=5,
        variance_floor=1e-6,
        pdf_floor=1e-12,
    ):
        self.X = ensure_2d(X)
        self.Y = np.asarray(Y).reshape(-1)
        if self.X.shape[0] != self.Y.shape[0]:
            raise ValueError("X and Y must contain the same number of samples.")
        if self.X.shape[0] < 5:
            raise ValueError("Regression source requires at least five samples.")

        self.learner = learner
        self.n_splits = max(2, min(n_splits, self.X.shape[0] - 1))
        self.variance_floor = variance_floor
        self.pdf_floor = pdf_floor
        self.log_variance_model = None
        self._log_var_constant = None

        self._fit_models()

    def _build_regressor(self):
        learner = (self.learner or 'gbm').lower()
        if learner == 'rf':
            return RandomForestRegressor(
                n_estimators=500,
                min_samples_leaf=3,
                n_jobs=-1,
                random_state=RANDOM_SEED,
            )
        return HistGradientBoostingRegressor(random_state=RANDOM_SEED)

    def _fit_models(self):
        base_mean = self._build_regressor()
        cv = KFold(n_splits=self.n_splits, shuffle=True, random_state=RANDOM_SEED)
        oof_mu = np.zeros(self.X.shape[0])

        for train_idx, val_idx in cv.split(self.X):
            model = clone(base_mean)
            model.fit(self.X[train_idx], self.Y[train_idx])
            oof_mu[val_idx] = model.predict(self.X[val_idx])

        self.mean_model = clone(base_mean)
        self.mean_model.fit(self.X, self.Y)

        resid_sq = (self.Y - oof_mu) ** 2
        resid_sq = np.maximum(resid_sq, self.variance_floor)
        self._log_var_constant = float(np.log(np.mean(resid_sq)))

        if np.allclose(resid_sq, resid_sq[0], atol=1e-10):
            self.log_variance_model = None
        else:
            base_var = self._build_regressor()
            log_resid = np.log(resid_sq)
            self.log_variance_model = clone(base_var)
            self.log_variance_model.fit(self.X, log_resid)

    def _predict_log_variance(self, Xq):
        if self.log_variance_model is None:
            return np.full(Xq.shape[0], self._log_var_constant)
        log_var = self.log_variance_model.predict(Xq)
        min_log = np.log(self.variance_floor)
        return np.maximum(log_var, min_log)

    def predict_mu(self, x_query):
        Xq = ensure_2d(x_query)
        return self.mean_model.predict(Xq)

    def predict_sigma(self, x_query):
        Xq = ensure_2d(x_query)
        log_var = self._predict_log_variance(Xq)
        var = np.maximum(np.exp(log_var), self.variance_floor)
        return np.sqrt(var)

    def marginal_pdf_x(self, x_query):
        Xq = ensure_2d(x_query)
        return np.ones(Xq.shape[0])

    def conditional_pdf_y_given_x(self, x_query, y_vals):
        Xq = ensure_2d(x_query)
        y_vals = np.asarray(y_vals)
        mu = self.predict_mu(Xq)[:, None]
        sigma = self.predict_sigma(Xq)[:, None]
        sigma = np.maximum(sigma, 1e-12)
        z = (y_vals[None, :] - mu) / sigma
        pdf = (1.0 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * z ** 2)
        return np.clip(pdf, self.pdf_floor, None)

    def joint_pdf(self, x_query, y_vals):
        return self.conditional_pdf_y_given_x(x_query, y_vals)

    def joint_pdf_xy(self, x_query, y_vals):
        return self.joint_pdf(x_query, y_vals)

    def joint_pdf_at_pairs(self, Xpairs, Ypairs):
        Xpairs = ensure_2d(Xpairs)
        Ypairs = np.asarray(Ypairs).reshape(-1)
        mu = self.predict_mu(Xpairs)
        sigma = self.predict_sigma(Xpairs)
        sigma = np.maximum(sigma, 1e-12)
        z = (Ypairs - mu) / sigma
        pdf = (1.0 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * z ** 2)
        return np.clip(pdf, self.pdf_floor, None)


# --------------------------
# Lambda parameterizations
# --------------------------
class LambdaSpline:
    r"""
    lambda_j(x) = softplus( Phi(x) @ Theta.T ), Theta shape (K, m)
    Fit Theta by minimizing empirical marginal objective with scipy.minimize.
    
    The marginal loss function is:
    Phi_marg^(n)(lambda(·)) := (1-alpha) ∫_X Σ_j lambda_j(x) d_nv(x) + ∫_X ∫_Y (1 - h_lambda^(n)(x,y))_- d_mv(y) d_nv(x)
    where h_lambda^(n)(x,y) = Σ_j lambda_j(x) \hat{f}_j^(n)(x,y)
    
    We approximate the integrals using the empirical distribution on the training set.
    """
    def __init__(self, X_train, K, n_splines=5, degree=3, include_bias=True, diff_order=2,
                 gamma1=0.01, gamma2=0.01, gamma3=0):
        self.X_train = ensure_2d(X_train)
        self.K = K  # number of sources
        self.spline = SplineTransformer(n_knots=n_splines, degree=degree, include_bias=include_bias)
        self.Phi = self.spline.fit_transform(self.X_train)  # (n, m), m depends on n_knots and degree
        self.n, self.m = self.Phi.shape
        
        # Initialize Theta
        rng = np.random.default_rng(RANDOM_SEED)
        self.Theta = rng.standard_normal((self.K, self.m))

        diff_order = diff_order  # difference order for smoothness penalty
        m = self.m
        if diff_order == 1:
            # D shape (m-1, m)
            D = np.zeros((m-1, m))
            for i in range(m-1):
                D[i, i] = -1.0
                D[i, i+1] = 1.0
        else:
            # second difference: D shape (m-2, m)
            D = np.zeros((m-2, m))
            for i in range(m-2):
                D[i, i] = 1.0
                D[i, i+1] = -2.0
                D[i, i+2] = 1.0
        self._D = D
        self._gamma1 = gamma1  # L2 penalty on lambda
        self._gamma2 = gamma2  # smoothness penalty on Theta
        self._gamma3 = gamma3  # penalty on h_lambda values

    def lambda_at_Phi(self, Phi):
        A = Phi @ self.Theta.T  # (q, K)
        return softplus_np(A)   # (q, K)

    def lambda_at_x(self, x_query):
        Xq = ensure_2d(x_query)
        Phi_q = self.spline.transform(Xq)
        return self.lambda_at_Phi(Phi_q)

    def objective_flat_marginal(self, theta_flat, F_train_joint, X_train, Y_train, alpha, task='regression', 
                                source_weights=None):
        r"""
        Empirical marginal objective function:
        \hat{Phi}_marg(λ(·)) := (1/N) Σ_i [(1 - h_λ(X_i,Y_i))_- / \hat{p}_data(Y_i|X_i)] + (1-alpha) (1/N) Σ_i Σ_j λ_j(X_i)
        
        where \hat{p}_data(y|x) = Σ_k \hat{p}_k \hat{f}_k(x,y) / Σ_k \hat{p}_k \hat{f}_k(x) is the conditional density under the mixture
        
        :param F_train_joint: (n, K) with F_{i,j} = f_j(x_i, y_i) (joint densities)
        :param X_train,Y_train: training data
        :param source_weights: (K,) array of \hat{p}_k = n_k/N weights, if None assumes equal weights
        """
        K, m = self.K, self.m
        n = len(X_train)
        Theta = theta_flat.reshape((K, m))
        A = self.Phi @ Theta.T
        Lambda = softplus_np(A)  # (n,K)
        
        # Set default equal weights if not provided
        if source_weights is None:
            source_weights = np.ones(K) / K
        
        # Compute mixture densities for conditional p_data(y|x)
        # \hat{p}_data(x_i, y_i) = Σ_k \hat{p}_k \hat{f}_k(x_i, y_i)
        p_data_joint = np.sum(source_weights[None, :] * F_train_joint, axis=1)  # (n,)
        
        # For \hat{p}_data(x_i), we need marginal densities from each source
        # This requires computing \hat{f}_k(x_i) for each source k and point i
        # We'll compute this by integrating over y or using the marginal density methods
        p_data_marginal = np.zeros(n)
        for i in range(n):
            x_i = X_train[i:i+1]  # reshape to (1, d)
            marginal_i = 0.0
            for k in range(K):
                # Get marginal density f_k(x_i) from source k
                if hasattr(self, '_sources') and self._sources is not None:
                    src = self._sources[k]
                    if task == 'classification':
                        # For classification: use f_x() method from new SourceModelClassification
                        marginal_k = src.f_x(x_i)[0]
                    else:
                        # For regression: use the marginal PDF method
                        marginal_k = src.marginal_pdf_x(x_i)[0]
                else:
                    # Fallback: estimate marginal from available joint densities
                    # This is an approximation, if we'd have access to source models
                    marginal_k = F_train_joint[i, k] / max(1e-6, np.mean(F_train_joint[:, k]))
                
                marginal_i += source_weights[k] * marginal_k
            p_data_marginal[i] = max(marginal_i, 1e-6)
        
        # Conditional density \hat{p}_data(y_i|x_i) = \hat{p}_data(x_i, y_i) / \hat{p}_data(x_i)
        p_data_conditional = p_data_joint / p_data_marginal
        p_data_conditional = np.maximum(p_data_conditional, 1e-6)  # numerical stability
        
        # Compute h_λ(x_i, y_i) = Σ_j λ_j(x_i) * \hat{f}_j(x_i, y_i)
        h_values = np.sum(Lambda * F_train_joint, axis=1)  # (n,)
        
        # First term: (1/N) Σ_i [(1 - h_λ(X_i,Y_i))_- / \hat{p}_data(Y_i|X_i)]
        hinge_loss = np.minimum(1.0 - h_values, 0.0)   # corrected sign
        term1 = np.mean(hinge_loss / p_data_conditional)
        
        # Second term: (1-alpha) (1/N) Σ_i Σ_j λ_j(X_i)
        term2 = (1.0 - alpha) * np.mean(np.sum(Lambda, axis=1))
        
        # Penalty term
        l2_pen = self._gamma1 * np.mean(np.sum(Lambda**2, axis=1))
        smooth_pen = self._gamma2 * sum(np.sum((self._D @ Theta[j])**2) for j in range(K))
        h_pen = self._gamma3 * np.mean(h_values**2)

        # minus penalty, since we maximize the dual (what we return)
        return term1 + term2 - l2_pen - smooth_pen - h_pen

    def fit_torch(self, sources, X_pool, Y_pool, alpha=0.1,
                epochs=10000, batch_size=256, lr=1e-3,
                device='cpu', source_weights=None, verbose=False,
                tol=1e-4):
        """
        PyTorch-based vectorized fit for LambdaSpline.
        Replaces the scipy.minimize pipeline with minibatch Adam + autograd.
        Updates self.Theta (shape K x m) in-place.

        Relies on:
        - self.Phi (n x m) precomputed in __init__
        - self._D (r x m) difference matrix for smoothness
        - self._gamma1, _gamma2, _gamma3 penalties
        - sources: list of source models (used only to compute marginals)
        - build_F_train_joint_from_sources(...) existing helper to build joint densities.

        Early stopping:
        - Stops when relative change in the full objective falls below ``tol`` (default 1e-4).
        """
        import torch
        import torch.nn.functional as F
        device = torch.device(device)

        # store sources for marginal density computation
        self._sources = sources
        task = 'classification' if isinstance(sources[0], SourceModelClassification) else 'regression'
        n, m = self.Phi.shape
        K = self.K

        # Build F_train_joint like the original fit (n x K numpy)
        F_train_joint = build_F_train_joint_from_sources(sources, X_pool, Y_pool, task)  # numpy
        F_np = np.asarray(F_train_joint, dtype=np.float32)  # (n, K)

        # Source weights (default equal)
        if source_weights is None:
            source_weights = np.ones(K, dtype=np.float32) / float(K)
            if verbose:
                print("Using equal source weights. Consider providing source_weights for better accuracy.")
        source_weights = np.asarray(source_weights, dtype=np.float32)

        # Precompute mixture joint densities and marginal mixture p_data_marginal (vectorized)
        # p_data_joint = sum_k p_k * f_k(x_i, y_i)  (shape n,)
        p_data_joint_np = (F_np * source_weights[None, :]).sum(axis=1)  # (n,)

        # For marginal p_data(x_i) = sum_k p_k * f_k(x_i)  --> call each source's marginal function vectorized
        p_data_marginal_np = np.zeros(n, dtype=np.float32)
        for k, src in enumerate(sources):
            if task == 'classification':
                # classification sources expose f_x(...) returning (q,)
                marg_k = src.f_x(X_pool).astype(np.float32)  # (n,)
            else:
                # regression sources expose marginal_pdf_x
                marg_k = src.marginal_pdf_x(X_pool).astype(np.float32)  # (n,)
            p_data_marginal_np += source_weights[k] * marg_k
        eps = 1e-8
        p_data_marginal_np = np.maximum(p_data_marginal_np, eps)
        p_data_conditional_np = np.maximum(p_data_joint_np / p_data_marginal_np, eps)  # (n,)

        # Move tensors to torch
        Phi_t = torch.tensor(self.Phi.astype(np.float32), device=device)           # (n, m)
        F_t = torch.tensor(F_np, device=device)                                    # (n, K)
        p_data_cond_t = torch.tensor(p_data_conditional_np.astype(np.float32), device=device)  # (n,)
        DT = torch.tensor(self._D.astype(np.float32), device=device)               # (r, m)

        # Initialize Theta as a torch parameter (start from existing self.Theta if available)
        theta_init = np.array(self.Theta, dtype=np.float32) if hasattr(self, 'Theta') else np.zeros((K, m), dtype=np.float32)
        Theta = torch.tensor(theta_init, dtype=torch.float32, device=device, requires_grad=True)  # (K, m)

        # Optimizer
        optimizer = torch.optim.Adam([Theta], lr=lr)

        # Training loop with minibatches
        n_idx = n
        indices = np.arange(n_idx)
        steps_per_epoch = max(1, int(np.ceil(n_idx / batch_size)))

        prev_obj = None
        for epoch in range(epochs):
            # shuffle each epoch
            np.random.shuffle(indices)
            epoch_loss = 0.0
            for step in range(steps_per_epoch):
                start = step * batch_size
                end = min((step + 1) * batch_size, n_idx)
                if start >= end:
                    break
                batch_idx = indices[start:end]
                b = len(batch_idx)

                # batch tensors
                Phi_b = Phi_t[batch_idx, :]            # (b, m)
                F_b = F_t[batch_idx, :]                # (b, K)
                pcond_b = p_data_cond_t[batch_idx]     # (b,)

                # compute A = Phi_b @ Theta.T  -> (b, K)
                A = Phi_b @ Theta.T
                # nonnegativity via softplus (matches original param)
                Lambda = F.softplus(A)                 # (b, K)

                # compute h = sum_j lambda_j(x_i) * f_j(x_i, y_i)
                h = (Lambda * F_b).sum(dim=1)          # (b,)

                # hinge: (1 - h)_- == min(1-h, 0) == -relu(h-1)
                hinge = -F.relu(h - 1.0)               # (b,)

                # term1: mean over batch of hinge / p_data_conditional
                term1 = (hinge / pcond_b).mean()

                # term2: (1-alpha) * mean_i sum_j lambda_j(x_i)
                term2 = (1.0 - alpha) * Lambda.sum(dim=1).mean()

                # penalties
                l2_pen = self._gamma1 * (Lambda**2).sum(dim=1).mean()

                # smooth penalty: sum_j ||D @ Theta_j||^2
                # DT @ Theta.T -> (r, K), then square and sum
                DT_Theta = DT @ Theta.T                 # (r, K)
                smooth_pen = self._gamma2 * (DT_Theta**2).sum()

                # penalty on h
                h_pen = self._gamma3 * (h**2).mean()

                # objective (original returns term1 + term2 - penalties) -> we want to maximize that,
                # so we minimize negative of it
                obj = term1 + term2 - l2_pen - smooth_pen - h_pen
                loss = -obj

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += float(loss.detach().cpu().numpy()) * b

            # average loss for epoch
            epoch_loss = epoch_loss / n_idx
            with torch.no_grad():
                A_full = Phi_t @ Theta.T
                Lambda_full = F.softplus(A_full)
                h_full = (Lambda_full * F_t).sum(dim=1)
                hinge_full = -F.relu(h_full - 1.0)
                term1_full = (hinge_full / p_data_cond_t).mean()
                term2_full = (1.0 - alpha) * Lambda_full.sum(dim=1).mean()
                DT_Theta_full = DT @ Theta.T
                smooth_full = self._gamma2 * (DT_Theta_full**2).sum()
                l2_full = self._gamma1 * (Lambda_full**2).sum(dim=1).mean()
                h_pen_full = self._gamma3 * (h_full**2).mean()
                obj_full = term1_full + term2_full - l2_full - smooth_full - h_pen_full
                obj_value = obj_full.item()

            if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1):
                print(f"[Torch fit] epoch {epoch+1}/{epochs}  loss={epoch_loss:.6g}  obj={obj_value:.6g}")

            if tol is not None and prev_obj is not None:
                denom = max(1e-8, abs(prev_obj))  # turn to abs_tol = 1e-12 when objective value is small
                rel_change = abs(obj_value - prev_obj) / denom
                if rel_change < tol:
                    if verbose:
                        print(f"[Torch fit] early stopping at epoch {epoch+1} (rel_change={rel_change:.3g})")
                    break

            prev_obj = obj_value

        # write Theta back to numpy storage like original class expects
        self.Theta = Theta.detach().cpu().numpy().reshape((K, m))

        res = {
            'success': True,
            'message': 'fit_torch completed',
            'theta': self.Theta
        }
        return res


# --------------------------
# Training pipeline for multi-source aggregation
# --------------------------
def compute_source_weights_from_sizes(source_sizes):
    r"""
    Compute source weights \hat{p}_k = n_k/N from individual source sample sizes.
    
    :param source_sizes: list or array of sample sizes for each source
    :Returns array of normalized weights summing to 1
    """
    source_sizes = np.asarray(source_sizes)
    total_size = np.sum(source_sizes)
    return source_sizes / total_size

def build_F_train_joint_from_sources(sources, X_pool, Y_pool, task):
    """
    Build training matrix with joint densities f_j(x_i, y_i).
    
    :param sources: list of source-model objects (length K)
    :param X_pool,Y_pool: pooled dataset (n, d) and (n,)
    :param task: 'classification' or 'regression'

    :Returns F_train_joint shape (n, K) with F_{i,j} = f_j(x_i, y_i)
    """
    n = X_pool.shape[0]
    K = len(sources)
    F = np.zeros((n, K))
    for j, src in enumerate(sources):
        if task == 'classification':
            F[:, j] = src.joint_prob_at_pairs(X_pool, Y_pool)
        else:
            F[:, j] = src.joint_pdf_at_pairs(X_pool, Y_pool)
    return F

def fit_lambda_from_sources(sources, lambda_mode, X_pool, Y_pool,
                            alpha=0.1, spline_kwargs=None,
                            solver_kwargs=None, verbose=False, source_weights=None,
                            use_torch=True):
    r"""
    Fit lambda(x) parameterization using pooled training data from all sources with marginal loss.

    :param sources: list of per-source SourceModelClassification or SourceModelRegressionGaussian
    :param lambda_mode: 'spline'
    :param source_weights: (K,) array of \hat{p}_k = n_k/N weights for mixture density
    :param spline_kwargs: dict of kwargs for LambdaSpline

    :Returns an object with interface: lambda_at_x(x) -> (q, K) array

    spline_kwargs keys:
    - n_splines: int = 5,
    - degree: int = 3,
    - include_bias: bool = True,
    - diff_order: int = 2,
    - gamma1: float = 0.01, # L2 penalty on lambda
    - gamma2: float = 0.01, # smoothness penalty on Theta
    - gamma3: float = 0, # penalty on h_lambda values
    """
    X_pool = ensure_2d(X_pool)
    n = X_pool.shape[0]
    K = len(sources)

    if lambda_mode == 'spline':
        spline_kwargs = spline_kwargs or {}
        lam = LambdaSpline(X_pool, K, **spline_kwargs)
        solver_kwargs = solver_kwargs or {}
        solver_kwargs['source_weights'] = source_weights

        # if user asked for torch and torch is available, call the PyTorch fit
        if use_torch:
            if not TORCH_AVAILABLE:
                raise RuntimeError("Torch requested (use_torch=True) but TORCH_AVAILABLE is False")
            # fit_torch updates self.Theta in-place and returns a dict-like result;
            # keep the function behavior of returning the lambda object.
            lam.fit_torch(sources, X_pool, Y_pool, alpha=alpha, verbose=verbose, **solver_kwargs)
            return lam
        else:
            if not TORCH_AVAILABLE:
                raise RuntimeError("Torch not available. Install torch or set use_torch=False to use scipy minimize.")
            else:
                raise RuntimeError("use_torch=False is not supported when torch is available. Set use_torch=True.")
    else:
        raise ValueError("lambda_mode not supported, please use 'spline' for spline approximation.")


# --------------------------
# Construct aggregated conformal set for a new x
# --------------------------
def precompute_calibration_cache(
    lam_model,
    sources,
    X_cal_list,
    Y_cal_list,
):
    """Precompute per-source log h values on calibration data for reuse.

    Parameters
    ----------
    lam_model : callable
        Function mapping an array of inputs to lambda weights of shape (n_samples, K).
    sources : list
        List of source models compatible with MDCP.
    X_cal_list, Y_cal_list : list
        Calibration covariates/targets per source.

    Returns
    -------
    dict
        Mapping ``source_index -> {"log_h_cal": np.ndarray, "log_h_cal_sorted": np.ndarray, "n_j": int}``.
    """

    if lam_model is None:
        raise ValueError("lam_model must be provided to precompute calibration cache.")

    K = len(sources)
    calibration_cache = {}

    for j in range(K):
        X_cal_j = np.asarray(X_cal_list[j])
        Y_cal_j = np.asarray(Y_cal_list[j])

        if X_cal_j.ndim == 1:
            X_cal_j = X_cal_j.reshape(-1, X_cal_j.size)

        n_j = X_cal_j.shape[0]
        if n_j <= 0:
            raise ValueError(f"Calibration set for source {j} is empty.")

        lam_cal_all = lam_model(X_cal_j)
        if lam_cal_all.ndim == 1:
            lam_cal_all = lam_cal_all.reshape(n_j, -1)
        if lam_cal_all.shape[0] == 1 and n_j > 1:
            lam_cal_all = np.tile(lam_cal_all, (n_j, 1))
        if lam_cal_all.shape[0] != n_j:
            raise ValueError("lam_model returned unexpected number of rows for calibration cache.")
        if lam_cal_all.shape[1] != K:
            raise ValueError("lam_model returned unexpected number of columns for calibration cache.")

        joint_calib = np.zeros((n_j, K), dtype=float)
        for k, src_k in enumerate(sources):
            if hasattr(src_k, 'joint_pdf_at_pairs'):
                joint_calib[:, k] = src_k.joint_pdf_at_pairs(X_cal_j, Y_cal_j)
            elif hasattr(src_k, 'joint_prob_at_pairs'):
                joint_calib[:, k] = src_k.joint_prob_at_pairs(X_cal_j, Y_cal_j)
            else:
                raise AttributeError(
                    f"Source {k} must have either joint_pdf_at_pairs or joint_prob_at_pairs method"
                )

        h_cal = np.sum(lam_cal_all * joint_calib, axis=1)
        log_h_cal = np.log(np.maximum(h_cal, 1e-100))
        log_h_cal_sorted = np.sort(log_h_cal)

        calibration_cache[j] = {
            "log_h_cal": log_h_cal,
            "log_h_cal_sorted": log_h_cal_sorted,
            "n_j": n_j,
        }

    return calibration_cache


def aggregated_conformal_set_multi(
    lam_model,               # callable: lam_model(X) -> array (n_samples, K) OR None
    sources,                 # list of source objects; each must implement joint_pdf(X, y_grid) for regression or joint_prob(X, y_grid) for classification -> shape (n, m)
    X_cal_list,              # list of np.arrays, one per source j: X_cal_list[j].shape = (n_j, d)
    Y_cal_list,              # list of np.arrays, one per source j: Y_cal_list[j].shape = (n_j,)
    X_test,                  # single test covariate (shape (d,) or (1,d))
    Y_test,                  # true label for test point (for coverage)
    y_grid,                  # 1D array-like of candidate y values (length m)
    alpha=0.1,               # target miscoverage level
    randomize_ties=True,     # whether to randomized-break ties in empirical p-value
    calibration_cache=None,  # optional dict from precompute_calibration_cache
    lam_x=None,              # optional precomputed lambda values at X_test (shape (K,))
    rng=None,                # optional np.random.Generator for tie-breaking
):
    """
    Compute aggregated conformal set via per-source p-values that use the FULL combined score h(y).
    
    Returns
    -------
    p_values_y_grid : np.array shape (K, m) with p_j(y) computed using calibration samples from source j
    
    union_mask_y_grid : boolean array shape (m,) where union_mask_y_grid[idx] == True iff max_j p_j(y) > alpha

    h_y_grid : np.array shape (m,) = h(y) evaluated at X_test

    h_true : float = h(X_test, Y_test) value for true label (scalar for single test point)

    p_values_true_y : np.array shape (K,) with p_j(Y_test) for true label Y_test (single test point)

    union_mask_true_y : boolean, whether true label is in the aggregated conformal set (scalar for single test point)

    Notes
    -----
    - Each source object in `sources` must provide joint_pdf(X, y_grid) for regression or joint_prob(X, y_grid) for classification, returning shape (n, m).
    
    - If lam_model is provided, it should be callable: lam_model(X) -> (n_samples, K).
        If lam_model is None, `lam` must be provided and is used as a constant vector across x.

        - Supplying ``calibration_cache`` (from :func:`precompute_calibration_cache`) skips recomputation of
            the calibration statistics for every test point.

        - ``lam_x`` can be used to pass precomputed lambda values for the current test point to avoid an
            extra call to ``lam_model``.

        - ``rng`` allows sharing a random generator across calls for reproducible tie-breaking without
            repeatedly reinitializing the default generator.
    
    - p_j(y) = EmpiricalProb_{(X,Y) in cal_j} [ h(X, Y) <= h(X_test, y) ]
        with optional randomized tie-breaking for equality.
    """
    K = len(sources)
    y_grid = np.asarray(y_grid)
    m = y_grid.size

    if lam_model is None and lam_x is None:
        raise ValueError("Either lam_model or lam_x must be provided.")

    # Make sure X_test is 2D for calling joint_pdf uniformly (single test point)
    X_test_arr = np.asarray(X_test)
    Y_test_scalar = np.asarray(Y_test)
    
    # Ensure X_test is single test point reshaped to (1, d)
    if X_test_arr.ndim == 1:
        X_test_2d = X_test_arr.reshape(1, -1)
    elif X_test_arr.ndim == 2 and X_test_arr.shape[0] == 1:
        X_test_2d = X_test_arr
    else:
        raise ValueError(f"X_test must be a single test point, got shape {X_test_arr.shape}")
    
    # Ensure Y_test is scalar
    if Y_test_scalar.ndim > 0 and Y_test_scalar.size == 1:
        Y_test_scalar = Y_test_scalar.item()
    elif Y_test_scalar.ndim == 0:
        Y_test_scalar = Y_test_scalar.item()
    else:
        raise ValueError(f"Y_test must be a single scalar value, got shape {Y_test_scalar.shape}")

    # -- Compute f_grid_test: per-source densities at (X_test, y_grid)
    # f_grid_test shape: (K, m)
    f_grid_test = np.zeros((K, m), dtype=float)
    joint_true = np.zeros(K, dtype=float)  # Single test point, shape (K,)
    y_grid_with_true = np.concatenate([y_grid, np.array([Y_test_scalar])])
    for k, src in enumerate(sources):
        # Handle both classification (joint_prob) and regression (joint_pdf) sources
        if hasattr(src, 'joint_pdf'):
            vals = src.joint_pdf(X_test_2d, y_grid_with_true)  # shape (1, m+1) expected
        elif hasattr(src, 'joint_prob'):
            vals = src.joint_prob(X_test_2d, y_grid_with_true)  # shape (1, m+1) expected
        else:
            raise AttributeError(f"Source {k} must have either joint_pdf or joint_prob method")

        if vals.ndim == 1:
            vals = vals.reshape(1, -1)
        f_grid_test[k, :] = vals[0, :m]
        joint_true[k] = vals[0, m]

    # -- Compute lam at X_test (1, K) then h_y_grid (m,)
    if lam_x is not None:
        lam_test = np.asarray(lam_x)
        if lam_test.ndim > 1:
            lam_test = lam_test.reshape(-1)
        if lam_test.shape[0] != K:
            raise ValueError("lam_x must have length equal to number of sources K.")
    else:
        lam_test = lam_model(X_test_2d)  # shape (1, K)
        if lam_test.ndim == 1:
            lam_test = lam_test.reshape(1, -1)
        lam_test = lam_test[0]  # Extract to shape (K,)
    
    # h at X_test: h(y) = sum_k lam_k(X_test) * f_k(X_test, y)
    h_y_grid = np.dot(lam_test, f_grid_test)  # shape (m,)
    h_true = np.sum(lam_test * joint_true)    # scalar for single test point

    # Prepare output container: p-values per source and per y
    p_values_y_grid = np.zeros((K, m), dtype=float)
    p_values_true_y = np.zeros(K, dtype=float)  # Single test point, so shape (K,)

    # Precompute log h values once
    log_h_test = np.log(np.maximum(h_y_grid, 1e-100))
    log_h_true_scalar = np.log(np.maximum(h_true, 1e-100))

    # RNG for randomized ties
    rng_local = rng if rng is not None else np.random.default_rng(RANDOM_SEED)
    atol = 1e-6
    rtol = 1e-4

    # -- For each source j, compute p_j(y) using h evaluated on source j calibration samples
    for j in range(K):
        X_cal_j = np.asarray(X_cal_list[j])
        Y_cal_j = np.asarray(Y_cal_list[j])
        if X_cal_j.ndim == 1:
            X_cal_j = X_cal_j.reshape(-1, X_cal_j.size)  # attempt to fix shape
        n_j = X_cal_j.shape[0]
        if n_j <= 0:
            raise ValueError(f"Calibration set for source {j} is empty.")

        cache_entry = None
        if calibration_cache is not None:
            cache_entry = calibration_cache.get(j)
            if cache_entry is not None:
                log_h_cal = np.asarray(cache_entry.get("log_h_cal"))
                cache_n_j = int(cache_entry.get("n_j", log_h_cal.shape[0]))
                if log_h_cal.shape[0] != cache_n_j or cache_n_j != n_j:
                    cache_entry = None

        if cache_entry is None:
            # lam evaluated at calibration X_cal_j: shape (n_j, K)
            if lam_model is None:
                raise ValueError(
                    "lam_model must be provided when calibration_cache lacks precomputed values."
                )
            lam_cal_all = lam_model(X_cal_j)
            if lam_cal_all.ndim == 1:
                lam_cal_all = lam_cal_all.reshape(n_j, -1)
            if lam_cal_all.shape[0] == 1 and n_j > 1:
                lam_cal_all = np.tile(lam_cal_all, (n_j, 1))
            if lam_cal_all.shape[0] != n_j or lam_cal_all.shape[1] != K:
                raise ValueError("lam_model returned unexpected shape for calibration X.")

            joint_calib = np.zeros((n_j, K), dtype=float)
            for k, src_k in enumerate(sources):
                if hasattr(src_k, 'joint_pdf_at_pairs'):
                    joint_calib[:, k] = src_k.joint_pdf_at_pairs(X_cal_j, Y_cal_j)
                elif hasattr(src_k, 'joint_prob_at_pairs'):
                    joint_calib[:, k] = src_k.joint_prob_at_pairs(X_cal_j, Y_cal_j)
                else:
                    raise AttributeError(
                        f"Source {k} must have either joint_pdf_at_pairs or joint_prob_at_pairs method"
                    )

            # now compute h_cal (n_j,) = sum_k lam_cal_all[i,k] * joint_calib[i,k]
            h_cal = np.sum(lam_cal_all * joint_calib, axis=1)   # shape (n_j,)
            log_h_cal = np.log(np.maximum(h_cal, 1e-100))  # avoid -inf

            log_h_cal_sorted = np.sort(log_h_cal)
            if calibration_cache is not None:
                calibration_cache[j] = {
                    "log_h_cal": log_h_cal,
                    "log_h_cal_sorted": log_h_cal_sorted,
                    "n_j": n_j,
                }
        else:
            log_h_cal = np.asarray(cache_entry["log_h_cal"])
            log_h_cal_sorted = np.asarray(cache_entry.get("log_h_cal_sorted"))
            if log_h_cal_sorted is None or log_h_cal_sorted.size != log_h_cal.size:
                log_h_cal_sorted = np.sort(log_h_cal)
                if calibration_cache is not None:
                    cache_entry["log_h_cal_sorted"] = log_h_cal_sorted
            n_j = int(cache_entry.get("n_j", log_h_cal.shape[0]))

        inv_denom = 1.0 / (n_j + 1)

        # Vectorized counts for less-than and tie handling via tolerance bands
        if randomize_ties:
            u = rng_local.random(size=m)
        else:
            u = np.ones(m)

        less_counts = np.searchsorted(log_h_cal_sorted, log_h_test, side='left')
        tol = atol + rtol * np.abs(log_h_test)
        lower_bounds = log_h_test - tol
        upper_bounds = log_h_test + tol
        lower_idx = np.searchsorted(log_h_cal_sorted, lower_bounds, side='left')
        upper_idx = np.searchsorted(log_h_cal_sorted, upper_bounds, side='right')
        eq_counts = np.maximum(0, upper_idx - lower_idx)
        p_values_y_grid[j, :] = (1.0 + less_counts + u * eq_counts) * inv_denom

        # Compute p-value for true label (single test point)
        if randomize_ties:
            u_true = rng_local.random()
        else:
            u_true = 1.0

        tol_true = atol + rtol * abs(log_h_true_scalar)
        less_true = np.searchsorted(log_h_cal_sorted, log_h_true_scalar, side='left')
        lower_true = log_h_true_scalar - tol_true
        upper_true = log_h_true_scalar + tol_true
        lower_true_idx = np.searchsorted(log_h_cal_sorted, lower_true, side='left')
        upper_true_idx = np.searchsorted(log_h_cal_sorted, upper_true, side='right')
        eq_true = max(0, upper_true_idx - lower_true_idx)
        p_values_true_y[j] = (1.0 + less_true + u_true * eq_true) * inv_denom


    # Aggregated set via max-p (union), represented by boolean masks
    union_mask_y_grid = np.any(p_values_y_grid > alpha, axis=0)   # shape (m,)
    union_mask_true_y = np.any(p_values_true_y > alpha)           # scalar boolean for single test point

    return {
        "p_values_y_grid": p_values_y_grid,               # shape (K, m)
        "union_mask_y_grid": union_mask_y_grid,           # shape (m,) boolean, True if included
        "h_y_grid": h_y_grid,                             # shape (m,)
        "y_grid": y_grid,                                 # shape (m,), full grid of candidate y values
        "h_true": h_true,                                 # scalar for single test point
        "p_values_true_y": p_values_true_y,               # shape (K,) for single test point
        "union_mask_true_y": union_mask_true_y,           # scalar boolean for single test point
    }
