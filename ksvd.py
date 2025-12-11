"""
Simple K-SVD implementation for real-valued data.
Uses OMP from sklearn for sparse coding.
This implementation alternates:
 - Sparse coding (OMP) given dictionary D
 - Atom update via SVD on residuals (classic K-SVD)
Warning: Not heavily optimized. Suitable for small-medium problems.
"""

import numpy as np
from sklearn.linear_model import orthogonal_mp_gram
from sklearn.linear_model import orthogonal_mp
from sklearn.utils import check_random_state
from joblib import Parallel, delayed

class KSVDDictLearner:
    def __init__(self, n_atoms=512, sparsity=6, n_iter=10, random_state=None, n_jobs=1):
        self.n_atoms = n_atoms
        self.sparsity = sparsity
        self.n_iter = n_iter
        self.random_state = check_random_state(random_state)
        self.n_jobs = n_jobs
        self.D = None

    def _initialize_dictionary(self, X):
        # X shape: (n_samples, n_features)
        n_samples, n_features = X.shape
        # initialize by randomly selecting data samples as atoms and normalize
        idx = self.random_state.choice(n_samples, self.n_atoms, replace=False)
        D = X[idx].T  # (n_features, n_atoms)
        D = D / (np.linalg.norm(D, axis=0, keepdims=True) + 1e-12)
        return D

    def _omp(self, D, X):
        # returns coefficient matrix Gamma of shape (n_atoms, n_samples)
        # using sklearn's orthogonal_mp (fits y = D * x ; note sklearn uses shape (n_samples_features, n_atoms)?)
        # We'll use orthogonal_mp which expects D (n_features, n_atoms) and X (n_samples, n_features) => solves for coef (n_samples, n_atoms)
        # To maintain consistency: X is (n_samples, n_features)
        # Return Gamma as (n_atoms, n_samples)
        # sklearn's orthogonal_mp handles batch via loop or multiprocessing; use orthogonal_mp for each sample (Parallel)
        D_for_omp = D.copy()  # (n_features, n_atoms)
        DtD = D_for_omp.T.dot(D_for_omp)
        DtX = D_for_omp.T.dot(X.T)  # (n_atoms, n_samples)
        # Use orthogonal_mp_gram for speed if memory allows (use Gram matrix)
        # orthogonal_mp_gram expects Gram (n_atoms,n_atoms) and Xy (n_atoms, n_samples)
        Gamma = orthogonal_mp_gram(DtD, DtX, n_nonzero_coefs=self.sparsity)
        # Gamma shape: (n_atoms, n_samples)
        return Gamma

    def fit(self, X):
        """
        X: (n_samples, n_features) -- note transposed relative to many K-SVD implementations
        We'll return D (n_features, n_atoms)
        """
        n_samples, n_features = X.shape
        # initialize D
        D = self._initialize_dictionary(X)  # (n_features, n_atoms)
        for it in range(self.n_iter):
            # Sparse coding step (OMP)
            Gamma = self._omp(D, X)  # (n_atoms, n_samples)
            # Update atoms
            for k in range(self.n_atoms):
                wk = Gamma[k, :]  # (n_samples,)
                nonzero_idx = np.where(np.abs(wk) > 1e-12)[0]
                if nonzero_idx.size == 0:
                    # replace the atom with a random sample
                    rnd = self.random_state.randint(0, n_samples)
                    D[:, k] = X[rnd] / (np.linalg.norm(X[rnd]) + 1e-12)
                    continue
                # compute error excluding the k-th atom contribution
                # E = X^T - sum_{j != k} d_j * wk_j
                Dk = D.copy()
                Dk[:, k] = 0
                # reconstruct for selected samples only
                Gamma_sub = Gamma[:, nonzero_idx]  # (n_atoms, m)
                X_sub = X[nonzero_idx].T  # (n_features, m)
                # residual = X_sub - Dk * Gamma_sub
                residual = X_sub - (Dk.dot(Gamma_sub))
                # SVD on residual to update atom k
                U, s, Vt = np.linalg.svd(residual, full_matrices=False)
                D[:, k] = U[:, 0]
                # update the coefficients of atom k
                Gamma[k, nonzero_idx] = s[0] * Vt[0, :]
            # optional: normalize dictionary
            D = D / (np.linalg.norm(D, axis=0, keepdims=True) + 1e-12)
            print(f"K-SVD iteration {it+1}/{self.n_iter} done.")
        self.D = D
        return self

    def transform(self, X):
        """Compute sparse codes for X using OMP with learned dictionary D.
        X: (n_samples, n_features) -> returns Gamma (n_samples, n_atoms)
        """
        if self.D is None:
            raise ValueError("Call fit() first.")
        # orthogonal_mp returns coef matrix shape (n_samples, n_atoms)
        from sklearn.linear_model import orthogonal_mp
        coef = orthogonal_mp(self.D, X.T, n_nonzero_coefs=self.sparsity)
        return coef.T  # (n_samples, n_atoms)
