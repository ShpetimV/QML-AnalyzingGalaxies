"""
Quantum kernel methods for SVM classification (PennyLane + scikit-learn).
"""

import numpy as np
import pennylane as qml
from sklearn.svm import SVC
from sklearn.metrics.pairwise import rbf_kernel


def _make_device(n_qubits):
    """Claim the GPU if available, else fall back to a CPU statevector sim."""
    try:
        dev = qml.device("lightning.gpu", wires=n_qubits)
    except Exception:
        try:
            dev = qml.device("lightning.qubit", wires=n_qubits)
        except Exception:
            dev = qml.device("default.qubit", wires=n_qubits)
    return dev


class QuantumSVM:
    """
    Quantum-kernel SVM with a scikit-learn precomputed-kernel SVC backend.
    """

    def __init__(self, n_features=8, kernel_type="fidelity", c_param=1.0,
                 reps=1, entanglement="ring", angle_bound=np.pi / 2,
                 squash="tanh", double_axis=True, pqk_gamma="median",
                 bandwidth=1.0):
        self.n_qubits = n_features
        self.kernel_type = kernel_type
        self.reps = reps
        self.entanglement = entanglement
        self.angle_bound = angle_bound
        self.squash = squash
        self.double_axis = double_axis
        self.pqk_gamma = pqk_gamma
        self.bandwidth = bandwidth   # data scale before encoding

        self.dev = _make_device(self.n_qubits)
        print(f"Initialized QuantumSVM ({kernel_type}) with {self.n_qubits} "
              f"qubits on device: {self.dev.name}")

        self.svm = SVC(kernel="precomputed", C=c_param, probability=True,
                       random_state=42)
        self.x_train_fit = None          # raw (post-scaler) training features
        self._proj_train = None          # cached projected train features (PQK)
        self._gamma_value = None         # resolved RBF bandwidth (PQK)

        # ---- quantum circuits -------------------------------------------------
        @qml.qnode(self.dev)
        def fidelity_circuit(x1, x2):
            self._feature_map(x1)
            qml.adjoint(self._feature_map)(x2)
            return qml.probs(wires=range(self.n_qubits))

        @qml.qnode(self.dev)
        def projection_circuit(x):
            self._feature_map(x)
            obs = (
                [qml.expval(qml.PauliX(i)) for i in range(self.n_qubits)]
                + [qml.expval(qml.PauliY(i)) for i in range(self.n_qubits)]
                + [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
            )
            return obs

        @qml.qnode(self.dev)
        def state_circuit(x):
            self._feature_map(x)
            return qml.state()

        self._fidelity_circuit = fidelity_circuit
        self._projection_circuit = projection_circuit
        self._state_circuit = state_circuit
        self.use_statevector = True

    # ---- encoding -------------------------------------------------------------
    def _to_angles(self, x):
        """Map a feature vector into a bounded angle range (no 2*pi wrap-around)."""
        x = np.asarray(x, dtype=float) * self.bandwidth
        if self.squash == "tanh":
            return self.angle_bound * np.tanh(x)
        elif self.squash == "clip":
            return np.clip(x, -self.angle_bound, self.angle_bound)
        elif self.squash == "none":
            return x
        raise ValueError(f"Unknown squash mode: {self.squash}")

    def _feature_map(self, x):
        """Bounded, configurable data-encoding circuit."""
        for _ in range(self.reps):
            qml.AngleEmbedding(x, wires=range(self.n_qubits), rotation="Y")
            if self.double_axis:
                qml.AngleEmbedding(x, wires=range(self.n_qubits), rotation="Z")
            if self.entanglement == "ring":
                for i in range(self.n_qubits):
                    qml.CNOT(wires=[i, (i + 1) % self.n_qubits])
            elif self.entanglement == "linear":
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
            # "none" -> no entangling layer

    # ---- shape helpers --------------------------------------------------------
    def _as_2d(self, x):
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        return x

    def _check_dim(self, X):
        if X.shape[1] != self.n_qubits:
            raise ValueError(f"Expected {self.n_qubits} features, got {X.shape[1]}.")

    # ---- fidelity kernel ------------------------------------------------------
    def _statevectors(self, X):
        """One circuit per sample -> array of statevectors, shape (N, 2**n)."""
        return np.array([np.asarray(self._state_circuit(self._to_angles(r)))
                         for r in X])

    def _fidelity_matrix(self, X1, X2):
        if self.use_statevector:
            try:
                psi1 = self._statevectors(X1)
                psi2 = self._statevectors(X2)
                return np.abs(psi1 @ psi2.conj().T) ** 2
            except Exception:
                pass  # fall back to the overlap-circuit loop below

        n1, n2 = len(X1), len(X2)
        K = np.zeros((n1, n2))
        symmetric = (X1.shape == X2.shape) and np.array_equal(X1, X2)
        A = np.array([self._to_angles(r) for r in X1])
        B = np.array([self._to_angles(r) for r in X2])
        for i in range(n1):
            for j in range(n2):
                if symmetric and j < i:
                    K[i, j] = K[j, i]
                else:
                    K[i, j] = float(self._fidelity_circuit(A[i], B[j])[0])
        return K

    # ---- projected (PQK) kernel ----------------------------------------------
    def _projected_features(self, X):
        """Return the (N, 3*n_qubits) table of local Pauli expectations."""
        feats = []
        for r in X:
            vals = self._projection_circuit(self._to_angles(r))
            feats.append(np.array(vals, dtype=float).ravel())
        return np.vstack(feats)

    def projected_features(self, X):
        """Public: quantum-projected feature table (X,Y,Z expectations per qubit)"""
        X = self._as_2d(X)
        self._check_dim(X)
        return self._projected_features(X)

    def _resolve_gamma(self, proj):
        if isinstance(self.pqk_gamma, (int, float)):
            return float(self.pqk_gamma)
        # median heuristic on pairwise squared distances of projected features
        n = len(proj)
        if n < 2:
            return 1.0
        diffs = proj[:, None, :] - proj[None, :, :]
        sq = np.sum(diffs ** 2, axis=-1)
        iu = np.triu_indices(n, k=1)
        med = np.median(sq[iu])
        return 1.0 / med if med > 0 else 1.0

    # ---- public kernel matrix -------------------------------------------------
    def kernel_matrix(self, X1, X2):
        """Gram matrix between X1 and X2 for the configured kernel."""
        X1 = self._as_2d(X1); X2 = self._as_2d(X2)
        self._check_dim(X1); self._check_dim(X2)
        if self.kernel_type == "fidelity":
            return self._fidelity_matrix(X1, X2)
        elif self.kernel_type == "projected":
            p1 = self._projected_features(X1)
            p2 = self._projected_features(X2)
            return rbf_kernel(p1, p2, gamma=self._gamma_value)
        raise ValueError(f"Unknown kernel_type: {self.kernel_type}")

    # ---- sklearn-style API ----------------------------------------------------
    def fit(self, x_train, y_train):
        x_train = self._as_2d(x_train); self._check_dim(x_train)
        y_train = np.asarray(y_train).ravel()
        self.x_train_fit = x_train
        if self.kernel_type == "projected":
            self._proj_train = self._projected_features(x_train)
            self._gamma_value = self._resolve_gamma(self._proj_train)
            K = rbf_kernel(self._proj_train, self._proj_train, gamma=self._gamma_value)
        else:
            K = self._fidelity_matrix(x_train, x_train)
        self.svm.fit(K, y_train)
        return self

    def _test_kernel(self, x_test):
        x_test = self._as_2d(x_test); self._check_dim(x_test)
        if self.kernel_type == "projected":
            p_test = self._projected_features(x_test)
            return rbf_kernel(p_test, self._proj_train, gamma=self._gamma_value)
        return self._fidelity_matrix(x_test, self.x_train_fit)

    def predict(self, x_test):
        return self.svm.predict(self._test_kernel(x_test))

    def predict_proba(self, x_test):
        return self.svm.predict_proba(self._test_kernel(x_test))

    # convenience: build the train Gram matrix without fitting (for diagnostics)
    def train_gram(self, x_train):
        x_train = self._as_2d(x_train); self._check_dim(x_train)
        if self.kernel_type == "projected":
            proj = self._projected_features(x_train)
            gamma = self._resolve_gamma(proj)
            return rbf_kernel(proj, proj, gamma=gamma)
        return self._fidelity_matrix(x_train, x_train)


def geometric_difference(K_classical, K_quantum, reg=1e-8):
    """Compute the geometric difference between two kernel matrices, as defined in"""
    Kc = np.asarray(K_classical, dtype=float)
    Kq = np.asarray(K_quantum, dtype=float)
    N = Kc.shape[0]
    Kc = N * Kc / np.trace(Kc)
    Kq = N * Kq / np.trace(Kq)

    w, V = np.linalg.eigh(Kq)
    w = np.clip(w, 0.0, None)
    sqrt_Kq = (V * np.sqrt(w)) @ V.T

    Kc_inv = np.linalg.inv(Kc + reg * np.eye(N))
    M = sqrt_Kq @ Kc_inv @ sqrt_Kq
    return float(np.sqrt(np.linalg.norm(M, ord=2)))