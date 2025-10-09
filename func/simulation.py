import numpy as np

def _triangular_safe(o: float, m: float, p: float, size: int, rng: np.random.Generator) -> np.ndarray:
    # Asegurar orden y modo válido
    left = min(o, m, p)
    right = max(o, m, p)
    mode = min(max(m, left), right)
    if right == left:
        return np.full(size, mode)
    return rng.triangular(left=left, mode=mode, right=right, size=size)

def _beta_pert_safe(o: float, m: float, p: float, size: int, rng: np.random.Generator, lam: float = 4.0) -> np.ndarray:
    # Ordenar para asegurar o<=m<=p
    o_, m_, p_ = sorted([o, m, p])
    if p_ == o_:
        return np.full(size, m_)
    alpha = 1 + lam * (m_ - o_) / (p_ - o_ + 1e-12)
    beta = 1 + lam * (p_ - m_) / (p_ - o_ + 1e-12)
    # Evitar parámetros inválidos
    alpha = max(alpha, 1e-6)
    beta = max(beta, 1e-6)
    x = rng.beta(alpha, beta, size=size)
    return o_ + x * (p_ - o_)

def _sample_activity(o: float, m: float, p: float, size: int, dist: str, rng: np.random.Generator) -> np.ndarray:
    if dist == "triangular":
        return _triangular_safe(o, m, p, size, rng)
    elif dist == "beta-pert":
        return _beta_pert_safe(o, m, p, size, rng)
    else:
        raise ValueError("Distribución no soportada.")
