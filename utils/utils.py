import numpy as np
from typing import Optional, Tuple, Dict, Any, List

def _logit(p: float) -> float:
    p = float(np.clip(p, 1e-6, 1 - 1e-6))
    return float(np.log(p / (1 - p)))


def sigmoid_time_weights(t_win: int, *, w_min: float = 0.1, w_max: float = 0.8) -> np.ndarray:
    if t_win < 2:
        raise ValueError(f"t_win must be >= 2, got {t_win}")
    b = _logit(w_max)
    a = (b - _logit(w_min)) / float(t_win - 1)
    d = np.arange(-(t_win - 1), 1, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-(a * d + b)))


def _soft_mask_from_power(power_1d: np.ndarray, *, energy_ratio: float, eps: float = 1e-12) -> np.ndarray:
    power_1d = np.asarray(power_1d, dtype=np.float64)
    k_sc = int(power_1d.shape[0])
    total = float(power_1d.sum())
    if total <= eps:
        L = k_sc - 1
    else:
        L = int(np.searchsorted(np.cumsum(power_1d), energy_ratio * total, side="left"))
    L = int(np.clip(L, 0, min(k_sc - 1, 29)))
    mask = (np.arange(k_sc) <= L).astype(np.float64)
    tail = min(4, k_sc - L - 1)
    if tail > 0:
        mask[L + 1 : L + 1 + tail] = 0.5 * (1 + np.cos(np.linspace(0, np.pi, tail)))
    return mask, L


def gadf_from_signal_1d(signal_1d: np.ndarray, *, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(signal_1d, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError(f"signal_1d must be 1D, got shape={x.shape}")
    x_min = float(x.min())
    x_max = float(x.max())
    if x_max - x_min > eps:
        x_norm = (x - x_min) / (x_max - x_min + eps)
    else:
        x_norm = np.zeros_like(x, dtype=np.float64)
    phi = np.arccos(x_norm)
    diff = phi[:, None] - phi[None, :]
    # amplified_diff = np.sign(diff) * np.power(np.abs(diff), 1)
    return np.sin(diff)


# def gadf_from_signal_1d(signal_1d: np.ndarray, *, eps: float = 1e-8) -> np.ndarray:
#     """
#     Use STFT instead of GADF:
#     - Input: 1D signal (length K)
#     - Output: 2D matrix (K, K) (interpolate/scale STFT time frames to K)
#     """
#     x = np.asarray(signal_1d, dtype=np.float64)
#     alpha = 0.97
#     x = np.append(x[0], x[1:] - alpha * x[:-1])
#     if x.ndim != 1:
#         raise ValueError(f"signal_1d must be 1D, got shape={x.shape}")
#     K = int(x.shape[0])
#     if K == 0:
#         raise ValueError("signal_1d length cannot be 0")

#     # --------- STFT parameters (internal only, I/O shape unchanged) ----------
#     win_len = min(K, max(8, K // 8))   # window length
#     hop = max(1, win_len // 4)         # hop length
#     window = np.hanning(win_len).astype(np.float64)

#     # Compute required frames and pad to cover the last frame
#     n_frames = 1 if K <= win_len else int(np.ceil((K - win_len) / hop)) + 1
#     total_len = (n_frames - 1) * hop + win_len
#     if total_len > K:
#         x_pad = np.pad(x, (0, total_len - K), mode="constant")
#     else:
#         x_pad = x

#     frames = np.empty((n_frames, win_len), dtype=np.float64)
#     for i in range(n_frames):
#         start = i * hop
#         frames[i] = x_pad[start:start + win_len]

#     frames *= window[None, :]
#     X = np.fft.fft(frames, n=K, axis=1)      # (n_frames, K)
#     S = np.abs(X).T                          # (K, n_frames)  frequency x time

#     # Compress dynamic range + normalize to [0,1]

#     s_min, s_max = float(S.min()), float(S.max())
#     if s_max - s_min > eps:
#         S = (S - s_min) / (s_max - s_min + eps)
#     else:
#         S = np.zeros((K, n_frames), dtype=np.float64)
    
#     S = np.power(S, 0.5)

#     # --------- Scale time frames to K, producing (K, K) output ----------
#     if n_frames == 1:
#         out = np.repeat(S, K, axis=1)  # (K, K)
#     else:
#         old_t = np.linspace(0.0, 1.0, n_frames)
#         new_t = np.linspace(0.0, 1.0, K)
#         out = np.stack([np.interp(new_t, old_t, S[f]) for f in range(K)], axis=0)

#     return out.astype(np.float64)


def psee(
    H_tilde: np.ndarray,
    *,
    energy_ratio: float = 0.95,
    apply_ifftshift: bool = True,
    w_min: float = 0.1,
    w_max: float = 0.8,
    eps: float = 1e-12,
    return_intermediates: bool = False,
) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
    H_tilde = np.asarray(H_tilde)
    if H_tilde.ndim != 2:
        raise ValueError(f"H_tilde must be a 2D matrix (T,K), got shape={H_tilde.shape}")
    if not (0.0 < float(energy_ratio) <= 1.0):
        raise ValueError(f"energy_ratio must be in (0,1], got {energy_ratio!r}")
    t_win: int = 52

    T, K = H_tilde.shape
    weights = sigmoid_time_weights(t_win, w_min=w_min, w_max=w_max).astype(np.float64)
    H_env = np.zeros((T, K), dtype=np.complex128)
    H_bg = np.zeros((T, K), dtype=np.complex128)
    mask = np.zeros((T, K), dtype=np.float64)

    h_hist: List[np.ndarray] = []
    H_env_hist: List[np.ndarray] = []
    l_list = []
    for t in range(T):
        H_t = H_tilde[t]
        H_for_ifft = np.fft.ifftshift(H_t) if apply_ifftshift else H_t
        h_t = np.fft.ifft(H_for_ifft)

        h_hist.append(h_t)
        if len(h_hist) > t_win:
            h_hist.pop(0)

        h_stack = np.asarray(h_hist)
        len_hist = h_stack.shape[0]
        w_used = weights[-len_hist:]
        w_sum = float(w_used.sum()) + eps
        power_weighted = (np.abs(h_stack) ** 2 * w_used[:, None]).sum(axis=0) / w_sum
        mask_t, L_list_tmp = _soft_mask_from_power(power_weighted, energy_ratio=energy_ratio, eps=eps)
        mask[t] = mask_t
        l_list.append(L_list_tmp)
        h_env_t = h_t * mask_t
        H_env_t = np.fft.fft(h_env_t)
        H_env_t = np.fft.fftshift(H_env_t) if apply_ifftshift else H_env_t
        H_env[t] = H_env_t
        H_env_hist.append(H_env_t)
        if len(H_env_hist) > t_win:
            H_env_hist.pop(0)
        H_env_stack = np.asarray(H_env_hist)
        len_hist2 = H_env_stack.shape[0]
        w_used2 = weights[-len_hist2:]
        H_bg[t] = np.sum(H_env_stack * w_used2[:, None], axis=0) / (np.sum(w_used2) + eps)


    if not return_intermediates:
        return H_bg, None
    return H_bg, {"mask": mask, "H_env": H_env, "weights": weights}


def pswr(
    H_tilde: np.ndarray,
    H_bg: np.ndarray,
    *,
    eps: float = 1e-12
) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
    H_tilde = np.asarray(H_tilde)
    H_bg = np.asarray(H_bg)
    if H_tilde.shape != H_bg.shape:
        raise ValueError(
            f"H_tilde and H_bg must have the same shape (T,K), got {H_tilde.shape} vs {H_bg.shape}"
        )
    if H_tilde.ndim != 2:
        raise ValueError(f"H_tilde must be a 2D matrix (T,K), got shape={H_tilde.shape}")

    E_tk = H_tilde - H_bg
    abs_E = np.abs(E_tk)
    gadf_list = [gadf_from_signal_1d(abs_E[t], eps=eps) for t in range(abs_E.shape[0])]
    gadf = np.stack(gadf_list, axis=0).astype(np.float64)

    return gadf, abs_E

def cgs(
    data_fixed: np.ndarray,
    *,
    energy_ratio: float = 0.99,
    apply_ifftshift: bool = True,
    w_min: float = 0.1,
    w_max: float = 0.8,
    eps: float = 1e-12,
) -> List[np.ndarray]:
    data_fixed = np.asarray(data_fixed)
    if data_fixed.ndim != 2:
        raise ValueError(f"data_fixed must be 2D (K, n_frames), got shape={data_fixed.shape}")
    H_tilde = data_fixed.T
    H_bg, _ = psee(
        H_tilde,
        energy_ratio=energy_ratio,
        apply_ifftshift=apply_ifftshift,
        w_min=w_min,
        w_max=w_max,
        eps=eps
    )
    gadf_stack, abs_E = pswr(
        H_tilde,
        H_bg,
        eps=eps
    )
    # return [gadf_stack[t].astype(np.float64, copy=False) for t in range(gadf_stack.shape[0])]
    return gadf_stack.astype(np.float64), abs_E
