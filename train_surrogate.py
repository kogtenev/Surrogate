#!/usr/bin/env python3
"""
Мультипольный суррогат взаимных S-параметров системы антенн, обучаемый по модулям |S_ij| в дБ.

Модель (для одной частоты), для каждой пары антенн i != j:

    S_ij = u_j^T  G(r_j - r_i)  u_i,      u_i = D(R_i) a_{t(i)},

  a_t  -- коэффициенты антенн типа t при функциях  d^alpha delta(x - r_i) e_m  (|alpha| <= MAX_ORDER,
          m = x, y, z) в собственной системе координат антенны; комплексные, ищутся подгонкой.
          По умолчанию у каждой антенны свой тип (ANTENNA_TYPES = None);
  D(R) -- поворот коэффициентов (каждый тензорный индекс поворачивается матрицей R);
  G    -- матрица связи: G[(beta,n),(alpha,m)] = (-1)^|beta| d^(alpha+beta) G_nm(d),
          G(d) = (I + grad grad / k^2) exp(ik|d|) / (4 pi |d|) -- функция Грина;
          расстояния безразмерные (k*d), постоянные множители входят в a.
  По желанию -- изображения в металлической плоскости под антеннами (GROUND_PLANE_DEPTH_MM).
  По взаимности S_ij = S_ji: обе строки (если есть) описываются одним значением модели.

Фазы в данных нет, поэтому подгоняется 20 lg|S_ij| к S_dB методом Левенберга-Марквардта
(нормальные уравнения собираются блоками по парам антенн) с несколькими случайными стартами и продолжением по порядку
мультиполей (0 -> 1 -> 2). Значения ниже THRESHOLD_DB не подгоняются: штраф только если модель
предсказывает для них больше порога. Диагональные S_ii не используются (их может не быть).

Обучение: eval_id % 3 == 1 (eval_id = 1, 4, 7, ...), тест -- остальные. Порядок мультиполей
(<= MAX_ORDER) и регуляризация выбираются кросс-валидацией только по обучающей выборке; тест
используется только для итоговой оценки. Знак угла (по или против часовой) выбирается по
функции потерь на обучающей выборке, если THETA_SIGN = 0.

Формат данных:
  геометрия: eval_id, ignore, ignore, x0_mm, y0_mm, th0_deg, x1_mm, y1_mm, th1_deg, x2_mm, ...
             (столбцы x{i}_mm, y{i}_mm, th{i}_deg ищутся по именам; если имен нет -- тройками
             после первых трех столбцов);
  S:         eval_id, freq_idx, freq_GHz, ant_i, ant_j, S_ij_dB  (ant_i -- номер i из геометрии;
             нумерация с 1 тоже распознается).

Запуск:
    python3 train_surrogate.py                          # файлы из POOLS ниже
    python3 train_surrogate.py g1.csv s1.csv g2.csv s2.csv g3.csv s3.csv
"""
import os
import re
import sys
import time

import numpy as np
import pandas as pd

# ============================== Настройки ==============================
POOLS = [  # (файл геометрии, файл S-параметров) для каждого пула
    ("pool1_geom.csv", "pool1_s.csv"),
    ("pool2_geom.csv", "pool2_s.csv"),
    ("pool3_geom.csv", "pool3_s.csv"),
]
FREQ_IDX = 0                 # индекс частоты, для которой обучается модель
THRESHOLD_DB = -35.0         # S ниже этого уровня не важны
MAX_ORDER = 2                # 0: delta (диполи), 1: + производные, 2: + вторые производные (до квадруполей)
ANTENNA_TYPES = None         # None -- у каждой антенны свои коэффициенты; "same" -- все антенны
                             # одинаковые; или список номеров типов по антеннам, напр. [0, 0, 1, 1, ...]
SHARED_ACROSS_POOLS = True   # True -- во всех пулах те же антенны (одна модель на все пулы)
THETA_SIGN = 0               # +1: угол против часовой стрелки, -1: по часовой, 0: выбрать по обучающей выборке
GROUND_PLANE_DEPTH_MM = None # глубина металлической плоскости под центрами антенн (мм) или None
SCREEN_RECT_MM = None        # края экрана/подложки (xmin, xmax, ymin, ymax) для диагностики; None --
                             # вместо них прямоугольник, охватывающий центры антенн в пуле
DO_CV = True                 # выбирать порядок и регуляризацию кросс-валидацией
REG_GRID = [0.1, 1.0, 10.0]  # кандидаты регуляризации (гребневая, растет с порядком)
REG_DEFAULT = 1e-1           # регуляризация без кросс-валидации и при выборе знака угла
CV_FOLDS = 3                 # блоков кросс-валидации (по конфигурациям)
N_RESTARTS = 6               # случайных стартов на нулевом порядке
CV_RESTARTS = 2              # то же при кросс-валидации
MAX_ITER = 100               # итераций Левенберга-Марквардта на одну подгонку
SEED = 0
OUT_DIR = "surrogate_out"
C_MM_PER_NS = 299.792458     # скорость света, мм/нс (f в ГГц -> длина волны в мм)
# ======================================================================

TRAIN_MOD = 3                # обучающая выборка: eval_id % 3 == 1
DB = 20.0 / np.log(10.0)
CHUNK = 4000                 # пар за раз при вычислении матриц связи

# ---------------------------------------------------------------------
#  Базис:  d^alpha delta e_m,  alpha -- мультииндекс (n_x, n_y, n_z), |alpha| <= 2
# ---------------------------------------------------------------------
MULTI = [(0, 0, 0),
         (1, 0, 0), (0, 1, 0), (0, 0, 1),
         (2, 0, 0), (1, 1, 0), (1, 0, 1), (0, 2, 0), (0, 1, 1), (0, 0, 2)]
BASIS = [(a, m) for a in MULTI for m in range(3)]          # 30 функций
BASIS_ORDER = np.array([sum(a) for a, _ in BASIS])
NB = len(BASIS)


def _second_order_pairs():
    """Пары (l, n), l <= n, в порядке MULTI[4:]."""
    pairs = []
    for a in MULTI[4:]:
        idx = [i for i in range(3) for _ in range(a[i])]
        pairs.append((idx[0], idx[1]))
    return pairs


PAIRS2 = _second_order_pairs()


def rotation_z(theta):
    """Матрицы поворота вокруг z, theta -- массив углов (рад)."""
    c, s = np.cos(theta), np.sin(theta)
    R = np.zeros(theta.shape + (3, 3))
    R[..., 0, 0], R[..., 0, 1], R[..., 1, 0], R[..., 1, 1], R[..., 2, 2] = c, -s, s, c, 1.0
    return R


def basis_rotation(R):
    """D(R): поворот коэффициентов базиса, R -- (..., 3, 3) -> (..., 30, 30).

    Источник sum c_{l..m} d_l.. delta e_m в локальных осях; в глобальных каждый индекс
    тензора c поворачивается матрицей R. Для второго порядка храним b_{(ln),m}, l <= n:
    b_{(ll)} = c_{ll}, b_{(ln)} = c_{ln} + c_{nl} = 2 c_{ln}.
    """
    shp = R.shape[:-2]
    D = np.zeros(shp + (NB, NB))
    D[..., 0:3, 0:3] = R
    D[..., 3:12, 3:12] = (R[..., :, None, :, None] * R[..., None, :, None, :]).reshape(shp + (9, 9))
    # второй порядок: b (18) -> c (27) -> R x R x R -> b
    to_full = np.zeros((27, 18))
    from_full = np.zeros((18, 27))
    for p, (l, n) in enumerate(PAIRS2):
        for m in range(3):
            col = 3 * p + m
            if l == n:
                to_full[(l * 3 + n) * 3 + m, col] = 1.0
                from_full[col, (l * 3 + n) * 3 + m] = 1.0
            else:
                to_full[(l * 3 + n) * 3 + m, col] = 0.5
                to_full[(n * 3 + l) * 3 + m, col] = 0.5
                from_full[col, (l * 3 + n) * 3 + m] = 2.0
    R3 = (R[..., :, None, None, :, None, None] * R[..., None, :, None, None, :, None]
          * R[..., None, None, :, None, None, :]).reshape(shp + (27, 27))
    D[..., 12:30, 12:30] = from_full @ R3 @ to_full
    return D


# ---------------------------------------------------------------------
#  Производные Phi = exp(i r) / (4 pi r) (безразмерные координаты, k = 1)
# ---------------------------------------------------------------------
# Для радиальной функции: d_i f_n(r) = x_i f_{n+1}(r), f_{n+1} = (1/r) d f_n / dr. Поэтому
# d^gamma Phi = sum coef * x^beta * f_n(r), а f_n = exp(i r)/(4 pi) * sum_j c_{n,j} r^-j.
MAXD = 6  # порядок производных Phi: 2 (G) + 2 + 2 (мультиполи обеих антенн)


def _radial_coeffs(nmax):
    coeffs = [{1: 1.0 + 0j}]
    for _ in range(nmax):
        nxt = {}
        for j, c in coeffs[-1].items():
            nxt[j + 1] = nxt.get(j + 1, 0) + 1j * c       # (1/r) d/dr e^{ir} r^-j = e^{ir} (i r^{-j-1} - j r^{-j-2})
            nxt[j + 2] = nxt.get(j + 2, 0) - j * c
        coeffs.append(nxt)
    return coeffs


def _derivative_terms(maxd):
    """{gamma: {(beta, n): coef}} для всех |gamma| <= maxd."""
    terms = {(0, 0, 0): {((0, 0, 0), 0): 1.0}}
    order = [(0, 0, 0)]
    for total in range(1, maxd + 1):
        for gx in range(total, -1, -1):
            for gy in range(total - gx, -1, -1):
                g = (gx, gy, total - gx - gy)
                i = next(d for d in range(3) if g[d] > 0)
                base = list(g)
                base[i] -= 1
                src = terms[tuple(base)]
                out = {}
                for (beta, n), c in src.items():
                    if beta[i] > 0:                         # d_i x^beta
                        b2 = list(beta)
                        b2[i] -= 1
                        key = (tuple(b2), n)
                        out[key] = out.get(key, 0) + c * beta[i]
                    b3 = list(beta)                         # x^beta * x_i f_{n+1}
                    b3[i] += 1
                    key = (tuple(b3), n + 1)
                    out[key] = out.get(key, 0) + c
                terms[g] = out
                order.append(g)
    return terms, order


RADIAL = _radial_coeffs(MAXD + 1)
DTERMS, GAMMAS = _derivative_terms(MAXD)
GAMMA_INDEX = {g: i for i, g in enumerate(GAMMAS)}


def _phi_tables():
    """Матрица коэффициентов: d^gamma Phi = sum_{(beta,n)} C[gamma,(beta,n)] x^beta f_n(r)."""
    feats = sorted({key for g in GAMMAS for key in DTERMS[g]})
    fidx = {key: q for q, key in enumerate(feats)}
    C = np.zeros((len(GAMMAS), len(feats)))
    for gi, g in enumerate(GAMMAS):
        for key, c in DTERMS[g].items():
            C[gi, fidx[key]] = c
    betas = np.array([b for b, _ in feats])
    ns = np.array([n for _, n in feats])
    return C, betas, ns


PHI_C, PHI_BETA, PHI_N = _phi_tables()


def phi_derivatives(X):
    """X: (N, 3) безразмерные точки -> (N, n_gamma) значения d^gamma Phi."""
    r = np.linalg.norm(X, axis=1)
    e = np.exp(1j * r) / (4 * np.pi)
    f = np.stack([e * sum(c * r ** (-j) for j, c in RADIAL[n].items()) for n in range(MAXD + 2)], axis=1)
    pw = np.stack([X ** p for p in range(MAXD + 2)], axis=2)          # (N, 3, степень)
    mono = pw[:, 0, PHI_BETA[:, 0]] * pw[:, 1, PHI_BETA[:, 1]] * pw[:, 2, PHI_BETA[:, 2]]
    return (mono * f[:, PHI_N]) @ PHI_C.T


def _coupling_index():
    """Индексы для сборки G[(beta,n),(alpha,m)] = (-1)^|beta| (delta_nm d^g Phi + d^(g+e_n+e_m) Phi)."""
    i1 = np.zeros((NB, NB), dtype=int)
    i2 = np.zeros((NB, NB), dtype=int)
    diag = np.zeros((NB, NB))
    sign = np.zeros((NB, NB))
    for r, (beta, n) in enumerate(BASIS):
        for c, (alpha, m) in enumerate(BASIS):
            g = tuple(alpha[d] + beta[d] for d in range(3))
            g2 = list(g)
            g2[n] += 1
            g2[m] += 1
            i1[r, c] = GAMMA_INDEX[g]
            i2[r, c] = GAMMA_INDEX[tuple(g2)]
            diag[r, c] = 1.0 if n == m else 0.0
            sign[r, c] = (-1.0) ** sum(beta)
    return i1, i2, diag, sign


CI1, CI2, CDIAG, CSIGN = _coupling_index()


def coupling_matrix(dk):
    """dk: (N, 3) безразмерные векторы r_1 - r_0 -> (N, 30, 30) матрицы связи."""
    P = phi_derivatives(dk)
    return CSIGN * (CDIAG * P[:, CI1] + P[:, CI2])




# ---------------------------------------------------------------------
#  Данные
# ---------------------------------------------------------------------
def parse_geometry(geom, gfile):
    """-> (eval_id, метки антенн, X, Y, TH) -- массивы (n_conf, n_ant)."""
    geom.columns = [str(c).strip() for c in geom.columns]
    cols = list(geom.columns)
    eid = geom["eval_id"] if "eval_id" in cols else geom.iloc[:, 0]
    labels = sorted(int(m.group(1)) for c in cols for m in [re.fullmatch(r"x(\d+)_mm", c)] if m)
    if labels and all(f"y{i}_mm" in cols and f"th{i}_deg" in cols for i in labels):
        X = geom[[f"x{i}_mm" for i in labels]].to_numpy(float)
        Y = geom[[f"y{i}_mm" for i in labels]].to_numpy(float)
        TH = geom[[f"th{i}_deg" for i in labels]].to_numpy(float)
    else:  # без имен: тройки (x, y, th) после eval_id и двух служебных столбцов
        rest = geom.iloc[:, 3:]
        if rest.shape[1] % 3:
            raise SystemExit(f"{gfile}: не удалось разобрать столбцы геометрии {cols}")
        n = rest.shape[1] // 3
        labels = list(range(n))
        X = rest.iloc[:, 0::3].to_numpy(float)
        Y = rest.iloc[:, 1::3].to_numpy(float)
        TH = rest.iloc[:, 2::3].to_numpy(float)
    return eid.to_numpy(), labels, X, Y, TH


def load_pools(files, freq_idx):
    """Наблюдения S_ij (i != j) со всеми пулами: DataFrame + число антенн."""
    obs_all, n_ant = [], None
    geometry = {}
    for pid, (gfile, sfile) in enumerate(files):
        eid, labels, X, Y, TH = parse_geometry(pd.read_csv(gfile), gfile)
        row_of = {e: r for r, e in enumerate(eid)}
        s = pd.read_csv(sfile)
        s.columns = [str(c).strip() for c in s.columns]
        s = s[s["freq_idx"] == freq_idx]
        if s.empty:
            raise SystemExit(f"{sfile}: нет строк с freq_idx = {freq_idx}")
        n_diag = int((s["ant_i"] == s["ant_j"]).sum())
        s = s[s["ant_i"] != s["ant_j"]]
        # метки антенн в S -> номера столбцов геометрии
        lab = set(s["ant_i"]).union(s["ant_j"])
        pos = {l: q for q, l in enumerate(labels)}
        if lab <= set(labels):
            amap = pos
        elif {l - 1 for l in lab} <= set(labels):
            amap = {l: pos[l - 1] for l in lab}
            print(f"  {sfile}: номера антенн с 1, сдвинуты на -1")
        else:
            raise SystemExit(f"{sfile}: номера антенн {sorted(lab)} не соответствуют геометрии {labels}")
        missing = set(s["eval_id"]) - set(row_of)
        if missing:
            print(f"  {sfile}: нет геометрии для {len(missing)} eval_id -- строки пропущены")
            s = s[s["eval_id"].isin(row_of.keys())]
        r = s["eval_id"].map(row_of).to_numpy()
        i = s["ant_i"].map(amap).to_numpy()
        j = s["ant_j"].map(amap).to_numpy()
        df = pd.DataFrame({
            "pool": pid, "eval_id": s["eval_id"].to_numpy(), "ant_i": s["ant_i"].to_numpy(),
            "ant_j": s["ant_j"].to_numpy(), "i": i, "j": j, "freq_GHz": s["freq_GHz"].to_numpy(float),
            "S_ij_dB": s["S_ij_dB"].to_numpy(float),
            "xi": X[r, i], "yi": Y[r, i], "thi": TH[r, i], "xj": X[r, j], "yj": Y[r, j], "thj": TH[r, j]})
        print(f"  пул {pid + 1}: {gfile}, {sfile}: конфигураций {len(eid)}, антенн {len(labels)}, "
              f"строк S_ij (i != j) {len(df)}" + (f", пропущено S_ii {n_diag}" if n_diag else ""))
        if n_ant is not None and n_ant != len(labels) and SHARED_ACROSS_POOLS:
            raise SystemExit("в пулах разное число антенн -- поставьте SHARED_ACROSS_POOLS = False")
        n_ant = len(labels)
        obs_all.append(df)
        geometry[pid] = (row_of, X, Y, TH)
    obs = pd.concat(obs_all, ignore_index=True)
    freqs = obs["freq_GHz"].unique()
    if np.ptp(freqs) > 1e-9 * freqs.max():
        raise SystemExit(f"для freq_idx = {freq_idx} разные частоты: {freqs}")
    obs["train"] = (obs["eval_id"] % TRAIN_MOD) == 1
    obs["group"] = obs["pool"].astype(str) + ":" + obs["eval_id"].astype(str)
    obs["dist_mm"] = np.hypot(obs["xj"] - obs["xi"], obs["yj"] - obs["yi"])
    obs.attrs["geometry"] = geometry
    return obs, n_ant, float(freqs[0])


def build_pairs(obs):
    """Уникальные пары (конфигурация, {i, j}) -- одна на S_ij и S_ji. Пары ориентированы lo -> hi."""
    lo = np.minimum(obs["i"], obs["j"]).to_numpy()
    hi = np.maximum(obs["i"], obs["j"]).to_numpy()
    key = (obs["group"] + "|" + pd.Series(lo, index=obs.index).astype(str) + "|"
           + pd.Series(hi, index=obs.index).astype(str)).to_numpy()
    _, first, inv = np.unique(key, return_index=True, return_inverse=True)
    sw = (obs["i"].to_numpy() > obs["j"].to_numpy())[first]  # строка-представитель записана как (hi, lo)
    o = obs.iloc[first]
    pick = lambda a, b: np.where(sw, o[b].to_numpy(), o[a].to_numpy())
    pairs = pd.DataFrame({
        "group": o["group"].to_numpy(), "train": o["train"].to_numpy(), "lo": lo[first], "hi": hi[first],
        "x_lo": pick("xi", "xj"), "y_lo": pick("yi", "yj"), "th_lo": pick("thi", "thj"),
        "x_hi": pick("xj", "xi"), "y_hi": pick("yj", "yi"), "th_hi": pick("thj", "thi")})
    return pairs, inv


def configuration_matrices(pairs, k, theta_sign):
    """M_p = D(R_hi)^T [G(d) - G(d_img) D(P)] D(R_lo), d = r_hi - r_lo  -> (n_pairs, 30, 30) complex64."""
    n = len(pairs)
    M = np.empty((n, NB, NB), dtype=np.complex64)
    DP = basis_rotation(np.diag([1.0, 1.0, -1.0]))     # отражение; ток изображения -P J
    for a in range(0, n, CHUNK):
        b = min(n, a + CHUNK)
        q = pairs.iloc[a:b]
        r0 = np.stack([q["x_lo"], q["y_lo"], np.zeros(b - a)], axis=1)
        r1 = np.stack([q["x_hi"], q["y_hi"], np.zeros(b - a)], axis=1)
        D0 = basis_rotation(rotation_z(np.deg2rad(q["th_lo"].to_numpy(float)) * theta_sign))
        D1 = basis_rotation(rotation_z(np.deg2rad(q["th_hi"].to_numpy(float)) * theta_sign))
        G = coupling_matrix(k * (r1 - r0))
        if GROUND_PLANE_DEPTH_MM is not None:
            r0img = r0.copy()
            r0img[:, 2] = -2.0 * GROUND_PLANE_DEPTH_MM
            G = G - coupling_matrix(k * (r1 - r0img)) @ DP
        M[a:b] = np.matmul(np.matmul(D1.transpose(0, 2, 1), G), D0)
    return M


class Problem:
    """Подмножество пар и наблюдений для подгонки."""

    def __init__(self, M, t0, t1, obs_pair, y):
        self.M, self.t0, self.t1, self.obs_pair, self.y = M, t0, t1, obs_pair, y
        self._block = (None, None)

    def block(self, act):
        """Блок M на активных функциях (кэшируется): (n_pairs, p, p) complex64."""
        if self._block[0] is None or len(self._block[0]) != len(act) or np.any(self._block[0] != act):
            Mb = self.M if len(act) == NB else np.ascontiguousarray(self.M[:, act][:, :, act])
            self._block = (act.copy(), Mb)
        return self._block[1]

    @staticmethod
    def subset(M, t0, t1, inv, y, obs_mask):
        pairs_used = np.unique(inv[obs_mask])
        remap = -np.ones(len(M), dtype=int)
        remap[pairs_used] = np.arange(len(pairs_used))
        return Problem(M[pairs_used], t0[pairs_used], t1[pairs_used], remap[inv[obs_mask]], y[obs_mask])


# ---------------------------------------------------------------------
#  Модель и подгонка
# ---------------------------------------------------------------------
class Model:
    """S_pair = a_{t1}^T M a_{t0}; коэффициенты на активных функциях (порядок <= order)."""

    def __init__(self, order, n_types):
        self.order, self.T = order, n_types
        self.act = np.where(BASIS_ORDER <= order)[0]
        self.p = len(self.act)
        self.A = np.zeros((n_types, NB), complex)

    def pack(self):
        a = self.A[:, self.act]
        return np.concatenate([a.real, a.imag], axis=1).ravel()

    def unpack(self, x):
        v = x.reshape(self.T, 2 * self.p)
        self.A = np.zeros((self.T, NB), complex)
        self.A[:, self.act] = v[:, :self.p] + 1j * v[:, self.p:]

    def s(self, prob):
        A = self.A[:, self.act].astype(np.complex64)
        Mb = prob.block(self.act)
        g0 = np.matmul(A[prob.t1][:, None, :], Mb)[:, 0, :]
        return np.sum(g0 * A[prob.t0], axis=1).astype(complex)

    def predict_db(self, prob):
        return DB * np.log(np.abs(self.s(prob))[prob.obs_pair] + 1e-300)


def thresholded_residual(p, y, thr):
    return np.where(y >= thr, p - y, np.maximum(p - thr, 0.0))


def residuals_and_blocks(model, x, prob, op, y, thr):
    """Невязки (дБ) и блоки якобиана по коэффициентам типов t0 и t1 каждого наблюдения.

    d(dB)/d a = (20/ln10) Re(conj(s) ds) / |s|^2;  ds/dRe a = g, ds/dIm a = i g.
    Возвращает res (n), J0, J1 (n, 2p): производные по [Re a_t, Im a_t] для t = t0 и t = t1."""
    model.unpack(x)
    A = model.A[:, model.act].astype(np.complex64)
    a0, a1 = A[prob.t0], A[prob.t1]
    Mb = prob.block(model.act)
    g0 = np.matmul(a1[:, None, :], Mb)[:, 0, :]       # ds/da_{t0}
    s = np.sum(g0 * a0, axis=1).astype(complex)
    g1 = np.matmul(Mb, a0[:, :, None])[:, :, 0]       # ds/da_{t1}
    so = s[op]
    mdb = DB * np.log(np.abs(so) + 1e-300)
    res = thresholded_residual(mdb, y, thr)
    active = (y >= thr) | (mdb > thr)
    w = (DB / (np.abs(so) ** 2 + 1e-300) * active)[:, None]
    sc = np.conj(so)[:, None]
    h0 = sc * g0[op]
    h1 = sc * g1[op]
    J0 = np.hstack([w * h0.real, -w * h0.imag])
    J1 = np.hstack([w * h1.real, -w * h1.imag])
    return res, J0, J1


def fit(model, prob, thr, reg, max_iter=MAX_ITER, tol=1e-7):
    """Левенберг-Марквардт. Нормальные уравнения собираются плотными блоками по парам типов
    (t0, t1): размер системы P = 2 p T, а не число наблюдений."""
    from scipy.linalg import cho_factor, cho_solve, LinAlgError
    q = 2 * model.p
    per = np.sqrt(reg) * (1.0 + BASIS_ORDER[model.act])
    reg_w2 = np.tile(np.concatenate([per, per]), model.T) ** 2
    # наблюдения, отсортированные по паре типов
    ot0, ot1 = prob.t0[prob.obs_pair], prob.t1[prob.obs_pair]
    perm = np.lexsort((ot1, ot0))
    op, y = prob.obs_pair[perm], prob.y[perm]
    ta, tb = ot0[perm], ot1[perm]
    cut = np.flatnonzero((np.diff(ta) != 0) | (np.diff(tb) != 0)) + 1
    bounds = np.concatenate([[0], cut, [len(perm)]])
    groups = [(a, b, ta[a], tb[a]) for a, b in zip(bounds[:-1], bounds[1:])]
    P = model.T * q

    def cost_of(x, res):
        return 0.5 * (res @ res + reg_w2 @ (x * x))

    x = model.pack()
    res, J0, J1 = residuals_and_blocks(model, x, prob, op, y, thr)
    cost = cost_of(x, res)
    mu = 1e-3
    for _ in range(max_iter):
        H = np.zeros((P, P))
        g = np.zeros(P)
        for a, b, t0, t1 in groups:
            r = res[a:b]
            s0, s1 = slice(t0 * q, (t0 + 1) * q), slice(t1 * q, (t1 + 1) * q)
            if t0 == t1:
                Jc = J0[a:b] + J1[a:b]
                H[s0, s0] += Jc.T @ Jc
                g[s0] += Jc.T @ r
            else:
                A0, A1 = J0[a:b], J1[a:b]
                H[s0, s0] += A0.T @ A0
                H[s1, s1] += A1.T @ A1
                X = A0.T @ A1
                H[s0, s1] += X
                H[s1, s0] += X.T
                g[s0] += A0.T @ r
                g[s1] += A1.T @ r
        H[np.diag_indices(P)] += reg_w2
        g += reg_w2 * x
        dH = np.diag(H).copy()
        improved = False
        while mu < 1e10:
            try:
                d = -cho_solve(cho_factor(H + mu * np.diag(dH + 1e-12 * dH.max())), g)
            except LinAlgError:
                mu *= 10
                continue
            xn = x + d
            rn, J0n, J1n = residuals_and_blocks(model, xn, prob, op, y, thr)
            cn = cost_of(xn, rn)
            if cn < cost:
                rel = (cost - cn) / max(cost, 1e-300)
                x, res, J0, J1, cost = xn, rn, J0n, J1n, cn
                mu = max(mu / 3, 1e-9)
                improved = True
                break
            mu *= 4
        if not improved or rel < tol:
            break
    model.unpack(x)
    return cost


def scale_to_data(model, prob, thr):
    """Общий масштаб коэффициентов: медиана предсказания = медиана данных (выше порога)."""
    big = prob.y >= thr
    d = np.median(prob.y[big] - model.predict_db(prob)[big])
    model.A *= np.sqrt(10 ** (d / 20.0))


def train(prob, n_types, thr, rng, reg, n_restarts=N_RESTARTS, refine_restarts=2, log=print):
    """Продолжение по порядку: 0 -> ... -> MAX_ORDER; на нулевом порядке -- несколько стартов.
    Возвращает [(порядок, функция потерь, модель)]."""
    log = log or (lambda *a: None)
    best = None
    for _ in range(n_restarts):
        m = Model(0, n_types)
        m.A[:, m.act] = rng.normal(size=(n_types, m.p)) + 1j * rng.normal(size=(n_types, m.p))
        scale_to_data(m, prob, thr)
        cost = fit(m, prob, thr, reg)
        if best is None or cost < best[0]:
            best = (cost, m)
    history = [(0, best[0], best[1])]
    log(f"    порядок 0: функция потерь {best[0]:.4g} (лучший из {n_restarts} стартов)")
    prev = best[1]
    for order in range(1, MAX_ORDER + 1):
        cand = None
        new = np.where(BASIS_ORDER == order)[0]
        for rs in range(refine_restarts):
            m = Model(order, n_types)
            m.A = prev.A.copy()
            if rs > 0:
                amp = 0.05 * np.abs(prev.A[:, prev.act]).mean()
                m.A[:, new] = amp * (rng.normal(size=(n_types, len(new))) + 1j * rng.normal(size=(n_types, len(new))))
            cost = fit(m, prob, thr, reg)
            if cand is None or cost < cand[0]:
                cand = (cost, m)
        history.append((order, cand[0], cand[1]))
        log(f"    порядок {order}: функция потерь {cand[0]:.4g}")
        prev = cand[1]
    return history


def cross_validate(M, t0, t1, inv, y, groups, train_mask, n_types, thr, rng):
    """Выбор (порядок, регуляризация) по RMS ошибки на отложенных блоках конфигураций
    обучающей выборки."""
    ug = np.unique(groups[train_mask])
    fold_of = dict(zip(ug, rng.permutation(len(ug)) % CV_FOLDS))
    fold = np.array([fold_of.get(g, -1) for g in groups])
    table = {}
    for reg in REG_GRID:
        sse = np.zeros(MAX_ORDER + 1)
        cnt = 0
        for f in range(CV_FOLDS):
            trn = train_mask & (fold != f)
            val = train_mask & (fold == f)
            ptr = Problem.subset(M, t0, t1, inv, y, trn)
            pva = Problem.subset(M, t0, t1, inv, y, val)
            hist = train(ptr, n_types, thr, rng, reg, n_restarts=CV_RESTARTS, refine_restarts=1, log=None)
            for order, _, model in hist:
                sse[order] += np.sum(thresholded_residual(model.predict_db(pva), pva.y, thr) ** 2)
            cnt += val.sum()
        table[reg] = np.sqrt(sse / cnt)
    reg, order = min(((r, o) for r in REG_GRID for o in range(MAX_ORDER + 1)), key=lambda t: table[t[0]][t[1]])
    return order, reg, table


# ---------------------------------------------------------------------
#  Оценка
# ---------------------------------------------------------------------
def metrics(y, p, thr):
    big = y >= thr
    small = ~big
    err = p[big] - y[big]
    ok = np.zeros_like(y, dtype=bool)
    ok[big] = np.abs(err) <= 3.0
    ok[small] = p[small] <= thr + 3.0          # маленькие значения: достаточно не завысить
    return {
        "n": len(y), "n_big": int(big.sum()),
        "mae": np.mean(np.abs(err)) if err.size else np.nan,
        "rmse": np.sqrt(np.mean(err ** 2)) if err.size else np.nan,
        "max": np.max(np.abs(err)) if err.size else np.nan,
        "within3": np.mean(np.abs(err) <= 3.0) if err.size else np.nan,
        "pass": ok.mean() if len(y) else np.nan,
    }


def fmt(mt):
    return (f"MAE {mt['mae']:5.2f} дБ, RMS {mt['rmse']:5.2f}, max {mt['max']:5.1f}, "
            f"|ошибка|<=3 дБ: {100 * mt['within3']:5.1f}% (из {mt['n_big']} выше порога), "
            f"годных с учетом порога: {100 * mt['pass']:5.1f}% (из {mt['n']})")


def distance_baseline(r, y, thr, train_mask, free_slope):
    """S_dB = A - B * 10 lg r  (B = 2 -- закон 1/r)."""
    lr = 10 * np.log10(r)
    big = (y >= thr) & train_mask
    if free_slope:
        A = np.vstack([np.ones(big.sum()), -lr[big]]).T
        c = np.linalg.lstsq(A, y[big], rcond=None)[0]
        return c[0] - c[1] * lr
    return np.mean(y[big] + 2 * lr[big]) - 2 * lr


def antenna_types(n_ant):
    if ANTENNA_TYPES is None:
        return np.arange(n_ant)
    if isinstance(ANTENNA_TYPES, str) and ANTENNA_TYPES == "same":
        return np.zeros(n_ant, dtype=int)
    t = np.asarray(ANTENNA_TYPES, dtype=int)
    if len(t) != n_ant:
        raise SystemExit(f"ANTENNA_TYPES: длина {len(t)}, а антенн {n_ant}")
    _, t = np.unique(t, return_inverse=True)
    return t


def run(obs, n_ant, freq, tag, rng):
    k = 2 * np.pi * freq / C_MM_PER_NS
    lam = 2 * np.pi / k
    y = obs["S_ij_dB"].to_numpy(float)
    tr = obs["train"].to_numpy()
    r = obs["dist_mm"].to_numpy()
    groups = obs["group"].to_numpy()
    pairs, inv = build_pairs(obs)
    types = antenna_types(n_ant)
    T = int(types.max()) + 1
    t0 = types[pairs["lo"].to_numpy()]
    t1 = types[pairs["hi"].to_numpy()]
    print(f"\n=== {tag}: f = {freq:g} ГГц, lambda = {lam:.2f} мм; антенн {n_ant}, типов {T}")
    print(f"  наблюдений: обучение {tr.sum()}, тест {(~tr).sum()}; уникальных пар {len(pairs)} "
          f"(матрицы связи {len(pairs) * NB * NB * 8 / 2 ** 30:.2f} ГБ); конфигураций: обучение "
          f"{len(np.unique(groups[tr]))}, тест {len(np.unique(groups[~tr]))}")
    print(f"  расстояния {r.min():.1f}..{r.max():.1f} мм ({r.min() / lam:.2f}..{r.max() / lam:.2f} lambda); "
          f"выше {THRESHOLD_DB:g} дБ: {100 * np.mean(y >= THRESHOLD_DB):.0f}% наблюдений")
    print("  параметров модели (порядок " + "/".join(str(o) for o in range(MAX_ORDER + 1)) + "): "
          + "/".join(str(2 * T * int((BASIS_ORDER <= o).sum())) for o in range(MAX_ORDER + 1)))

    # 1. Знак угла
    signs = [THETA_SIGN] if THETA_SIGN in (1, -1) else [1, -1]
    results = {}
    for sgn in signs:
        t_s = time.time()
        print(f"  знак угла {sgn:+d} (регуляризация {REG_DEFAULT:g}):")
        M = configuration_matrices(pairs, k, sgn)
        prob = Problem.subset(M, t0, t1, inv, y, tr)
        hist = train(prob, T, THRESHOLD_DB, rng, REG_DEFAULT)
        print(f"    ({time.time() - t_s:.0f} с)")
        results[sgn] = (hist[-1][1], hist)
        if len(signs) > 1 and sgn == signs[0]:
            del M  # память: матрицы для второго знака
    sgn = min(results, key=lambda s: results[s][0])
    if len(signs) > 1:
        print(f"  выбран знак угла {sgn:+d} (функция потерь {results[sgn][0]:.4g} против {results[-sgn][0]:.4g})")
        M = configuration_matrices(pairs, k, sgn) if sgn == signs[0] else M
    hist = results[sgn][1]
    order, reg = MAX_ORDER, REG_DEFAULT

    # 2. Порядок и регуляризация: кросс-валидация внутри обучающей выборки
    if DO_CV:
        t_s = time.time()
        order, reg, table = cross_validate(M, t0, t1, inv, y, groups, tr, T, THRESHOLD_DB, rng)
        print(f"  кросс-валидация ({CV_FOLDS} блока, {time.time() - t_s:.0f} с), RMS ошибки с учетом порога, дБ:")
        print("    регуляризация " + "".join(f"  порядок {o}" for o in range(MAX_ORDER + 1)))
        for rg in REG_GRID:
            print(f"    {rg:<13g} " + "".join(f"  {table[rg][o]:9.2f}" for o in range(MAX_ORDER + 1)))
        print(f"  выбрано: порядок {order}, регуляризация {reg:g}")
        if len(REG_GRID) > 1 and reg in (min(REG_GRID), max(REG_GRID)):
            print("  (выбрана крайняя регуляризация сетки -- имеет смысл расширить REG_GRID)")
        if reg != REG_DEFAULT:
            print(f"  обучение с регуляризацией {reg:g}:")
            hist = train(Problem.subset(M, t0, t1, inv, y, tr), T, THRESHOLD_DB, rng, reg)

    # 3. Ошибки
    allp = Problem.subset(M, t0, t1, inv, y, np.ones(len(y), bool))
    print("  Ошибки (обучение / тест):")
    for name, pred in [("1/r, подгонка уровня", distance_baseline(r, y, THRESHOLD_DB, tr, False)),
                       ("A - B*10lg r", distance_baseline(r, y, THRESHOLD_DB, tr, True))]:
        print(f"    {name:<26} обуч.: {fmt(metrics(y[tr], pred[tr], THRESHOLD_DB))}")
        print(f"    {'':<26} тест : {fmt(metrics(y[~tr], pred[~tr], THRESHOLD_DB))}")
    preds = {}
    for o, _, model in hist:
        pred = model.predict_db(allp)
        preds[o] = pred
        name = f"мультиполи, порядок {o}" + (" (*)" if o == order else "")
        print(f"    {name:<26} обуч.: {fmt(metrics(y[tr], pred[tr], THRESHOLD_DB))}")
        print(f"    {'':<26} тест : {fmt(metrics(y[~tr], pred[~tr], THRESHOLD_DB))}")
    if DO_CV:
        print("    (*) -- выбрано кросс-валидацией; тестовая выборка в выборе не участвует")
    final = preds[order]
    if obs["pool"].nunique() > 1:
        print(f"  Тест по пулам (порядок {order}):")
        for pid in sorted(obs["pool"].unique()):
            msk = (~tr) & (obs["pool"].to_numpy() == pid)
            print(f"    пул {pid + 1}: {fmt(metrics(y[msk], final[msk], THRESHOLD_DB))}")
    print("  Тест по расстоянию:")
    edges = np.array([0, 0.5, 1, 2, 4, np.inf]) * lam
    for a, b in zip(edges[:-1], edges[1:]):
        msk = (~tr) & (r >= a) & (r < b)
        if msk.any():
            name = f"{a / lam:.1f}..{b / lam:.1f} lambda" if np.isfinite(b) else f">= {a / lam:.1f} lambda"
            print(f"    {name:<16} {fmt(metrics(y[msk], final[msk], THRESHOLD_DB))}")
    return final, hist[order][2], sgn, k, types


def plots(obs, pred, path, thr):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    y = obs["S_ij_dB"].to_numpy(float)
    te = ~obs["train"].to_numpy()
    r = obs["dist_mm"].to_numpy()
    fig, ax = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)
    lo = min(y.min(), pred.min()) - 2
    hi = max(y.max(), pred.max()) + 2
    for pid in sorted(obs["pool"].unique()):
        m = te & (obs["pool"].to_numpy() == pid)
        ax[0].scatter(y[m], pred[m], s=5, alpha=0.4, label=f"пул {pid + 1}")
    ax[0].plot([lo, hi], [lo, hi], "k-", lw=1)
    ax[0].plot([lo, hi], [lo + 3, hi + 3], "k--", lw=0.7)
    ax[0].plot([lo, hi], [lo - 3, hi - 3], "k--", lw=0.7)
    ax[0].axvline(thr, color="r", lw=0.8)
    ax[0].axhline(thr, color="r", lw=0.8)
    ax[0].set_xlabel("|S_ij| расчет, дБ")
    ax[0].set_ylabel("|S_ij| модель, дБ")
    ax[0].set_title("Тест: модель против расчета (±3 дБ пунктиром)")
    ax[0].legend()
    big = te & (y >= thr)
    ax[1].scatter(r[big], pred[big] - y[big], s=5, alpha=0.4)
    ax[1].axhline(3, color="k", ls="--", lw=0.7)
    ax[1].axhline(-3, color="k", ls="--", lw=0.7)
    ax[1].set_xlabel("расстояние между центрами, мм")
    ax[1].set_ylabel("ошибка, дБ")
    ax[1].set_title(f"Тест: ошибка от расстояния (|S| >= {thr:g} дБ)")
    ax[2].scatter(r[te], y[te], s=4, alpha=0.4, label="расчет")
    ax[2].scatter(r[te], pred[te], s=4, alpha=0.4, label="модель")
    ax[2].axhline(thr, color="r", lw=0.8)
    ax[2].set_xlabel("расстояние между центрами, мм")
    ax[2].set_ylabel("|S_ij|, дБ")
    ax[2].set_title("Тест: |S_ij| от расстояния")
    ax[2].legend()
    fig.savefig(path, dpi=110)
    print(f"  график: {path}")


# ---------------------------------------------------------------------
#  Диагностика промахов
# ---------------------------------------------------------------------
def miss_flags(y, p, thr):
    """Промах: выше порога |ошибка| > 3 дБ; ниже порога -- модель завысила больше чем до thr + 3."""
    return np.where(y >= thr, np.abs(p - y) > 3.0, p > thr + 3.0)


def row_features(obs, geometry, k, sgn):
    """Признаки каждой строки: затенение, теснота, край, направление на соседа."""
    lam = 2 * np.pi / k
    n = len(obs)
    shadow = np.full(n, np.inf)      # мин. расстояние от третьей антенны до отрезка i-j
    n_shadow = np.zeros(n, int)      # третьих антенн ближе lambda/2 к отрезку
    crowd = np.full(n, np.inf)       # мин. расстояние от i или j до остальных антенн
    edge = np.full(n, np.inf)        # мин. расстояние от i или j до края экрана (области)
    for pid, (row_of, X, Y, TH) in geometry.items():
        m = (obs["pool"] == pid).to_numpy()
        if not m.any():
            continue
        r = obs.loc[m, "eval_id"].map(row_of).to_numpy()
        i = obs.loc[m, "i"].to_numpy()
        j = obs.loc[m, "j"].to_numpy()
        PX, PY = X[r], Y[r]                                  # (rows, n_ant)
        ax, ay = PX[np.arange(len(r)), i], PY[np.arange(len(r)), i]
        bx, by = PX[np.arange(len(r)), j], PY[np.arange(len(r)), j]
        dx, dy = (bx - ax)[:, None], (by - ay)[:, None]
        L2 = dx ** 2 + dy ** 2
        t = np.clip(((PX - ax[:, None]) * dx + (PY - ay[:, None]) * dy) / L2, 0, 1)
        ds = np.hypot(PX - ax[:, None] - t * dx, PY - ay[:, None] - t * dy)
        di = np.hypot(PX - ax[:, None], PY - ay[:, None])
        dj = np.hypot(PX - bx[:, None], PY - by[:, None])
        own = np.zeros_like(ds, bool)
        own[np.arange(len(r)), i] = True
        own[np.arange(len(r)), j] = True
        ds[own] = np.inf
        shadow[m] = ds.min(axis=1)
        n_shadow[m] = (ds < lam / 2).sum(axis=1)
        crowd[m] = np.where(own, np.inf, np.minimum(di, dj)).min(axis=1)
        if SCREEN_RECT_MM is not None:
            x0, x1, y0, y1 = SCREEN_RECT_MM
        else:
            x0, x1, y0, y1 = np.nanmin(X), np.nanmax(X), np.nanmin(Y), np.nanmax(Y)
        e_a = np.minimum.reduce([ax - x0, x1 - ax, ay - y0, y1 - ay])
        e_b = np.minimum.reduce([bx - x0, x1 - bx, by - y0, y1 - by])
        edge[m] = np.minimum(e_a, e_b)
    # направление на соседа в собственных осях антенны i (градусы)
    phi = np.degrees(np.arctan2(obs["yj"] - obs["yi"], obs["xj"] - obs["xi"])).to_numpy() - sgn * obs["thi"].to_numpy()
    phi = np.mod(phi, 360.0)
    return pd.DataFrame({"dist_lambda": obs["dist_mm"].to_numpy() / lam, "shadow_lambda": shadow / lam,
                         "n_shadow": n_shadow, "crowd_lambda": crowd / lam, "edge_lambda": edge / lam,
                         "phi_local_deg": phi}, index=obs.index)


def diagnostics(obs, pred, geometry, k, sgn, prefix):
    thr = THRESHOLD_DB
    te = ~obs["train"].to_numpy()
    y = obs["S_ij_dB"].to_numpy(float)
    f = row_features(obs, geometry, k, sgn)
    f["S_ij_dB"], f["S_pred_dB"] = y, pred
    f["err_dB"] = pred - y
    f["miss"] = miss_flags(y, pred, thr)
    f["split"] = np.where(te, "test", "train")
    for c in ["pool", "eval_id", "ant_i", "ant_j"]:
        f[c] = obs[c].to_numpy()
    f["pool"] += 1
    f.to_csv(prefix + "_rows.csv", index=False)
    T = f[te]
    base = T["miss"].mean()
    print(f"\n  Диагностика промахов на тесте (промах: |ошибка| > 3 дБ выше порога или завышение "
          f"больше чем до {thr + 3:g} дБ ниже порога); всего промахов {100 * base:.1f}% из {len(T)}")

    def table(title, col, edges, labels=None, note=""):
        print(f"  {title}{note}:")
        cat = pd.cut(T[col], edges, right=False, labels=labels)
        rows = []
        for lab, g in T.groupby(cat, observed=True):
            big = g["S_ij_dB"] >= thr
            mae = np.abs(g.loc[big, "err_dB"]).mean() if big.any() else np.nan
            rows.append((str(lab), len(g), 100 * g["miss"].mean(), mae))
            flag = "  <--" if len(g) >= 30 and g["miss"].mean() > 2 * base and g["miss"].mean() > 0.02 else ""
            mae_s = f"{mae:5.2f} дБ" if np.isfinite(mae) else "   —   "
            print(f"    {str(lab):<18} строк {len(g):7d}   промахов {100 * g['miss'].mean():5.1f}%   "
                  f"MAE выше порога {mae_s}{flag}")
        return rows

    inf = np.inf
    out = {}
    out["dist"] = table("Расстояние между антеннами", "dist_lambda", [0, 0.5, 1, 2, 4, inf],
                        ["<0.5 λ", "0.5-1 λ", "1-2 λ", "2-4 λ", ">=4 λ"])
    out["shadow"] = table("Затенение: ближайшая третья антенна до отрезка i-j", "shadow_lambda",
                          [0, 0.25, 0.5, 1, 2, inf], ["<0.25 λ", "0.25-0.5 λ", "0.5-1 λ", "1-2 λ", ">=2 λ"])
    out["nshadow"] = table("Число антенн ближе λ/2 к отрезку i-j", "n_shadow", [0, 1, 2, 3, inf],
                           ["0", "1", "2", ">=3"])
    out["crowd"] = table("Теснота: ближайшая третья антенна к i или j", "crowd_lambda",
                         [0, 0.5, 1, 2, inf], ["<0.5 λ", "0.5-1 λ", "1-2 λ", ">=2 λ"])
    out["edge"] = table("Край: расстояние от i или j до края", "edge_lambda", [-inf, 0.25, 0.5, 1, 2, inf],
                        ["<0.25 λ", "0.25-0.5 λ", "0.5-1 λ", "1-2 λ", ">=2 λ"],
                        "" if SCREEN_RECT_MM is not None else " (края экрана не заданы: край области размещения)")
    out["level"] = table("Уровень |S_ij| в расчете", "S_ij_dB", [-inf, thr, thr + 5, -20, -10, inf],
                         [f"<{thr:g}", f"{thr:g}..{thr + 5:g}", f"{thr + 5:g}..-20", "-20..-10", ">-10"])
    out["phi"] = table("Направление на соседа в осях антенны i", "phi_local_deg",
                       list(np.arange(0, 361, 45)), [f"{a}-{a + 45}°" for a in range(0, 360, 45)])
    # Затенение, теснота и край при фиксированном расстоянии: у длинных пар больше антенн рядом
    # с отрезком и они проще, поэтому без разбивки по расстоянию эффекты смешиваются.
    dist_bins = [(0, 1, "<1 λ"), (1, 2, "1-2 λ"), (2, 4, "2-4 λ"), (4, inf, ">=4 λ")]

    def stratified(title, cond, yes, no):
        print(f"  {title} при фиксированном расстоянии (промахов, %):")
        rows = []
        for a, b, nm in dist_bins:
            msk = (T["dist_lambda"] >= a) & (T["dist_lambda"] < b)
            g1, g0 = T[msk & cond], T[msk & ~cond]
            if len(g1) + len(g0) == 0:
                continue
            m1 = f"{100 * g1['miss'].mean():5.1f}%" if len(g1) else "   — "
            m0 = f"{100 * g0['miss'].mean():5.1f}%" if len(g0) else "   — "
            print(f"    {nm:<8} {yes}: {m1} (строк {len(g1):6d})   {no}: {m0} (строк {len(g0):6d})")
            rows.append((nm, 100 * g1["miss"].mean() if len(g1) else np.nan, len(g1),
                         100 * g0["miss"].mean() if len(g0) else np.nan, len(g0)))
        return rows, yes, no

    strat = {
        "shadow": stratified("Затенение (третья антенна ближе λ/2 к отрезку)", T["shadow_lambda"] < 0.5, "затенено", "открыто"),
        "crowd": stratified("Теснота (третья антенна ближе λ к i или j)", T["crowd_lambda"] < 1, "тесно", "свободно"),
        "edge": stratified("Край (i или j ближе λ/2 к краю)", T["edge_lambda"] < 0.5, "у края", "внутри"),
    }
    # По антеннам
    print("  Промахи по антеннам (антенна участвует как i или j):")
    ants = sorted(set(T["ant_i"]).union(T["ant_j"]))
    per_ant = []
    for a in ants:
        g = T[(T["ant_i"] == a) | (T["ant_j"] == a)]
        per_ant.append((a, 100 * g["miss"].mean(), len(g)))
    per_ant.sort(key=lambda t: -t[1])
    print("    " + ", ".join(f"{a}: {m:.1f}%" for a, m, _ in per_ant))
    # Худшие пары антенн
    pair = T.assign(pa=np.minimum(T["ant_i"], T["ant_j"]), pb=np.maximum(T["ant_i"], T["ant_j"]))
    gp = pair.groupby(["pa", "pb"])["miss"].agg(["mean", "size"])
    gp = gp[gp["size"] >= 10].sort_values("mean", ascending=False).head(8)
    print("  Пары антенн с наибольшей долей промахов: "
          + ", ".join(f"{a}-{b}: {100 * r['mean']:.0f}% (из {int(r['size'])})" for (a, b), r in gp.iterrows()))
    # Худшие строки
    sev = np.where(T["S_ij_dB"] >= thr, np.abs(T["err_dB"]), T["S_pred_dB"] - (thr + 3.0))
    worst = T[T["miss"]].assign(sev=sev[T["miss"].to_numpy()]).sort_values("sev", ascending=False).head(10)
    print("  Худшие промахи теста:")
    for _, w in worst.iterrows():
        print(f"    пул {w['pool']}, eval_id {w['eval_id']}, S{w['ant_i']},{w['ant_j']}: расчет {w['S_ij_dB']:6.1f}, "
              f"модель {w['S_pred_dB']:6.1f} дБ; r = {w['dist_lambda']:.2f} λ, затенение {w['shadow_lambda']:.2f} λ, "
              f"теснота {w['crowd_lambda']:.2f} λ, край {w['edge_lambda']:.2f} λ")
    print(f"  строки с признаками: {prefix}_rows.csv")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    titles = {"dist": "расстояние между антеннами", "level": "уровень |S| расчета, дБ",
              "phi": "направление на соседа (оси антенны i)", "nshadow": "антенн ближе λ/2 к отрезку (без учета расстояния)",
              "shadow": "затенение: третья антенна ближе λ/2 к отрезку", "crowd": "теснота: третья антенна ближе λ к i или j",
              "edge": "i или j ближе λ/2 к краю"}
    fig, axs = plt.subplots(2, 4, figsize=(22, 9), constrained_layout=True)
    for ax, key in zip(axs.flat, ["dist", "shadow", "crowd", "edge", "level", "phi", "nshadow"]):
        if key in strat:
            rows, yes, no = strat[key]
            xs = np.arange(len(rows))
            ax.bar(xs - 0.2, [r[1] for r in rows], 0.4, label=yes, color="#b55a4a")
            ax.bar(xs + 0.2, [r[3] for r in rows], 0.4, label=no, color="#4a7fb5")
            for q, r in enumerate(rows):
                ax.text(q - 0.2, np.nan_to_num(r[1]), f"{r[2]}", ha="center", va="bottom", fontsize=7)
                ax.text(q + 0.2, np.nan_to_num(r[3]), f"{r[4]}", ha="center", va="bottom", fontsize=7)
            ax.set_xticks(xs)
            ax.set_xticklabels([r[0] for r in rows])
            ax.set_xlabel("расстояние между антеннами")
            ax.legend()
        else:
            labs = [r[0] for r in out[key]]
            ax.bar(range(len(labs)), [r[2] for r in out[key]], color="#4a7fb5")
            ax.set_xticks(range(len(labs)))
            ax.set_xticklabels(labs, rotation=30, ha="right")
            for q, r in enumerate(out[key]):
                ax.text(q, r[2], f"n={r[1]}", ha="center", va="bottom", fontsize=7)
        ax.axhline(100 * base, color="k", ls="--", lw=0.8)
        ax.set_ylabel("промахов, %")
        ax.set_title(titles[key], fontsize=10)
    ax = axs.flat[7]
    ax.bar(range(len(per_ant)), [m for _, m, _ in per_ant], color="#b55a4a")
    ax.axhline(100 * base, color="k", ls="--", lw=0.8)
    ax.set_xticks(range(len(per_ant)))
    ax.set_xticklabels([str(a) for a, _, _ in per_ant], rotation=90)
    ax.set_ylabel("промахов, %")
    ax.set_title("по антеннам (отсортировано)")
    fig.suptitle(f"Доля промахов на тесте (пунктир -- в среднем {100 * base:.1f}%)")
    fig.savefig(prefix + ".png", dpi=100)
    print(f"  график диагностики: {prefix}.png")


def main():
    files = POOLS
    if len(sys.argv) > 1:
        args = sys.argv[1:]
        if len(args) % 2:
            raise SystemExit("нужны пары файлов: геометрия S-параметры ...")
        files = list(zip(args[0::2], args[1::2]))
    rng = np.random.default_rng(SEED)
    t0 = time.time()
    print(f"Чтение данных (freq_idx = {FREQ_IDX}):")
    obs, n_ant, freq = load_pools(files, FREQ_IDX)
    geometry = obs.attrs["geometry"]
    os.makedirs(OUT_DIR, exist_ok=True)

    groups = [("все пулы", obs)] if SHARED_ACROSS_POOLS else \
        [(f"пул {p + 1}", obs[obs["pool"] == p]) for p in sorted(obs["pool"].unique())]
    out = []
    for tag, part in groups:
        part = part.reset_index(drop=True)
        pred, model, sgn, k, types = run(part, n_ant, freq, tag, rng)
        out.append(part.assign(S_pred_dB=pred, split=np.where(part["train"], "train", "test")))
        suffix = "all" if SHARED_ACROSS_POOLS else f"pool{part['pool'].iloc[0] + 1}"
        np.savez(os.path.join(OUT_DIR, f"model_f{FREQ_IDX}_{suffix}.npz"), A=model.A, antenna_type=types,
                 basis=np.array([list(a) + [m] for a, m in BASIS]), order=model.order, theta_sign=sgn,
                 k_per_mm=k, freq_GHz=freq,
                 ground_plane_depth_mm=np.nan if GROUND_PLANE_DEPTH_MM is None else GROUND_PLANE_DEPTH_MM)
        plots(part, pred, os.path.join(OUT_DIR, f"fit_f{FREQ_IDX}_{suffix}.png"), THRESHOLD_DB)
        diagnostics(part, pred, geometry, k, sgn, os.path.join(OUT_DIR, f"diag_f{FREQ_IDX}_{suffix}"))
    res = pd.concat(out, ignore_index=True)
    res["pool"] = res["pool"] + 1
    cols = ["pool", "eval_id", "split", "freq_GHz", "ant_i", "ant_j", "S_ij_dB", "S_pred_dB", "dist_mm",
            "xi", "yi", "thi", "xj", "yj", "thj"]
    path = os.path.join(OUT_DIR, f"predictions_f{FREQ_IDX}.csv")
    res[cols].to_csv(path, index=False)
    print(f"\nПредсказания: {path}; всего {time.time() - t0:.0f} с")


if __name__ == "__main__":
    main()
