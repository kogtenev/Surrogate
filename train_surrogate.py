#!/usr/bin/env python3
"""
Мультипольный суррогат взаимных S-параметров системы антенн, обучаемый по модулям |S_ij| в дБ.

Модель (для одной частоты), для каждой пары антенн i != j:

    S_ij = sum_{p in i} sum_{q in j}  u_{j,q}^T  G(r_{j,q} - r_{i,p})  u_{i,p},     u_{i,p} = D(R_i) a_{m(i),p},

  p, q     -- полюсы антенн. Полюсы уникальной модели антенны m находятся по ее полю из VTK
              (см. ниже); у копий модели они переносятся с учетом положения и поворота копии;
  a_{m,p}  -- коэффициенты полюса p модели m при функциях d^alpha delta(x - r_p) e_m
              (|alpha| <= MAX_ORDER, m = x, y, z) в собственных осях антенны; комплексные, ищутся
              подгонкой; общие для всех копий модели;
  D(R)     -- поворот коэффициентов (каждый тензорный индекс поворачивается матрицей R);
  G        -- матрица связи: G[(beta,n),(alpha,m)] = (-1)^|beta| d^(alpha+beta) G_nm(d),
              G(d) = (I + grad grad / k^2) exp(ik|d|) / (4 pi |d|) -- функция Грина;
              расстояния безразмерные (k*d), постоянные множители входят в a.
  По желанию -- изображения в металлической плоскости под антеннами (GROUND_PLANE_DEPTH_MM).
  По взаимности S_ij = S_ji: обе строки (если есть) описываются одним значением модели.

Полюсы (YAML-файл с описанием антенн, ANTENNAS_YAML или первый аргумент командной строки):
  для каждой уникальной модели из ее VTK-файла (ASCII legacy) берутся элементы с указанными
  объемными атрибутами; вес элемента -- L2-норма поля E на элементе, деленная на объем элемента;
  граф элементов (соседство по граням) разбивается pymetis на "Number of Poles" частей с этими
  весами вершин и минимизацией объема обмена (communication volume); полюс -- взвешенное среднее
  центров элементов части. Origin и Rotation модели в YAML задают положение и поворот антенны
  в ее VTK-расчете: полюсы переводятся в собственные оси антенны и затем переносятся на каждую ее
  копию по координатам и углу из данных обучения. Для проверки в ParaView в OUT_DIR пишутся
  poles_model<m>_submesh.vtk (подсетка: cluster, weight, log10_weight, E_L2; cluster = -1 --
  ячейка не кластеризовалась) и poles_model<m>_centers.vtk (центры кластеров: точка + октаэдр,
  массив cluster), в единицах сетки -- совпадают с исходным решением.
  POLE_TOP_FRACTION < 1 -- кластеризуется только доля ячеек с наибольшими весами (горячие области
  у порта и кромок); если в ParaView кластеры выглядят как секторы всей подсетки, а не как
  области вокруг порта и кромок, стоит попробовать 0.05.
  Начальные коэффициенты полюсов (FIELD_INIT): линейный МНК по комплексному полю из VTK в дальней
  от полюсов половине подсетки, затем масштаб каждой модели по |S| и уточнение по всем данным.
  Без этого подгонка по модулям (без фаз) с несколькими полюсами часто застревает в локальных
  минимумах.

Фазы в данных нет, поэтому подгоняется 20 lg|S_ij| к S_dB методом Левенберга-Марквардта
(нормальные уравнения собираются блоками по парам моделей) с несколькими случайными стартами и
продолжением по порядку мультиполей. Значения ниже THRESHOLD_DB не подгоняются: штраф только если
модель предсказывает для них больше порога. Диагональные S_ii не используются (их может не быть).

Обучение: eval_id % 3 == 1 (eval_id = 1, 4, 7, ...), тест -- остальные. Порядок мультиполей
(<= MAX_ORDER) и регуляризация выбираются кросс-валидацией только по обучающей выборке; тест
используется только для итоговой оценки. Знак угла (по или против часовой) выбирается по
функции потерь на обучающей выборке, если THETA_SIGN = 0 (тот же знак применяется к Rotation в YAML).

Формат данных:
  геометрия: eval_id, ignore, ignore, x0_mm, y0_mm, th0_deg, x1_mm, y1_mm, th1_deg, x2_mm, ...
             (столбцы x{i}_mm, y{i}_mm, th{i}_deg ищутся по именам; если имен нет -- тройками
             после первых трех столбцов);
  S:         eval_id, freq_idx, freq_GHz, ant_i, ant_j, S_ij_dB  (ant_i -- номер i из геометрии;
             нумерация с 1 тоже распознается);
  YAML:      Antennas: список {ID, Unique model}; Unique models: {номер: {Number of Poles,
             Solution File, Real Field Name, Imag Field Name, Attributes, Origin, Rotation}}
             (необязательно: Attribute Field Name -- имя массива атрибутов в VTK). Пути к VTK --
             относительно папки YAML-файла.

Запуск:
    python3 train_surrogate.py                                   # файлы из настроек ниже
    python3 train_surrogate.py g1.csv s1.csv g2.csv s2.csv g3.csv s3.csv
    python3 train_surrogate.py antennas.yaml g1.csv s1.csv ...  # с полюсами из YAML
Нужны: numpy, pandas, scipy, matplotlib; с YAML -- еще pyyaml и pymetis.
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
ANTENNAS_YAML = None         # YAML с описанием антенн (полюсы из VTK); None -- один полюс в центре антенны
FREQ_IDX = 0                 # индекс частоты, для которой обучается модель
THRESHOLD_DB = -35.0         # S ниже этого уровня не важны
MAX_ORDER = None             # порядок мультиполей в каждом полюсе: 0 (диполи), 1, 2 (до квадруполей);
                             # None -- 1, если у какой-либо модели несколько полюсов, иначе 2
ANTENNA_TYPES = None         # только без YAML: None -- у каждой антенны свои коэффициенты; "same" --
                             # все одинаковые; или список номеров типов по антеннам
SHARED_ACROSS_POOLS = True   # True -- во всех пулах те же антенны (одна модель на все пулы)
THETA_SIGN = 0               # +1: угол против часовой стрелки, -1: по часовой, 0: выбрать по обучающей выборке
GROUND_PLANE_DEPTH_MM = None # глубина металлической плоскости под центрами антенн (мм) или None
SCREEN_RECT_MM = None        # края экрана/подложки (xmin, xmax, ymin, ymax) для диагностики; None --
                             # вместо них прямоугольник, охватывающий центры антенн в пуле
MESH_TO_MM = 1.0             # единица длины сетки VTK (и Origin в YAML) в мм: 1 -- мм, 1000 -- метры
ATTRIBUTE_FIELD_NAMES = ["attribute", "attributes", "material", "gmsh:physical", "CellEntityIds", "elem_attr"]
METIS_UFACTOR = None         # допустимый дисбаланс весов кластеров в METIS (30 = 3%); None -- как в METIS
POLE_TOP_FRACTION = 1.0      # доля ячеек с наибольшими весами, которые кластеризуются (1.0 -- все);
                             # например 0.05 -- только "горячие" области у порта и кромок
POLE_WEIGHT_POWER = 1.0      # вес ячейки = (||E||_L2 / V) ** POLE_WEIGHT_POWER (для METIS и центров)
FIELD_INIT = True            # начальные коэффициенты полюсов -- МНК по комплексному полю из VTK
FIELD_SAMPLES = 4000         # точек поля для этого МНК (ячейки подсетки, дальние от полюсов)
DO_CV = True                 # выбирать порядок и регуляризацию кросс-валидацией
REG_GRID = [0.1, 1.0, 10.0]  # кандидаты регуляризации (гребневая, растет с порядком)
REG_DEFAULT = 1e-1           # регуляризация без кросс-валидации и при выборе знака угла
CV_FOLDS = 3                 # блоков кросс-валидации (по конфигурациям)
N_RESTARTS = 6               # случайных стартов на нулевом порядке
CV_RESTARTS = 2              # то же при кросс-валидации
MAX_ITER = 100               # итераций Левенберга-Марквардта на одну подгонку
MEMORY_WARN_GB = 4.0         # предупреждать, если матрицы связи обучающей выборки больше
SEED = 0
OUT_DIR = "surrogate_out"
C_MM_PER_NS = 299.792458     # скорость света, мм/нс (f в ГГц -> длина волны в мм)
# ======================================================================

TRAIN_MOD = 3                # обучающая выборка: eval_id % 3 == 1
DB = 20.0 / np.log(10.0)
CHUNK_ENTRIES = 2_000_000    # комплексных элементов матриц связи за раз (ограничение памяти)

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
    obs.attrs["labels"] = labels
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



# ---------------------------------------------------------------------
#  YAML с описанием антенн
# ---------------------------------------------------------------------
def _norm_key(k):
    return re.sub(r"[^a-z0-9]", "", str(k).lower())


def _get(d, *patterns, required=True, what=""):
    """Значение по ключу: сначала точное совпадение нормализованного ключа с одним из шаблонов,
    затем ключ, содержащий шаблон (терпимо к опечаткам вида 'Nummber of Poles', 'Imag Filed Name')."""
    keys = {_norm_key(k): k for k in d}
    for p in patterns:
        if p in keys:
            return d[keys[p]]
    for p in patterns:
        for nk, k in keys.items():
            if p in nk:
                return d[k]
    if required:
        raise SystemExit(f"YAML: не найден ключ {what or patterns[0]} среди {list(d)}")
    return None


def _fix_yaml_text(text):
    """Мягкое исправление частых отступлений от синтаксиса YAML:
    '-ID: 0' (нет пробела после '-') и ключи элемента списка с отступом меньше его содержимого."""
    out, item_col = [], None
    for line in text.splitlines():
        body = line.lstrip(" ")
        ind = len(line) - len(body)
        if body.startswith("-") and len(body) > 1 and body[1] not in " -0123456789.":
            body = "- " + body[1:]
        if body.startswith("- "):
            item_col = ind + 2
        elif body and not body.startswith("#") and item_col is not None:
            if item_col - 2 < ind < item_col:
                ind = item_col
            elif ind <= item_col - 2:
                item_col = None
        out.append(" " * ind + body)
    return "\n".join(out)


def load_antennas_yaml(path):
    """-> ({ID антенны: имя модели}, {имя модели: параметры}), имена моделей -- строки."""
    try:
        import yaml
    except ImportError:
        raise SystemExit("для YAML нужен пакет pyyaml: pip install pyyaml")
    raw = open(path, encoding="utf-8").read()
    data = None
    try:
        data = yaml.safe_load(raw)
        if not (isinstance(data, dict) and isinstance(_get(data, "antennas", required=False), list)):
            data = None
    except yaml.YAMLError:
        data = None
    if data is None:
        try:
            data = yaml.safe_load(_fix_yaml_text(raw))
        except yaml.YAMLError as e:
            raise SystemExit(f"{path}: не удалось разобрать YAML: {e}")
        print(f"  {path}: YAML исправлен ('-ID' -> '- ID', отступы ключей элементов списка)")
    name = lambda v: str(int(v)) if isinstance(v, (int, float)) or str(v).strip().lstrip("-").isdigit() else str(v).strip()
    ants = {}
    for item in _get(data, "antennas", what="Antennas"):
        aid = int(_get(item, "id", what="ID"))
        ants[aid] = name(_get(item, "uniquemodel", "model", what="Unique model"))
    base = os.path.dirname(os.path.abspath(path))
    models = {}
    for key, cfg in _get(data, "uniquemodels", "models", what="Unique models").items():
        fname = str(_get(cfg, "solutionfile", "file", what="Solution File")).strip()
        models[name(key)] = {
            "n_poles": int(_get(cfg, "numberofpoles", "poles", "pole", what="Number of Poles")),
            "file": fname if os.path.isabs(fname) else os.path.join(base, fname),
            "real": str(_get(cfg, "realfieldname", "real", what="Real Field Name")).strip(),
            "imag": str(_get(cfg, "imagfieldname", "imag", what="Imag Field Name")).strip(),
            "attributes": [int(a) for a in np.atleast_1d(_get(cfg, "attributes", what="Attributes"))],
            "attr_field": _get(cfg, "attributefieldname", "attributefield", required=False),
            "origin": np.array(_get(cfg, "origin", what="Origin"), dtype=float).ravel()[:3],
            "rotation": float(_get(cfg, "rotation", "angle", what="Rotation")),
        }
        if len(models[name(key)]["origin"]) == 2:
            models[name(key)]["origin"] = np.append(models[name(key)]["origin"], 0.0)
    missing = sorted(set(ants.values()) - set(models))
    if missing:
        raise SystemExit(f"YAML: модели {missing} указаны у антенн, но не описаны в Unique models")
    return ants, models


# ---------------------------------------------------------------------
#  Чтение ASCII VTK (legacy, UNSTRUCTURED_GRID)
# ---------------------------------------------------------------------
VTK_3D = {10: 4, 12: 8, 13: 6, 14: 5, 24: 10, 25: 20}  # тетраэдр, гексаэдр, призма, пирамида, квадр. тет/гекс
CORNERS = {10: 4, 12: 8, 13: 6, 14: 5, 24: 4, 25: 8}   # угловые вершины
FACES = {
    4: [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)],
    8: [(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)],
    6: [(0, 1, 2), (3, 4, 5), (0, 1, 4, 3), (1, 2, 5, 4), (2, 0, 3, 5)],
    5: [(0, 1, 2, 3), (0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)],
}
TETS = {  # разбиение на тетраэдры для объема
    4: [(0, 1, 2, 3)],
    8: [(0, 1, 2, 6), (0, 2, 3, 6), (0, 3, 7, 6), (0, 7, 4, 6), (0, 4, 5, 6), (0, 5, 1, 6)],
    6: [(0, 1, 2, 5), (0, 1, 5, 4), (0, 4, 5, 3)],
    5: [(0, 1, 2, 4), (0, 2, 3, 4)],
}


class _Lines:
    def __init__(self, f):
        self.f, self.back = f, None

    def next(self):
        if self.back is not None:
            line, self.back = self.back, None
            return line
        while True:
            line = self.f.readline()
            if not line:
                return None
            if line.strip():
                return line

    def push(self, line):
        self.back = line

    def numbers(self, count, dtype=float):
        parts, n = [], 0
        while n < count:
            line = self.next()
            if line is None:
                raise ValueError("файл VTK оборвался")
            parts.append(line)
            n += len(line.split())
        arr = np.array(" ".join(parts).split(), dtype=dtype)
        if len(arr) != count:
            raise ValueError(f"ожидалось {count} чисел, прочитано {len(arr)}")
        return arr


def read_vtk_ascii(path):
    """-> dict(points (N,3), cell_types (C,), offsets (C+1,), conn, cell {имя: (C,k)}, point {имя: (N,k)})."""
    out = {"cell": {}, "point": {}}
    with open(path, encoding="latin-1") as f:
        f.readline()
        f.readline()
        if f.readline().strip().upper() != "ASCII":
            raise SystemExit(f"{path}: поддерживается только ASCII VTK")
        L = _Lines(f)
        ctx, count = None, 0
        while True:
            line = L.next()
            if line is None:
                break
            tok = line.split()
            key = tok[0].upper()
            if key == "DATASET":
                if tok[1].upper() != "UNSTRUCTURED_GRID":
                    raise SystemExit(f"{path}: поддерживается только UNSTRUCTURED_GRID, а не {tok[1]}")
            elif key == "POINTS":
                out["points"] = L.numbers(3 * int(tok[1])).reshape(-1, 3)
            elif key == "CELLS":
                nxt = L.next()
                if nxt.split()[0].upper() == "OFFSETS":          # формат VTK 5.x
                    offs = L.numbers(int(tok[1]), np.int64)
                    hdr = L.next()
                    assert hdr.split()[0].upper() == "CONNECTIVITY"
                    out["conn"] = L.numbers(int(tok[2]), np.int64)
                    out["offsets"] = offs
                else:
                    L.push(nxt)
                    raw = L.numbers(int(tok[2]), np.int64)
                    nc = int(tok[1])
                    offs = np.zeros(nc + 1, np.int64)
                    conn = []
                    p = 0
                    for c in range(nc):
                        k = raw[p]
                        conn.append(raw[p + 1:p + 1 + k])
                        offs[c + 1] = offs[c] + k
                        p += k + 1
                    out["conn"] = np.concatenate(conn) if conn else np.zeros(0, np.int64)
                    out["offsets"] = offs
            elif key == "CELL_TYPES":
                out["cell_types"] = L.numbers(int(tok[1]), np.int64)
            elif key in ("CELL_DATA", "POINT_DATA"):
                ctx, count = ("cell" if key == "CELL_DATA" else "point"), int(tok[1])
            elif key == "SCALARS":
                ncomp = int(tok[3]) if len(tok) > 3 else 1
                nxt = L.next()
                if nxt.split()[0].upper() != "LOOKUP_TABLE":
                    L.push(nxt)
                out[ctx][tok[1]] = L.numbers(count * ncomp).reshape(count, ncomp)
            elif key in ("VECTORS", "NORMALS"):
                out[ctx][tok[1]] = L.numbers(3 * count).reshape(count, 3)
            elif key == "TENSORS":
                out[ctx][tok[1]] = L.numbers(9 * count).reshape(count, 9)
            elif key == "COLOR_SCALARS":
                out[ctx][tok[1]] = L.numbers(count * int(tok[2])).reshape(count, -1)
            elif key == "LOOKUP_TABLE":
                L.numbers(4 * int(tok[2]))
            elif key == "FIELD":
                for _ in range(int(tok[2])):
                    h = L.next().split()
                    nc, nt = int(h[1]), int(h[2])
                    arr = L.numbers(nc * nt).reshape(nt, nc)
                    if ctx is not None:
                        out[ctx][h[0]] = arr
            elif key == "METADATA":
                while True:                                     # блок до пустой строки
                    s = f.readline()
                    if not s or not s.strip():
                        break
            else:
                raise SystemExit(f"{path}: неизвестный раздел VTK '{tok[0]}'")
    for need in ("points", "conn", "offsets", "cell_types"):
        if need not in out:
            raise SystemExit(f"{path}: в файле нет раздела {need.upper()}")
    return out


# ---------------------------------------------------------------------
#  Полюсы: подсетка, веса, кластеры pymetis
# ---------------------------------------------------------------------
def _cell_field(vtk, name, sel_cells, corner_idx, ncorner):
    """Значения поля в углах выбранных ячеек (C, ncorner, 3) или в ячейках (C, 1, 3)."""
    if name in vtk["point"]:
        a = vtk["point"][name]
        return a[corner_idx], "point"
    if name in vtk["cell"]:
        return vtk["cell"][name][sel_cells][:, None, :], "cell"
    names = sorted(set(vtk["point"]) | set(vtk["cell"]))
    raise SystemExit(f"в VTK нет поля '{name}'; есть: {names}")


def _write_vtk_grid(path, points, cells_by_type, cell_data, point_data=None):
    """cells_by_type: список (vtk_type, conn (C, k)); данные ячеек в том же порядке."""
    with open(path, "w") as f:
        f.write("# vtk DataFile Version 3.0\nmultipole poles\nASCII\nDATASET UNSTRUCTURED_GRID\n")
        f.write(f"POINTS {len(points)} double\n")
        np.savetxt(f, points, fmt="%.9g")
        nc = sum(len(c) for _, c in cells_by_type)
        size = sum(c.size + len(c) for _, c in cells_by_type)
        f.write(f"CELLS {nc} {size}\n")
        for _, c in cells_by_type:
            np.savetxt(f, np.hstack([np.full((len(c), 1), c.shape[1]), c]), fmt="%d")
        f.write(f"CELL_TYPES {nc}\n")
        for t, c in cells_by_type:
            np.savetxt(f, np.full(len(c), t), fmt="%d")
        f.write(f"CELL_DATA {nc}\n")
        for name, (vals, fmt) in cell_data.items():
            f.write(f"SCALARS {name} {'int' if fmt == '%d' else 'double'} 1\nLOOKUP_TABLE default\n")
            np.savetxt(f, vals, fmt=fmt)
        if point_data:
            f.write(f"POINT_DATA {len(points)}\n")
            for name, (vals, fmt) in point_data.items():
                f.write(f"SCALARS {name} {'int' if fmt == '%d' else 'double'} 1\nLOOKUP_TABLE default\n")
                np.savetxt(f, vals, fmt=fmt)


def compute_model_poles(mname, cfg, out_dir):
    """Полюсы уникальной модели в координатах ее VTK-расчета (мм) + запись VTK для ParaView."""
    t0 = time.time()
    vtk = read_vtk_ascii(cfg["file"])
    types = vtk["cell_types"]
    offs, conn = vtk["offsets"], vtk["conn"]
    attr_name = cfg["attr_field"]
    cand = [attr_name] if attr_name else ATTRIBUTE_FIELD_NAMES
    attr = next((vtk["cell"][n][:, 0] for n in cand if n in vtk["cell"]), None)
    if attr is None:
        raise SystemExit(f"{cfg['file']}: нет массива атрибутов ячеек (искали {cand}); "
                         f"есть {sorted(vtk['cell'])} -- укажите 'Attribute Field Name' в YAML")
    sel = np.isin(np.rint(attr).astype(int), cfg["attributes"]) & np.isin(types, list(VTK_3D))
    if not sel.any():
        raise SystemExit(f"{cfg['file']}: нет объемных ячеек с атрибутами {cfg['attributes']}")
    # склейка совпадающих точек (MFEM пишет точки каждого элемента отдельно)
    P = vtk["points"]
    tol = 1e-9 * max(np.ptp(P, axis=0).max(), 1e-30)
    _, merged = np.unique(np.rint(P / tol).astype(np.int64), axis=0, return_inverse=True)
    merged = merged.ravel()
    sel_cells = np.flatnonzero(sel)
    vol = np.zeros(len(sel_cells))
    cen = np.zeros((len(sel_cells), 3))
    e2 = np.zeros(len(sel_cells))
    ecell = np.zeros((len(sel_cells), 3), complex)                # поле в ячейке (среднее по углам)
    faces, face_cell = [], []
    out_cells = []
    for t in np.unique(types[sel_cells]):
        loc = np.flatnonzero(types[sel_cells] == t)
        cells = sel_cells[loc]
        nn = VTK_3D[t]
        nc = CORNERS[t]
        nodes = conn[offs[cells][:, None] + np.arange(nn)[None, :]]
        corner = nodes[:, :nc]
        X = P[corner]                                             # (C, nc, 3)
        v = np.zeros(len(cells))
        for tet in TETS[nc]:
            a, b, c, d = (X[:, i] for i in tet)
            v += np.abs(np.einsum("ij,ij->i", b - a, np.cross(c - a, d - a))) / 6.0
        vol[loc] = v
        cen[loc] = X.mean(axis=1)
        Er, kind = _cell_field(vtk, cfg["real"], cells, corner, nc)
        Ei, _ = _cell_field(vtk, cfg["imag"], cells, corner, nc)
        sq = np.sum(Er ** 2 + Ei ** 2, axis=2)                   # |E|^2 в углах (C, nc) или (C, 1)
        if kind == "point" and nc == 4:                          # точный интеграл линейной функции
            sr = np.sum(Er, axis=1)
            si = np.sum(Ei, axis=1)
            e2[loc] = v / 20.0 * (sq.sum(axis=1) + np.sum(sr ** 2 + si ** 2, axis=1))
        else:
            e2[loc] = v * sq.mean(axis=1)
        ecell[loc] = np.mean(Er + 1j * Ei, axis=1)
        mc = merged[corner]
        for fc in FACES[nc]:
            fn = np.sort(mc[:, list(fc)], axis=1)
            if fn.shape[1] == 3:
                fn = np.hstack([np.full((len(fn), 1), -1), fn])
            faces.append(fn)
            face_cell.append(loc)
        out_cells.append((t, loc, merged[nodes]))
    l2 = np.sqrt(e2)
    w_all = (l2 / np.maximum(vol, 1e-300)) ** POLE_WEIGHT_POWER
    # кластеризуются ячейки с наибольшими весами (по умолчанию все)
    keep = np.ones(len(w_all), bool)
    if POLE_TOP_FRACTION < 1.0:
        keep = w_all >= np.quantile(w_all, 1.0 - POLE_TOP_FRACTION)
    kidx = -np.ones(len(w_all), int)
    kidx[keep] = np.arange(keep.sum())
    w = w_all[keep]
    cen_k = cen[keep]
    # граф соседства по граням (между оставленными ячейками)
    F = np.vstack(faces)
    FC = kidx[np.concatenate(face_cell)]
    F, FC = F[FC >= 0], FC[FC >= 0]
    _, finv, fcnt = np.unique(F, axis=0, return_inverse=True, return_counts=True)
    finv = finv.ravel()
    order = np.argsort(finv, kind="stable")
    fs = finv[order]
    shared = np.flatnonzero((fs[1:] == fs[:-1]))
    a, b = FC[order][shared], FC[order][shared + 1]
    import scipy.sparse as sps
    from scipy.sparse.csgraph import connected_components
    n = int(keep.sum())
    Adj = sps.coo_matrix((np.ones(len(a) * 2), (np.r_[a, b], np.r_[b, a])), shape=(n, n)).tocsr()
    Adj.sum_duplicates()
    Adj.setdiag(0)
    Adj.eliminate_zeros()
    ncomp, _ = connected_components(Adj, directed=False)
    k = int(cfg["n_poles"])
    if k <= 1:
        part = np.zeros(n, int)
    else:
        try:
            import pymetis
        except ImportError:
            raise SystemExit("для кластеризации нужен пакет pymetis: pip install pymetis")
        scale = min(1e6 / w.max(), 1e9 / w.sum())
        vw = np.maximum(1, np.rint(w * scale)).astype(np.int64)
        opts = pymetis.Options()
        try:
            opts.objtype = 1                                     # METIS_OBJTYPE_VOL
        except Exception:
            print("    pymetis.Options без objtype: минимизируется разрез, а не объем обмена")
        if METIS_UFACTOR is not None:
            opts.ufactor = int(METIS_UFACTOR)
        contig = ncomp == 1
        kw = dict(xadj=Adj.indptr.astype(np.int64), adjncy=Adj.indices.astype(np.int64),
                  vweights=vw.tolist(), recursive=False, options=opts)
        membership = None
        for extra in ({"contiguous": contig}, {}):               # старые pymetis без contiguous/options
            try:
                _, membership = pymetis.part_graph(k, **kw, **extra)
                break
            except TypeError:
                continue
        if membership is None:
            print("    pymetis без options/recursive: разбиение по умолчанию (минимум разреза)")
            _, membership = pymetis.part_graph(k, xadj=kw["xadj"], adjncy=kw["adjncy"], vweights=kw["vweights"])
        part = np.asarray(membership, int)
    used = np.unique(part)
    if len(used) < k:
        print(f"    METIS вернул {len(used)} непустых частей из {k}")
    part = np.searchsorted(used, part)
    poles = np.array([np.average(cen_k[part == c], axis=0, weights=w[part == c]) for c in range(len(used))])
    share = np.array([w[part == c].sum() for c in range(len(used))]) / w.sum()
    part_all = -np.ones(len(w_all), int)                         # -1 -- ячейка не кластеризовалась
    part_all[keep] = part
    # VTK для ParaView (в единицах сетки, совпадает с исходным решением)
    os.makedirs(out_dir, exist_ok=True)
    node_ids = np.unique(np.concatenate([c.ravel() for _, _, c in out_cells]))
    first = np.zeros(merged.max() + 1, np.int64)
    first[merged[::-1]] = np.arange(len(merged))[::-1]            # представитель каждой склеенной точки
    remap = -np.ones(merged.max() + 1, np.int64)
    remap[node_ids] = np.arange(len(node_ids))
    cells_out, cd_order = [], []
    for t, loc, nodes in out_cells:
        cells_out.append((t, remap[nodes]))
        cd_order.append(loc)
    cd_order = np.concatenate(cd_order)
    sub_path = os.path.join(out_dir, f"poles_model{mname}_submesh.vtk")
    _write_vtk_grid(sub_path, P[first[node_ids]], cells_out, {
        "cluster": (part_all[cd_order], "%d"), "weight": (w_all[cd_order], "%.6g"),
        "log10_weight": (np.log10(np.maximum(w_all[cd_order], 1e-300)), "%.6g"), "E_L2": (l2[cd_order], "%.6g")})
    # центры кластеров: точка + октаэдр
    diag = np.linalg.norm(np.ptp(cen, axis=0))                  # размер маркеров -- по всей подсетке
    rad = 0.04 * diag
    if len(poles) > 1:
        dmin = min(np.linalg.norm(poles[i] - poles[j]) for i in range(len(poles)) for j in range(i))
        rad = min(rad, 0.35 * dmin) if dmin > 0 else rad
    octv = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], float)
    octf = np.array([[0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4], [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5]])
    pts = [poles]
    tris = []
    for c, p in enumerate(poles):
        base = len(poles) + 6 * c
        pts.append(p + rad * octv)
        tris.append(octf + base)
    pts = np.vstack(pts)
    tris = np.vstack(tris)
    cl_cells = np.r_[np.arange(len(poles)), np.repeat(np.arange(len(poles)), 8)]
    cl_pts = np.r_[np.arange(len(poles)), np.repeat(np.arange(len(poles)), 6)]
    cen_path = os.path.join(out_dir, f"poles_model{mname}_centers.vtk")
    _write_vtk_grid(cen_path, pts, [(1, np.arange(len(poles))[:, None]), (5, tris)],
                    {"cluster": (cl_cells, "%d")}, {"cluster": (cl_pts, "%d")})
    print(f"  модель {mname}: {os.path.basename(cfg['file'])}, ячеек с атрибутами {cfg['attributes']}: {len(w_all)}"
          + (f" (кластеризуется {n} с наибольшими весами)" if n < len(w_all) else "")
          + (f", несвязных частей {ncomp}" if ncomp > 1 else "") + f", полюсов {len(poles)} ({time.time() - t0:.1f} с)")
    for c, p in enumerate(poles):
        print(f"    полюс {c}: ({p[0]:.4g}, {p[1]:.4g}, {p[2]:.4g}) ед. сетки, доля веса {100 * share[c]:.0f}%, "
              f"ячеек {int((part == c).sum())}")
    print(f"    ParaView: {sub_path} (раскраска cluster / log10_weight), {cen_path}")
    # точки для начального приближения коэффициентов: дальняя от полюсов половина ячеек
    dist = np.min(np.linalg.norm(cen[:, None, :] - poles[None, :, :], axis=2), axis=1)
    mag = np.linalg.norm(ecell, axis=1)
    ok = (dist >= np.median(dist)) & (mag > 1e-4 * mag.max())
    cand = np.flatnonzero(ok)
    if len(cand) > FIELD_SAMPLES:
        cand = np.random.default_rng(SEED).choice(cand, FIELD_SAMPLES, replace=False)
    samples = (cen[cand] * MESH_TO_MM, ecell[cand])
    return poles * MESH_TO_MM, share, samples


class AntennaModel:
    """Уникальная модель антенны: полюсы в координатах ее расчета, начало и поворот антенны там."""

    def __init__(self, name, poles_sim_mm, origin_mm=(0, 0, 0), rotation_deg=0.0, share=None, samples=None):
        self.name = str(name)
        self.poles_sim = np.atleast_2d(np.asarray(poles_sim_mm, float))
        self.origin = np.asarray(origin_mm, float)
        self.rotation = float(rotation_deg)
        self.share = share
        self.n = len(self.poles_sim)
        self.samples = samples                                   # (точки мм, комплексное E) из VTK или None

    def field_coefficients(self, k, sgn, nb, rel_reg=1e-8):
        """Коэффициенты полюсов в собственных осях антенны по полю из VTK (линейный МНК):
        E(x) = sum_p sum_j c_{p,j} E_j(x - r_p), E_j -- поле j-й мультипольной функции.
        -> (n, nb) complex или None, если поля нет."""
        if self.samples is None or len(self.samples[0]) == 0:
            return None
        X, E = self.samples
        cols = [coupling_matrix(k * (X - p), nb)[:, :3, :] for p in self.poles_sim]   # (N, 3, nb)
        A = np.concatenate(cols, axis=2).reshape(-1, self.n * nb)
        b = E.reshape(-1)
        sc = np.linalg.norm(A, axis=0) + 1e-300
        An = A / sc
        AhA = An.conj().T @ An
        c = np.linalg.solve(AhA + rel_reg * np.trace(AhA).real / len(AhA) * np.eye(len(AhA)), An.conj().T @ b) / sc
        rel = np.linalg.norm(A @ c - b) / np.linalg.norm(b)
        Dinv = basis_rotation(rotation_z(np.array(sgn * np.deg2rad(self.rotation))).T)[:nb, :nb]
        return (c.reshape(self.n, nb) @ Dinv.T), rel

    def local(self, sgn):
        """Полюсы в собственных осях антенны (мм): R(sgn*rot)^T (p - origin)."""
        R = rotation_z(np.array(sgn * np.deg2rad(self.rotation)))
        return (self.poles_sim - self.origin) @ R


def antenna_models(n_ant, labels, yaml_path):
    """-> (models: список AntennaModel, model_of: номер модели для каждой антенны)."""
    if yaml_path is None:
        types = antenna_types(n_ant)
        models = [AntennaModel(f"тип {t}", [[0.0, 0.0, 0.0]]) for t in range(int(types.max()) + 1)]
        return models, types
    print(f"\nПолюсы антенн ({yaml_path}):")
    ants, cfgs = load_antennas_yaml(yaml_path)
    names = sorted(set(ants.values()), key=lambda s: (len(s), s))
    models, index = [], {}
    for nm in names:
        cfg = cfgs[nm]
        poles, share, samples = compute_model_poles(nm, cfg, OUT_DIR)
        index[nm] = len(models)
        models.append(AntennaModel(nm, poles, cfg["origin"] * MESH_TO_MM, cfg["rotation"], share, samples))
    model_of = np.empty(n_ant, int)
    fallback = []
    for q, lab in enumerate(labels):
        if lab in ants:
            model_of[q] = index[ants[lab]]
        else:
            model_of[q] = len(models)
            models.append(AntennaModel(f"антенна {lab} (нет в YAML)", [[0.0, 0.0, 0.0]]))
            fallback.append(lab)
    if fallback:
        print(f"  антенн нет в YAML: {fallback} -- у каждой свой полюс в центре")
    unused = sorted(set(ants) - set(labels))
    if unused:
        print(f"  в YAML есть антенны, которых нет в данных: {unused}")
    for m in models:
        cnt = int((model_of == models.index(m)).sum())
        print(f"  модель {m.name}: полюсов {m.n}, антенн {cnt}")
    return models, model_of


def coupling_matrix(dk, nb=None):
    """dk: (N, 3) безразмерные векторы r_1 - r_0 -> (N, nb, nb) матрицы связи (первые nb функций)."""
    nb = NB if nb is None else nb
    P = phi_derivatives(dk)
    return CSIGN[:nb, :nb] * (CDIAG[:nb, :nb] * P[:, CI1[:nb, :nb]] + P[:, CI2[:nb, :nb]])


def n_basis(order):
    return int((BASIS_ORDER <= order).sum())


# ---------------------------------------------------------------------
#  Матрицы связи пар полюсов
# ---------------------------------------------------------------------
class Group:
    """Пары антенн с одинаковыми моделями (m0 у антенны lo, m1 у hi).
    M3[n, p*nb + j, q*nb + i] = связь функции j полюса p антенны lo с функцией i полюса q антенны hi,
    так что S = a_hi^T M3^T a_lo: G0 = M3 @ a_hi (dS/da_lo), G1 = a_lo @ M3 (dS/da_hi)."""

    def __init__(self, m0, m1, n0, n1, nb, M3):
        self.m0, self.m1, self.n0, self.n1, self.nb, self.M3 = m0, m1, n0, n1, nb, M3
        self._cache = {}

    def block(self, k):
        if k == self.nb:
            return self.M3
        if k not in self._cache:
            n = len(self.M3)
            v = self.M3.reshape(n, self.n0, self.nb, self.n1, self.nb)[:, :, :k, :, :k]
            self._cache = {k: np.ascontiguousarray(v).reshape(n, self.n0 * k, self.n1 * k)}
        return self._cache[k]

    def take(self, idx):
        return Group(self.m0, self.m1, self.n0, self.n1, self.nb, self.M3[idx])


def group_matrices(pairs, m0, m1, k, sgn, nb):
    """M3 для пар с моделями (m0, m1): (n, n0*nb, n1*nb) complex64."""
    P0, P1 = m0.local(sgn), m1.local(sgn)
    n0, n1, n = len(P0), len(P1), len(pairs)
    out = np.empty((n, n0 * nb, n1 * nb), dtype=np.complex64)
    DP = basis_rotation(np.diag([1.0, 1.0, -1.0]))[:nb, :nb]     # отражение; ток изображения -P J
    step = max(1, int(CHUNK_ENTRIES // (n0 * n1 * nb * nb)))
    x0, y0, t0 = (pairs[c].to_numpy(float) for c in ("x_lo", "y_lo", "th_lo"))
    x1, y1, t1 = (pairs[c].to_numpy(float) for c in ("x_hi", "y_hi", "th_hi"))
    for a in range(0, n, step):
        b = min(n, a + step)
        R0 = rotation_z(np.deg2rad(t0[a:b]) * sgn)
        R1 = rotation_z(np.deg2rad(t1[a:b]) * sgn)
        r0 = np.stack([x0[a:b], y0[a:b], np.zeros(b - a)], 1)[:, None, :] + np.einsum("cij,pj->cpi", R0, P0)
        r1 = np.stack([x1[a:b], y1[a:b], np.zeros(b - a)], 1)[:, None, :] + np.einsum("cij,pj->cpi", R1, P1)
        d = r1[:, None, :, :] - r0[:, :, None, :]                # (c, n0, n1, 3)
        G = coupling_matrix(k * d.reshape(-1, 3), nb)
        if GROUND_PLANE_DEPTH_MM is not None:
            r0i = r0.copy()
            r0i[..., 2] = -2.0 * GROUND_PLANE_DEPTH_MM - r0[..., 2]
            G = G - coupling_matrix(k * (r1[:, None, :, :] - r0i[:, :, None, :]).reshape(-1, 3), nb) @ DP
        G = G.reshape(b - a, n0, n1, nb, nb)
        D0 = basis_rotation(R0)[:, :nb, :nb]
        D1 = basis_rotation(R1)[:, :nb, :nb]
        Mpq = np.matmul(np.matmul(D1.transpose(0, 2, 1)[:, None, None], G), D0[:, None, None])   # (c,p,q,i,j)
        out[a:b] = Mpq.transpose(0, 1, 4, 2, 3).reshape(b - a, n0 * nb, n1 * nb)
    return out


def coupling_bytes(pairs, model_of, models, nb):
    n0 = np.array([models[m].n for m in model_of[pairs["lo"].to_numpy()]])
    n1 = np.array([models[m].n for m in model_of[pairs["hi"].to_numpy()]])
    return float(np.sum(n0 * n1)) * nb * nb * 8


class Problem:
    """Наблюдения, сгруппированные по парам моделей."""

    def __init__(self, groups, obs_rows, obs_pos, y, n_obs):
        self.groups, self.obs_rows, self.obs_pos, self.y, self.n_obs = groups, obs_rows, obs_pos, y, n_obs

    @staticmethod
    def build(pairs, inv, obs_mask, y, model_of, models, k, sgn, nb):
        obs_idx = np.flatnonzero(obs_mask)
        used, op = np.unique(inv[obs_idx], return_inverse=True)
        sub = pairs.iloc[used]
        ml, mh = model_of[sub["lo"].to_numpy()], model_of[sub["hi"].to_numpy()]
        groups, rows, poss = [], [], []
        for a, b in sorted(set(zip(ml.tolist(), mh.tolist()))):
            pidx = np.flatnonzero((ml == a) & (mh == b))
            g = Group(a, b, models[a].n, models[b].n, nb, group_matrices(sub.iloc[pidx], models[a], models[b], k, sgn, nb))
            pos_of = -np.ones(len(sub), int)
            pos_of[pidx] = np.arange(len(pidx))
            r = np.flatnonzero(pos_of[op] >= 0)
            groups.append(g)
            rows.append(r)
            poss.append(pos_of[op[r]])
        return Problem(groups, rows, poss, y[obs_idx], len(obs_idx))

    def subset(self, mask):
        """Подзадача по маске наблюдений (в порядке наблюдений этой задачи)."""
        new_index = -np.ones(self.n_obs, int)
        new_index[mask] = np.arange(mask.sum())
        groups, rows, poss = [], [], []
        for g, r, p in zip(self.groups, self.obs_rows, self.obs_pos):
            keep = mask[r]
            if not keep.any():
                continue
            pu, pn = np.unique(p[keep], return_inverse=True)
            groups.append(g.take(pu))
            rows.append(new_index[r[keep]])
            poss.append(pn)
        return Problem(groups, rows, poss, self.y[mask], int(mask.sum()))


# ---------------------------------------------------------------------
#  Модель и подгонка
# ---------------------------------------------------------------------
class Model:
    """Коэффициенты C[m] (полюсы модели m x NB) на первых k функциях (порядок <= order)."""

    def __init__(self, order, n_poles):
        self.order = order
        self.k = n_basis(order)
        self.n_poles = list(n_poles)
        self.C = [np.zeros((n, NB), complex) for n in self.n_poles]
        sizes = [2 * n * self.k for n in self.n_poles]
        self.off = np.concatenate([[0], np.cumsum(sizes)]).astype(int)

    def size(self, m):
        return self.off[m + 1] - self.off[m]

    def pack(self):
        return np.concatenate([np.concatenate([c[:, :self.k].real.ravel(), c[:, :self.k].imag.ravel()]) for c in self.C])

    def unpack(self, x):
        for m, n in enumerate(self.n_poles):
            v = x[self.off[m]:self.off[m + 1]]
            h = n * self.k
            self.C[m] = np.zeros((n, NB), complex)
            self.C[m][:, :self.k] = (v[:h] + 1j * v[h:]).reshape(n, self.k)

    def flat(self, m):
        return self.C[m][:, :self.k].ravel().astype(np.complex64)

    def copy_to(self, order):
        m = Model(order, self.n_poles)
        m.C = [c.copy() for c in self.C]
        return m

    def s_obs(self, prob):
        s = np.zeros(prob.n_obs, complex)
        for g, r, p in zip(prob.groups, prob.obs_rows, prob.obs_pos):
            sg = (g.block(self.k) @ self.flat(g.m1)) @ self.flat(g.m0)
            s[r] = sg[p]
        return s

    def predict_db(self, prob):
        return DB * np.log(np.abs(self.s_obs(prob)) + 1e-300)


def thresholded_residual(p, y, thr):
    return np.where(y >= thr, p - y, np.maximum(p - thr, 0.0))


def evaluate(model, x, prob, thr):
    """Невязки (дБ) по всем наблюдениям и блоки якобиана по группам.
    d(dB)/d a = (20/ln10) Re(conj(s) ds) / |s|^2;  ds/dRe a = g, ds/dIm a = i g."""
    model.unpack(x)
    res = np.zeros(prob.n_obs)
    blocks = []
    for g, r, p in zip(prob.groups, prob.obs_rows, prob.obs_pos):
        B = g.block(model.k)
        a0, a1 = model.flat(g.m0), model.flat(g.m1)
        G0 = B @ a1                                    # dS/da_lo  (n_g, n0 k)
        G1 = a0 @ B                                    # dS/da_hi  (n_g, n1 k)
        so = (G0 @ a0)[p].astype(complex)
        y = prob.y[r]
        mdb = DB * np.log(np.abs(so) + 1e-300)
        res[r] = thresholded_residual(mdb, y, thr)
        active = (y >= thr) | (mdb > thr)
        w = (DB / (np.abs(so) ** 2 + 1e-300) * active)[:, None]
        sc = np.conj(so)[:, None]
        h0 = sc * G0[p]
        h1 = sc * G1[p]
        blocks.append((g.m0, g.m1, r, np.hstack([w * h0.real, -w * h0.imag]), np.hstack([w * h1.real, -w * h1.imag])))
    return res, blocks


def fit(model, prob, thr, reg, max_iter=MAX_ITER, tol=1e-7):
    """Левенберг-Марквардт; нормальные уравнения собираются плотными блоками по парам моделей."""
    from scipy.linalg import cho_factor, cho_solve, LinAlgError
    per = np.sqrt(reg) * (1.0 + BASIS_ORDER[:model.k])
    reg_w2 = np.concatenate([np.tile(np.tile(per, n), 2) for n in model.n_poles]) ** 2
    P = model.off[-1]

    def cost_of(x, res):
        return 0.5 * (res @ res + reg_w2 @ (x * x))

    x = model.pack()
    res, blocks = evaluate(model, x, prob, thr)
    cost = cost_of(x, res)
    mu = 1e-3
    for _ in range(max_iter):
        H = np.zeros((P, P))
        gvec = np.zeros(P)
        for m0, m1, r, J0, J1 in blocks:
            rr = res[r]
            s0 = slice(model.off[m0], model.off[m0 + 1])
            s1 = slice(model.off[m1], model.off[m1 + 1])
            if m0 == m1:
                Jc = J0 + J1
                H[s0, s0] += Jc.T @ Jc
                gvec[s0] += Jc.T @ rr
            else:
                H[s0, s0] += J0.T @ J0
                H[s1, s1] += J1.T @ J1
                X = J0.T @ J1
                H[s0, s1] += X
                H[s1, s0] += X.T
                gvec[s0] += J0.T @ rr
                gvec[s1] += J1.T @ rr
        H[np.diag_indices(P)] += reg_w2
        gvec += reg_w2 * x
        dH = np.diag(H).copy()
        improved = False
        while mu < 1e10:
            try:
                d = -cho_solve(cho_factor(H + mu * np.diag(dH + 1e-12 * dH.max())), gvec)
            except LinAlgError:
                mu *= 10
                continue
            xn = x + d
            rn, bn = evaluate(model, xn, prob, thr)
            cn = cost_of(xn, rn)
            if cn < cost:
                rel = (cost - cn) / max(cost, 1e-300)
                x, res, blocks, cost = xn, rn, bn, cn
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
    f = np.sqrt(10 ** (d / 20.0))
    model.C = [c * f for c in model.C]


def field_init(models, k, sgn, max_order, log=print):
    """Начальные модели по полю из VTK для порядков 0..max_order: {порядок: Model} (без масштаба)."""
    if not FIELD_INIT or all(m.samples is None for m in models):
        return {}
    init = {}
    for order in range(max_order + 1):
        nb = n_basis(order)
        mod = Model(order, [m.n for m in models])
        msg = []
        for mi, m in enumerate(models):
            r = m.field_coefficients(k, sgn, nb)
            if r is None:                                        # нет поля: случайный старт
                mod.C[mi][:, :3] = np.random.default_rng(SEED + mi).normal(size=(m.n, 3))
                continue
            mod.C[mi][:, :nb], rel = r
            msg.append(f"{m.name}: {100 * rel:.1f}%")
        init[order] = mod
        if log:
            log(f"    поле из VTK, порядок {order}: относительная невязка МНК " + ", ".join(msg))
    return init


def scale_per_model(model, prob, thr):
    """Масштаб коэффициентов каждой модели (у полей из разных VTK своя нормировка):
    20 lg|S| = 20 lg|S0| + o_lo + o_hi -- линейный МНК по наблюдениям выше порога."""
    pred = model.predict_db(prob)
    big = prob.y >= thr
    M = len(model.C)
    Xd = np.zeros((prob.n_obs, M))
    for g, r in zip(prob.groups, prob.obs_rows):
        Xd[r, g.m0] += 1.0
        Xd[r, g.m1] += 1.0
    o = np.linalg.lstsq(Xd[big], (prob.y - pred)[big], rcond=None)[0]
    model.C = [c * 10 ** (o[m] / 20.0) for m, c in enumerate(model.C)]


def train(prob, n_poles, thr, rng, reg, max_order, n_restarts=N_RESTARTS, refine_restarts=2, log=print, init=None):
    """Продолжение по порядку: 0 -> ... -> max_order; на нулевом порядке -- несколько стартов.
    init -- начальные модели по полю из VTK ({порядок: Model}): на каждом порядке пробуются и они.
    Возвращает [(порядок, функция потерь, модель)]."""
    log = log or (lambda *a: None)
    init = init or {}
    best = None
    if 0 in init:
        m = init[0].copy_to(0)
        scale_per_model(m, prob, thr)
        cost = fit(m, prob, thr, reg)
        best = (cost, m)
        log(f"    порядок 0, старт по полю из VTK: функция потерь {cost:.4g}")
    for _ in range(n_restarts):
        m = Model(0, n_poles)
        for c in m.C:
            c[:, :m.k] = rng.normal(size=(len(c), m.k)) + 1j * rng.normal(size=(len(c), m.k))
        scale_to_data(m, prob, thr)
        cost = fit(m, prob, thr, reg)
        if best is None or cost < best[0]:
            best = (cost, m)
    history = [(0, best[0], best[1])]
    log(f"    порядок 0: функция потерь {best[0]:.4g} (лучший из {n_restarts} стартов)")
    prev = best[1]
    for order in range(1, max_order + 1):
        cand = None
        new = np.where(BASIS_ORDER == order)[0]
        for rs in range(refine_restarts):
            m = prev.copy_to(order)
            if rs > 0:
                amp = 0.05 * np.mean([np.abs(c[:, :prev.k]).mean() for c in prev.C])
                for c in m.C:
                    c[:, new] = amp * (rng.normal(size=(len(c), len(new))) + 1j * rng.normal(size=(len(c), len(new))))
            cost = fit(m, prob, thr, reg)
            if cand is None or cost < cand[0]:
                cand = (cost, m)
        if order in init:
            m = init[order].copy_to(order)
            scale_per_model(m, prob, thr)
            cost = fit(m, prob, thr, reg)
            log(f"    порядок {order}, старт по полю из VTK: функция потерь {cost:.4g}")
            if cost < cand[0]:
                cand = (cost, m)
        history.append((order, cand[0], cand[1]))
        log(f"    порядок {order}: функция потерь {cand[0]:.4g}")
        prev = cand[1]
    return history


def cross_validate(prob, groups, n_poles, thr, rng, max_order, init=None):
    """Выбор (порядок, регуляризация) по RMS ошибки на отложенных блоках конфигураций
    обучающей выборки; groups -- метка конфигурации каждого наблюдения задачи."""
    ug = np.unique(groups)
    fold_of = dict(zip(ug, rng.permutation(len(ug)) % CV_FOLDS))
    fold = np.array([fold_of[g] for g in groups])
    table = {}
    for reg in REG_GRID:
        sse = np.zeros(max_order + 1)
        cnt = 0
        for f in range(CV_FOLDS):
            ptr, pva = prob.subset(fold != f), prob.subset(fold == f)
            hist = train(ptr, n_poles, thr, rng, reg, max_order, n_restarts=CV_RESTARTS, refine_restarts=1, log=None,
                         init=init)
            for order, _, model in hist:
                sse[order] += np.sum(thresholded_residual(model.predict_db(pva), pva.y, thr) ** 2)
            cnt += pva.n_obs
        table[reg] = np.sqrt(sse / cnt)
    reg, order = min(((r, o) for r in REG_GRID for o in range(max_order + 1)), key=lambda t: table[t[0]][t[1]])
    return order, reg, table


def predict_all(model, pairs, inv, model_of, models, k, sgn, nb):
    """Предсказание для всех наблюдений (матрицы связи считаются кусками, без хранения)."""
    s_pair = np.zeros(len(pairs), complex)
    ml, mh = model_of[pairs["lo"].to_numpy()], model_of[pairs["hi"].to_numpy()]
    for a, b in sorted(set(zip(ml.tolist(), mh.tolist()))):
        pidx = np.flatnonzero((ml == a) & (mh == b))
        n0, n1 = models[a].n, models[b].n
        step = max(1, int(5 * CHUNK_ENTRIES // (n0 * n1 * nb * nb)))
        a0, a1 = model.flat(a), model.flat(b)
        kk = model.k
        for c in range(0, len(pidx), step):
            idx = pidx[c:c + step]
            g = Group(a, b, n0, n1, nb, group_matrices(pairs.iloc[idx], models[a], models[b], k, sgn, nb))
            s_pair[idx] = (g.block(kk) @ a1) @ a0
    return DB * np.log(np.abs(s_pair[inv]) + 1e-300)
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




def run(obs, models, model_of, freq, tag, rng):
    k = 2 * np.pi * freq / C_MM_PER_NS
    lam = 2 * np.pi / k
    y = obs["S_ij_dB"].to_numpy(float)
    tr = obs["train"].to_numpy()
    r = obs["dist_mm"].to_numpy()
    groups = obs["group"].to_numpy()
    pairs, inv = build_pairs(obs)
    n_poles = [m.n for m in models]
    max_order = MAX_ORDER if MAX_ORDER is not None else (1 if max(n_poles) > 1 else 2)
    nb = n_basis(max_order)
    used_models = sorted(set(model_of[pairs["lo"]].tolist()) | set(model_of[pairs["hi"]].tolist()))
    print(f"\n=== {tag}: f = {freq:g} ГГц, lambda = {lam:.2f} мм; антенн {len(model_of)}, моделей {len(used_models)}, "
          f"полюсов в моделях: {[models[m].n for m in used_models]}, порядок мультиполей до {max_order}")
    trp = pairs["train"].to_numpy()
    gb = coupling_bytes(pairs[trp], model_of, models, nb) / 2 ** 30
    print(f"  наблюдений: обучение {tr.sum()}, тест {(~tr).sum()}; уникальных пар {len(pairs)}; конфигураций: "
          f"обучение {len(np.unique(groups[tr]))}, тест {len(np.unique(groups[~tr]))}; "
          f"матрицы связи обучающей выборки {gb:.2f} ГБ")
    if gb > MEMORY_WARN_GB:
        print(f"  ВНИМАНИЕ: много памяти; уменьшите MAX_ORDER или число полюсов")
    print(f"  расстояния {r.min():.1f}..{r.max():.1f} мм ({r.min() / lam:.2f}..{r.max() / lam:.2f} lambda); "
          f"выше {THRESHOLD_DB:g} дБ: {100 * np.mean(y >= THRESHOLD_DB):.0f}% наблюдений")
    print("  параметров модели (порядок " + "/".join(str(o) for o in range(max_order + 1)) + "): "
          + "/".join(str(sum(2 * n * n_basis(o) for n in n_poles)) for o in range(max_order + 1)))

    # 1. Знак угла
    signs = [THETA_SIGN] if THETA_SIGN in (1, -1) else [1, -1]
    results = {}
    prob = None
    for sgn in signs:
        t_s = time.time()
        print(f"  знак угла {sgn:+d} (регуляризация {REG_DEFAULT:g}):")
        prob = None
        prob = Problem.build(pairs, inv, tr, y, model_of, models, k, sgn, nb)
        init = field_init(models, k, sgn, max_order)
        hist = train(prob, n_poles, THRESHOLD_DB, rng, REG_DEFAULT, max_order, init=init)
        print(f"    ({time.time() - t_s:.0f} с)")
        results[sgn] = (hist[-1][1], hist, init)
    sgn = min(results, key=lambda s: results[s][0])
    if len(signs) > 1:
        print(f"  выбран знак угла {sgn:+d} (функция потерь {results[sgn][0]:.4g} против {results[-sgn][0]:.4g})")
        if sgn == signs[0]:
            prob = None
            prob = Problem.build(pairs, inv, tr, y, model_of, models, k, sgn, nb)
    hist, init = results[sgn][1], results[sgn][2]
    order, reg = max_order, REG_DEFAULT

    # 2. Порядок и регуляризация: кросс-валидация внутри обучающей выборки
    if DO_CV:
        t_s = time.time()
        order, reg, table = cross_validate(prob, groups[tr], n_poles, THRESHOLD_DB, rng, max_order, init=init)
        print(f"  кросс-валидация ({CV_FOLDS} блока, {time.time() - t_s:.0f} с), RMS ошибки с учетом порога, дБ:")
        print("    регуляризация " + "".join(f"  порядок {o}" for o in range(max_order + 1)))
        for rg in REG_GRID:
            print(f"    {rg:<13g} " + "".join(f"  {table[rg][o]:9.2f}" for o in range(max_order + 1)))
        print(f"  выбрано: порядок {order}, регуляризация {reg:g}")
        if len(REG_GRID) > 1 and reg in (min(REG_GRID), max(REG_GRID)):
            print("  (выбрана крайняя регуляризация сетки -- имеет смысл расширить REG_GRID)")
        if reg != REG_DEFAULT:
            print(f"  обучение с регуляризацией {reg:g}:")
            hist = train(prob, n_poles, THRESHOLD_DB, rng, reg, max_order, init=init)
    prob = None

    # 3. Ошибки
    print("  Ошибки (обучение / тест):")
    for name, pred in [("1/r, подгонка уровня", distance_baseline(r, y, THRESHOLD_DB, tr, False)),
                       ("A - B*10lg r", distance_baseline(r, y, THRESHOLD_DB, tr, True))]:
        print(f"    {name:<26} обуч.: {fmt(metrics(y[tr], pred[tr], THRESHOLD_DB))}")
        print(f"    {'':<26} тест : {fmt(metrics(y[~tr], pred[~tr], THRESHOLD_DB))}")
    preds = {}
    for o, _, model in hist:
        pred = predict_all(model, pairs, inv, model_of, models, k, sgn, nb)
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
    return final, hist[order][2], sgn, k


def save_model(path, model, models, model_of, labels, sgn, k, freq):
    np.savez(path, order=model.order, theta_sign=sgn, k_per_mm=k, freq_GHz=freq,
             antenna_labels=np.array(labels), antenna_model=model_of,
             model_names=np.array([m.name for m in models]),
             poles_local_mm=np.array([m.local(sgn) for m in models], dtype=object),
             coefficients=np.array([c for c in model.C], dtype=object),
             basis=np.array([list(a) + [mm] for a, mm in BASIS]),
             ground_plane_depth_mm=np.nan if GROUND_PLANE_DEPTH_MM is None else GROUND_PLANE_DEPTH_MM)


def save_poles_csv(path, models, sgn):
    rows = []
    for m in models:
        loc = m.local(sgn)
        for p in range(m.n):
            rows.append({"model": m.name, "pole": p, "x_local_mm": loc[p, 0], "y_local_mm": loc[p, 1],
                         "z_local_mm": loc[p, 2], "x_sim_mm": m.poles_sim[p, 0], "y_sim_mm": m.poles_sim[p, 1],
                         "z_sim_mm": m.poles_sim[p, 2],
                         "weight_share": np.nan if m.share is None else m.share[p]})
    pd.DataFrame(rows).to_csv(path, index=False)


def main():
    files, yaml_path = POOLS, ANTENNAS_YAML
    args = sys.argv[1:]
    if args and args[0].lower().endswith((".yaml", ".yml")):
        yaml_path, args = args[0], args[1:]
    if args:
        if len(args) % 2:
            raise SystemExit("нужны пары файлов: геометрия S-параметры ...")
        files = list(zip(args[0::2], args[1::2]))
    rng = np.random.default_rng(SEED)
    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Чтение данных (freq_idx = {FREQ_IDX}):")
    obs, n_ant, freq = load_pools(files, FREQ_IDX)
    geometry, labels = obs.attrs["geometry"], obs.attrs["labels"]
    models, model_of = antenna_models(n_ant, labels, yaml_path)

    parts = [("все пулы", obs)] if SHARED_ACROSS_POOLS else \
        [(f"пул {p + 1}", obs[obs["pool"] == p]) for p in sorted(obs["pool"].unique())]
    out = []
    for tag, part in parts:
        part = part.reset_index(drop=True)
        pred, model, sgn, k = run(part, models, model_of, freq, tag, rng)
        out.append(part.assign(S_pred_dB=pred, split=np.where(part["train"], "train", "test")))
        suffix = "all" if SHARED_ACROSS_POOLS else f"pool{part['pool'].iloc[0] + 1}"
        save_model(os.path.join(OUT_DIR, f"model_f{FREQ_IDX}_{suffix}.npz"), model, models, model_of, labels, sgn, k, freq)
        save_poles_csv(os.path.join(OUT_DIR, f"poles_f{FREQ_IDX}_{suffix}.csv"), models, sgn)
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