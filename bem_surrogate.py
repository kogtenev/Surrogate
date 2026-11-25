#!/usr/bin/env python3
"""
Суррогат S-параметров решетки антенн на интегральных уравнениях (тонкие нагруженные рамки над экраном).

Модель (для одной частоты):
  Каждая антенна -- замкнутый контур (рамка) в горизонтальной плоскости на высоте h над экраном:
  прямоугольник (Geometry: Box) или правильный многоугольник вместо окружности (Geometry: Cylinder)
  в собственных осях антенны. Контур разбит на прямые отрезки p длиной ds_p <= lambda / SEG_PER_WAVELENGTH;
  неизвестные -- кусочно-постоянные плотности sigma_p. Для порта i:

      A sigma_i = b_i,      A = -i (G_dir + Gamma G_img) + diag(ds_p beta_p),

      G_pq = int_p int_q e^{ikR} / (4 pi R) ds dt,   R = sqrt(|x - y|^2 + a^2)  (ядро тонкого провода радиуса a),

  G_dir -- между отрезками, G_img -- с изображениями отрезков в экране (z -> -z, экран на z = 0);
  Gamma -- коэффициент отражения экрана (подгоняется, |Gamma| < 1); beta(s) -- импеданс рамки вдоль
  контура, Re beta > 0 (подгоняется, ряд Фурье по нормированной длине контура); b_i = ds_p g(s_p) на
  рамке антенны i, g(s) = sum_{n=-G..G} c_n e^{i n phi}, phi = 2 pi s / P -- профиль возбуждения порта
  (подгоняется). Отклик и S-параметры:

      Y_ij = b_j^T sigma_i (+ zeta_m на диагонали),     S = (E - Y)(E + Y)^{-1},

  zeta_m -- собственная проводимость порта модели (комплексная, подгоняется): рамка с большим beta
  поглощает мощность в своей нагрузке, и без zeta нельзя одновременно описать |S_ii| и |S_ij|.

  Взаимность точная (матрица A симметрична, прием тем же профилем, что и возбуждение). Переотражения,
  затенение рамками соседей и экран учитываются решением системы. beta большой -- рамка "прозрачна",
  остается одна прямая связь (первый член ряда Неймана); beta малый -- рамка сильно рассеивает.
  Матрицы Галеркина: дальние пары -- по средним точкам; близкие (ближе NEAR_FACTOR длин отрезка) --
  квадратура Гаусса по одному отрезку и аналитический интеграл статической части 1/R по другому;
  блоки внутри одной рамки одинаковы у всех копий модели и считаются один раз.

Подгоняется по модулям |S_ij| и |S_ii| в дБ (значения ниже THRESHOLD_DB -- только штраф за завышение)
методом Левенберга-Марквардта с точным якобианом:
  1) рамки прозрачны (beta = BETA_START): профили портов g до гармоники G_START, zeta, Gamma, несколько
     случайных стартов; при THETA_SIGN = 0 -- для обоих знаков угла, выбирается лучший;
  2) профили до G_ORDER;
  3) освобождаются импедансы рамок beta (гармоники до B_ORDER) -- переход к многократному рассеянию.

Рамки по YAML и VTK (для каждой уникальной модели):
  ячейки с атрибутами модели (Attributes) -- ограничивающий объем антенны;
  высота рамки -- z центра ячейки с наибольшим полем |E| (FRAME_Z_TOP_CELLS > 1 -- среднее по стольким
  ячейкам с наибольшим полем);
  Box: прямоугольник наименьшей площади, охватывающий узлы этих ячеек в плане (стороны могут быть не
  параллельны осям), задается тремя точками P0, P1, P2 (P1 - P0 и P2 - P0 -- стороны);
  Cylinder: наименьший круг, охватывающий узлы в плане (центр и радиус).
  Origin и Rotation модели в YAML переводят рамку в собственные оси антенны (как полюсы в
  train_surrogate.py); высота над экраном = z рамки - Screen Z Coordinate.
  Для ParaView в OUT_DIR: frame_model<m>.vtk (в координатах VTK-расчета модели: рамка с отрезками,
  ограничивающая призма, ячейки максимального поля, точки P0, P1, P2; массивы part, index),
  frames_params.csv (параметры рамок), frames_example.vtk (рамки одной тестовой конфигурации в
  координатах решетки, мм, экран на z = 0; массивы antenna, model).

YAML -- как для train_surrogate.py, плюс:
  Screen Z Coordinate: <z экрана в единицах сетки VTK>         (верхний уровень)
  Geometry: Box | Cylinder                                       (в каждой уникальной модели)
  Number of Poles не нужен (если есть -- не используется).

Данные: те же файлы геометрии и S-параметров, что для train_surrogate.py (диагональ S_ii -- строки
ant_i = ant_j, см. merge_diag.py). Обучение: eval_id % 3 == 1, тест -- остальные.

Печатается время: рамки из VTK, сборка матриц, обучение, предсказание всех S одной конфигурации.
Результаты (OUT_DIR): predictions_f*.csv, bem_model_f*.npz, рамки, графики и диагностика промахов.

Запуск:
    python3 bem_surrogate.py antennas.yaml g1.csv s1.csv g2.csv s2.csv g3.csv s3.csv
    python3 bem_surrogate.py                    # файлы из настроек ниже
Нужны: numpy, pandas, scipy, pyyaml, matplotlib и train_surrogate.py в той же папке.
"""
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_surrogate as T  # noqa: E402  (чтение данных, YAML/VTK, метрики, диагностика)

# ============================== Настройки ==============================
POOLS = [  # (файл геометрии, файл S-параметров) для каждого пула
    ("pool1_geom.csv", "pool1_s.csv"),
    ("pool2_geom.csv", "pool2_s.csv"),
    ("pool3_geom.csv", "pool3_s.csv"),
]
ANTENNAS_YAML = "antennas.yaml"
FREQ_IDX = 0
THRESHOLD_DB = -35.0
THETA_SIGN = 0               # +1: угол против часовой стрелки, -1: по часовой, 0: выбрать по обучающей выборке
MESH_TO_MM = 1.0             # единица длины сетки VTK (Origin, Screen Z Coordinate) в мм
FRAME_Z_TOP_CELLS = 1        # высота рамки -- z центра ячейки с наибольшим полем (или среднее по N таким)
SEG_PER_WAVELENGTH = 12      # отрезки контура не длиннее lambda / SEG_PER_WAVELENGTH
BOX_MIN_SEG_PER_SIDE = 2     # отрезков на сторону прямоугольника не меньше
CYLINDER_MIN_SIDES = 12      # окружность -- правильный многоугольник не менее чем с CYLINDER_MIN_SIDES сторонами
WIRE_RADIUS_FRACTION = 0.25  # радиус тонкого провода a = доля наибольшей длины отрезка
NEAR_FACTOR = 2.5            # пары отрезков ближе NEAR_FACTOR * длина отрезка считаются точнее
NQ_NEAR = 6                  # точек Гаусса для близких пар разных антенн
NQ_SELF = 16                 # точек Гаусса внутри одной рамки (один раз на модель)
G_START = 1                  # гармоники профиля порта на первом этапе (+-G_START)
G_ORDER = 3                  # гармоники профиля порта в итоговой модели (+-G_ORDER)
B_ORDER = 1                  # гармоники импеданса рамки (+-B_ORDER); None -- рамки остаются прозрачными
BETA_START = 10.0            # стартовый импеданс (прозрачные рамки, только прямая связь)
FIT_GAMMA = True             # подгонять коэффициент отражения экрана; иначе GAMMA_FIXED
GAMMA_FIXED = -1.0
GAMMA_STARTS = [-0.8, 0.0, 0.8]  # стартовые Gamma для случайных стартов (по кругу)
N_RESTARTS = 4               # случайных стартов на первом этапе (для каждого знака угла)
RESTART_ITER = 25            # итераций на старт
MAX_ITER = 80                # итераций на этап
REG_C = 1e-8                 # гребневая регуляризация коэффициентов профилей (относительно их масштаба)
REG_BETA = 1.0               # регуляризация импеданса к стартовому (в единицах функции потерь, дБ^2)
REG_ZETA = 1e-6             # то же для zeta (стартовое значение -- согласованные порты)
REG_GAMMA = 1e-3             # то же для коэффициента отражения экрана (к 0)
CHUNK = 16                   # конфигураций в пакете (память ~ CHUNK * M^2)
CACHE_GB = 2.0               # матрицы обучающих конфигураций хранить в памяти, если их меньше
SEED = 0
OUT_DIR = "bem_out"
C_MM_PER_NS = 299.792458
# ======================================================================

DB = 20.0 / np.log(10.0)
INV4PI = 1.0 / (4.0 * np.pi)


def _sync_settings():
    T.MESH_TO_MM = MESH_TO_MM
    T.THRESHOLD_DB = THRESHOLD_DB
    T.FREQ_IDX = FREQ_IDX
    T.SHARED_ACROSS_POOLS = True


# ---------------------------------------------------------------------
#  YAML
# ---------------------------------------------------------------------
def load_yaml(path):
    """-> ({ID антенны: имя модели}, {имя модели: параметры}, z экрана в единицах сетки или None)."""
    import yaml
    raw = open(path, encoding="utf-8").read()
    data = None
    try:
        data = yaml.safe_load(raw)
        if not (isinstance(data, dict) and isinstance(T._get(data, "antennas", required=False), list)):
            data = None
    except yaml.YAMLError:
        data = None
    if data is None:
        try:
            data = yaml.safe_load(T._fix_yaml_text(raw))
        except yaml.YAMLError as e:
            raise SystemExit(f"{path}: не удалось разобрать YAML: {e}")
        print(f"  {path}: YAML исправлен ('-ID' -> '- ID', отступы ключей элементов списка)")
    zs = T._get(data, "screenzcoordinate", "screenz", "screen", required=False)
    zs = None if zs is None else float(zs)
    name = lambda v: str(int(v)) if isinstance(v, (int, float)) or str(v).strip().lstrip("-").isdigit() else str(v).strip()
    ants = {}
    for item in T._get(data, "antennas", what="Antennas"):
        ants[int(T._get(item, "id", what="ID"))] = name(T._get(item, "uniquemodel", "model", what="Unique model"))
    base = os.path.dirname(os.path.abspath(path))
    models = {}
    for key, cfg in T._get(data, "uniquemodels", "models", what="Unique models").items():
        fname = str(T._get(cfg, "solutionfile", "file", what="Solution File")).strip()
        geo = str(T._get(cfg, "geometry", "shape", what="Geometry")).strip().lower()
        if geo.startswith("cyl") or geo.startswith("circ"):
            geo = "cylinder"
        elif geo.startswith(("box", "rect", "paral", "prism", "cub")):
            geo = "box"
        else:
            raise SystemExit(f"YAML: модель {key}: Geometry должно быть Box или Cylinder, а не '{geo}'")
        origin = np.array(T._get(cfg, "origin", what="Origin"), dtype=float).ravel()[:3]
        if len(origin) == 2:
            origin = np.append(origin, 0.0)
        models[name(key)] = {
            "file": fname if os.path.isabs(fname) else os.path.join(base, fname),
            "real": str(T._get(cfg, "realfieldname", "real", what="Real Field Name")).strip(),
            "imag": str(T._get(cfg, "imagfieldname", "imag", what="Imag Field Name")).strip(),
            "attributes": [int(a) for a in np.atleast_1d(T._get(cfg, "attributes", what="Attributes"))],
            "attr_field": T._get(cfg, "attributefieldname", "attributefield", required=False),
            "origin": origin, "rotation": float(T._get(cfg, "rotation", "angle", what="Rotation")),
            "geometry": geo,
        }
    missing = sorted(set(ants.values()) - set(models))
    if missing:
        raise SystemExit(f"YAML: модели {missing} указаны у антенн, но не описаны в Unique models")
    return ants, models, zs


# ---------------------------------------------------------------------
#  Рамки по VTK
# ---------------------------------------------------------------------
def min_area_rect(H):
    """Прямоугольник наименьшей площади вокруг выпуклой оболочки H (n, 2) -> P0, P1, P2."""
    best = None
    n = len(H)
    for q in range(n):
        e = H[(q + 1) % n] - H[q]
        L = np.hypot(*e)
        if L <= 0:
            continue
        u = e / L
        v = np.array([-u[1], u[0]])
        pu, pv = H @ u, H @ v
        area = np.ptp(pu) * np.ptp(pv)
        if best is None or area < best[0]:
            best = (area, u, v, pu.min(), pu.max(), pv.min(), pv.max())
    _, u, v, a0, a1, b0, b1 = best
    return a0 * u + b0 * v, a1 * u + b0 * v, a0 * u + b1 * v


def min_circle(P, seed=0):
    """Наименьший охватывающий круг (алгоритм Велцля, итеративно) -> центр, радиус."""
    P = P[np.random.default_rng(seed).permutation(len(P))]

    def circ2(a, b):
        c = 0.5 * (a + b)
        return c, np.hypot(*(a - c))

    def circ3(a, b, c):
        ax, ay = a
        bx, by = b
        cx, cy = c
        d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
        if abs(d) < 1e-300:
            pts = [a, b, c]
            best = max(((i, j) for i in range(3) for j in range(i + 1, 3)),
                       key=lambda ij: np.hypot(*(pts[ij[0]] - pts[ij[1]])))
            return circ2(pts[best[0]], pts[best[1]])
        ux = ((ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by) * (cy - ay) + (cx * cx + cy * cy) * (ay - by)) / d
        uy = ((ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by) * (ax - cx) + (cx * cx + cy * cy) * (bx - ax)) / d
        o = np.array([ux, uy])
        return o, np.hypot(*(a - o))

    eps = 1e-12 * max(np.ptp(P, axis=0).max(), 1e-30)
    c, r = P[0].copy(), 0.0
    for i in range(1, len(P)):
        if np.hypot(*(P[i] - c)) > r + eps:
            c, r = P[i].copy(), 0.0
            for j in range(i):
                if np.hypot(*(P[j] - c)) > r + eps:
                    c, r = circ2(P[i], P[j])
                    for q in range(j):
                        if np.hypot(*(P[q] - c)) > r + eps:
                            c, r = circ3(P[i], P[j], P[q])
    return c, r


def compute_frame(mname, cfg, z_screen):
    """Рамка уникальной модели по ее VTK (координаты расчета, единицы сетки)."""
    from scipy.spatial import ConvexHull
    t0 = time.time()
    vtk = T.read_vtk_ascii(cfg["file"])
    types, offs, conn = vtk["cell_types"], vtk["offsets"], vtk["conn"]
    cand = [cfg["attr_field"]] if cfg["attr_field"] else T.ATTRIBUTE_FIELD_NAMES
    attr = next((vtk["cell"][n][:, 0] for n in cand if n in vtk["cell"]), None)
    if attr is None:
        raise SystemExit(f"{cfg['file']}: нет массива атрибутов ячеек (искали {cand}); есть {sorted(vtk['cell'])}")
    sel = np.isin(np.rint(attr).astype(int), cfg["attributes"]) & np.isin(types, list(T.VTK_3D))
    if not sel.any():
        raise SystemExit(f"{cfg['file']}: нет объемных ячеек с атрибутами {cfg['attributes']}")
    P = vtk["points"]
    cells = np.flatnonzero(sel)
    cen = np.zeros((len(cells), 3))
    e2 = np.zeros(len(cells))
    node_ids = []
    for t in np.unique(types[cells]):
        loc = np.flatnonzero(types[cells] == t)
        cc = cells[loc]
        nc = T.CORNERS[t]
        corner = conn[offs[cc][:, None] + np.arange(nc)[None, :]]
        node_ids.append(corner.ravel())
        cen[loc] = P[corner].mean(axis=1)
        Er, _ = T._cell_field(vtk, cfg["real"], cc, corner, nc)
        Ei, _ = T._cell_field(vtk, cfg["imag"], cc, corner, nc)
        e2[loc] = np.mean(np.sum(Er ** 2 + Ei ** 2, axis=2), axis=1)       # средний |E|^2 по углам
    nodes = P[np.unique(np.concatenate(node_ids))]
    top = np.argsort(e2)[::-1][:max(1, int(FRAME_Z_TOP_CELLS))]
    z_frame = float(np.mean(cen[top, 2]))
    xy = nodes[:, :2]
    hull = xy[ConvexHull(xy).vertices]
    fr = {"name": mname, "kind": cfg["geometry"], "z": z_frame, "zmin": float(nodes[:, 2].min()),
          "zmax": float(nodes[:, 2].max()), "maxpts": cen[top], "origin": cfg["origin"], "rotation": cfg["rotation"],
          "z_screen": z_screen}
    if cfg["geometry"] == "box":
        P0, P1, P2 = min_area_rect(hull)
        fr.update(P0=P0, P1=P1, P2=P2)
        msg = (f"прямоугольник P0 = ({P0[0]:.4g}, {P0[1]:.4g}), P1 = ({P1[0]:.4g}, {P1[1]:.4g}), "
               f"P2 = ({P2[0]:.4g}, {P2[1]:.4g}); стороны {np.hypot(*(P1 - P0)):.4g} x {np.hypot(*(P2 - P0)):.4g}")
    else:
        c, r = min_circle(hull)
        fr.update(center=c, radius=r)
        msg = f"круг: центр ({c[0]:.4g}, {c[1]:.4g}), радиус {r:.4g}"
    h = (z_frame - z_screen) * MESH_TO_MM if z_screen is not None else np.inf
    print(f"  модель {mname}: {os.path.basename(cfg['file'])}, ячеек {len(cells)}: {msg}; z рамки {z_frame:.4g} "
          f"(ячейки max |E|), высота над экраном {h:.3g} мм; z узлов {fr['zmin']:.4g}..{fr['zmax']:.4g} "
          f"({time.time() - t0:.1f} с)")
    return fr


class Contour:
    """Разбиение рамки модели на отрезки: в координатах расчета (мм) и в собственных осях антенны."""

    def __init__(self, fr, lam, sgn, a):
        dmax = lam / SEG_PER_WAVELENGTH
        s = MESH_TO_MM
        if fr["kind"] == "box":
            P0, P1, P2 = fr["P0"] * s, fr["P1"] * s, fr["P2"] * s
            verts = np.array([P0, P1, P1 + P2 - P0, P2])
            nside = [max(BOX_MIN_SEG_PER_SIDE, int(np.ceil(np.hypot(*(verts[(q + 1) % 4] - verts[q])) / dmax)))
                     for q in range(4)]
        else:
            c, r = fr["center"] * s, fr["radius"] * s
            n = max(CYLINDER_MIN_SIDES, int(np.ceil(2 * np.pi * r / dmax)))
            t = 2 * np.pi * np.arange(n) / n
            verts = c + r * np.column_stack([np.cos(t), np.sin(t)])
            nside = [1] * n
        area = 0.5 * np.sum(verts[:, 0] * np.roll(verts[:, 1], -1) - np.roll(verts[:, 0], -1) * verts[:, 1])
        if area < 0:                                               # обход против часовой стрелки
            verts, nside = verts[::-1].copy(), nside[::-1]
        A, B = [], []
        for q in range(len(verts)):
            v0, v1 = verts[q], verts[(q + 1) % len(verts)]
            tt = np.linspace(0, 1, nside[q] + 1)
            pts = v0 + tt[:, None] * (v1 - v0)
            A.append(pts[:-1])
            B.append(pts[1:])
        self.a_sim, self.b_sim = np.vstack(A), np.vstack(B)          # мм, координаты расчета
        self.verts_sim = verts
        self.ds = np.hypot(*(self.b_sim - self.a_sim).T)
        self.M = len(self.ds)
        self.perimeter = self.ds.sum()
        sm = np.cumsum(self.ds) - 0.5 * self.ds
        self.phi = 2 * np.pi * sm / self.perimeter
        self.h = (fr["z"] - fr["z_screen"]) * s if fr["z_screen"] is not None else None
        if self.h is not None and self.h < 2 * a:
            print(f"  ВНИМАНИЕ: модель {fr['name']}: рамка на высоте {self.h:.3g} мм над экраном (меньше 2a) -- "
                  f"поднята до {2 * a:.3g} мм")
            self.h = 2 * a
        # без экрана -- высота относительно плоскости центров антенн (z = Origin)
        self.hz = (fr["z"] - fr["origin"][2]) * s if self.h is None else self.h
        o = fr["origin"][:2] * s
        R = T.rotation_z(np.array(sgn * np.deg2rad(fr["rotation"])))[:2, :2]
        self.a_loc = (self.a_sim - o) @ R                            # R^T (p - origin)
        self.b_loc = (self.b_sim - o) @ R
        self.name = fr["name"]

    def self_blocks(self, k, a, screen):
        """Блоки рамки с самой собой и со своим изображением (не зависят от положения антенны)."""
        z = np.full((self.M, 1), self.hz)
        A0 = np.hstack([self.a_loc, z])
        A1 = np.hstack([self.b_loc, z])
        p, q = np.meshgrid(np.arange(self.M), np.arange(self.M), indexing="ij")
        p, q = p.ravel(), q.ravel()
        Gd = pair_integrals(A0[p], A1[p], A0[q], A1[q], k, a, NQ_SELF).reshape(self.M, self.M)
        self.Gd = 0.5 * (Gd + Gd.T)
        if screen:
            I0, I1 = A0.copy(), A1.copy()
            I0[:, 2] = I1[:, 2] = -self.hz
            Gi = pair_integrals(A0[p], A1[p], I0[q], I1[q], k, a, NQ_SELF).reshape(self.M, self.M)
            self.Gi = 0.5 * (Gi + Gi.T)
        else:
            self.Gi = np.zeros((self.M, self.M), complex)


# ---------------------------------------------------------------------
#  Интегралы по отрезкам
# ---------------------------------------------------------------------
def _gauss(nq):
    t, w = np.polynomial.legendre.leggauss(nq)
    return 0.5 * (t + 1), 0.5 * w


def _pair_one_way(A0, A1, B0, B1, k, a, nq):
    """int_A int_B e^{ikR}/(4 pi R): по A -- Гаусс, статическая часть по B -- аналитически."""
    t, w = _gauss(nq)
    dA, dB = A1 - A0, B1 - B0
    LA, LB = np.linalg.norm(dA, axis=1), np.linalg.norm(dB, axis=1)
    u = dB / LB[:, None]
    X = A0[:, None, :] + t[None, :, None] * dA[:, None, :]          # (n, nq, 3)
    Yq = B0[:, None, :] + t[None, :, None] * dB[:, None, :]
    wv = X - B0[:, None, :]
    t0 = np.einsum("nqc,nc->nq", wv, u)
    rho = np.sqrt(np.maximum(np.einsum("nqc,nqc->nq", wv, wv) - t0 ** 2, 0.0) + a * a)
    stat = np.arcsinh((LB[:, None] - t0) / rho) + np.arcsinh(t0 / rho)  # int_B dt / R
    stat = LA * (stat @ w)
    d = X[:, :, None, :] - Yq[:, None, :, :]
    R = np.sqrt(np.einsum("nijc,nijc->nij", d, d) + a * a)
    dyn = np.expm1(1j * k * R) / R
    dyn = LA * LB * np.einsum("nij,i,j->n", dyn, w, w)
    return INV4PI * (stat + dyn)


def pair_integrals(A0, A1, B0, B1, k, a, nq, block=4096):
    """Интегралы для n пар отрезков (симметризованы: среднее A->B и B->A)."""
    out = np.empty(len(A0), complex)
    for s in range(0, len(A0), block):
        e = slice(s, s + block)
        out[e] = 0.5 * (_pair_one_way(A0[e], A1[e], B0[e], B1[e], k, a, nq)
                        + _pair_one_way(B0[e], B1[e], A0[e], A1[e], k, a, nq))
    return out


# ---------------------------------------------------------------------
#  Задача: конфигурации, матрицы, наблюдения
# ---------------------------------------------------------------------
class Layout:
    """Размещение неизвестных: антенны по порядку, отрезки рамки модели антенны подряд."""

    def __init__(self, model_of, contours, G):
        self.N = len(model_of)
        self.model_of = model_of
        self.U = len(contours)
        sizes = np.array([contours[m].M for m in model_of])
        self.off = np.concatenate([[0], np.cumsum(sizes)])
        self.M = int(self.off[-1])
        self.seg_ant = np.repeat(np.arange(self.N), sizes)
        self.seg_model = model_of[self.seg_ant]
        self.ds = np.concatenate([contours[m].ds for m in model_of])
        self.phi = np.concatenate([contours[m].phi for m in model_of])
        self.set_order(G)
        self.members = [np.flatnonzero(model_of == m) for m in range(self.U)]

    def set_order(self, G):
        self.G = G
        n = np.arange(-G, G + 1)
        self.Phi = self.ds[:, None] * np.exp(1j * self.phi[:, None] * n[None, :])   # (M, 2G+1)


def lattice_positions(cfg, geometry, n_ant):
    X = np.empty((len(cfg), n_ant))
    Y = np.empty((len(cfg), n_ant))
    TH = np.empty((len(cfg), n_ant))
    pools = cfg["pool"].to_numpy()
    for pid, (row_of, GX, GY, GT) in geometry.items():
        m = pools == pid
        if m.any():
            r = cfg.loc[m, "eval_id"].map(row_of).to_numpy()
            X[m], Y[m], TH[m] = GX[r], GY[r], GT[r]
    return X, Y, TH


def segments_global(x, y, th, contours, model_of, sgn):
    """Концы отрезков всех рамок одной конфигурации (M, 3)."""
    A, B = [], []
    for i, m in enumerate(model_of):
        c = contours[m]
        ang = sgn * np.deg2rad(th[i])
        R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
        z = np.full((c.M, 1), c.hz)
        A.append(np.hstack([np.array([x[i], y[i]]) + c.a_loc @ R.T, z]))
        B.append(np.hstack([np.array([x[i], y[i]]) + c.b_loc @ R.T, z]))
    return np.vstack(A), np.vstack(B)


def assemble(x, y, th, contours, lay, k, a, sgn, screen):
    """Матрицы Галеркина G_dir, G_img (M, M) одной конфигурации."""
    A0, A1 = segments_global(x, y, th, contours, lay.model_of, sgn)
    mid = 0.5 * (A0 + A1)
    ds = lay.ds
    w = ds[:, None] * ds[None, :] * INV4PI
    near_len = NEAR_FACTOR * np.maximum(ds[:, None], ds[None, :])
    other = lay.seg_ant[:, None] != lay.seg_ant[None, :]
    out = []
    for img in ([False, True] if screen else [False]):
        midq = mid.copy()
        if img:
            midq[:, 2] = -midq[:, 2]
        d = mid[:, None, :] - midq[None, :, :]
        R = np.sqrt(np.einsum("pqc,pqc->pq", d, d) + a * a)
        Gm = w * np.exp(1j * k * R) / R
        for i, m in enumerate(lay.model_of):                       # блоки рамка-рамка: заранее
            sl = slice(lay.off[i], lay.off[i + 1])
            Gm[sl, sl] = contours[m].Gi if img else contours[m].Gd
        near = other & (R < near_len)
        p, q = np.nonzero(np.triu(near, 1))
        if len(p):
            B0, B1 = A0[q].copy(), A1[q].copy()
            if img:
                B0[:, 2], B1[:, 2] = -B0[:, 2], -B1[:, 2]
            v = pair_integrals(A0[p], A1[p], B0, B1, k, a, NQ_NEAR)
            Gm[p, q] = v
            Gm[q, p] = v
        out.append(Gm)
    if not screen:
        out.append(None)
    return out[0], out[1]


class Problem:
    """Набор конфигураций: матрицы (по желанию -- в памяти) и наблюдения."""

    def __init__(self, cfg, geometry, contours, lay, k, a, sgn, screen, obs=None, diag=None, cache=True):
        self.cfg = cfg.reset_index(drop=True)
        self.C, self.N = len(cfg), lay.N
        self.contours, self.lay, self.k, self.a, self.sgn, self.screen = contours, lay, k, a, sgn, screen
        self.X, self.Y, self.TH = lattice_positions(self.cfg, geometry, lay.N)
        need = 16.0 * lay.M ** 2 * (2 if screen else 1) * self.C / 2 ** 30
        self.mats = None
        if cache and need <= CACHE_GB:
            self.mats = [self._assemble(c) for c in range(self.C)]
        self.Yobs = np.full((self.C, lay.N, lay.N), np.nan)
        cid = {key: q for q, key in enumerate(zip(self.cfg["pool"].tolist(), self.cfg["eval_id"].tolist()))}
        for df, col, jcol in ((obs, "S_ij_dB", "j"), (diag, "S_ii_dB", "i")):
            if df is None or len(df) == 0:
                continue
            c = np.array([cid.get(key, -1) for key in zip(df["pool"].tolist(), df["eval_id"].tolist())])
            ok = c >= 0
            self.Yobs[c[ok], df["i"].to_numpy()[ok], df[jcol].to_numpy()[ok]] = df[col].to_numpy(float)[ok]
        self.mask = ~np.isnan(self.Yobs)
        self.n_obs = int(self.mask.sum())
        d = np.arange(lay.N)
        self.diag_mask = np.zeros_like(self.mask)
        self.diag_mask[:, d, d] = True

    def _assemble(self, c):
        return assemble(self.X[c], self.Y[c], self.TH[c], self.contours, self.lay, self.k, self.a, self.sgn,
                        self.screen)

    def matrices(self, idx):
        if self.mats is not None:
            Gd = np.stack([self.mats[c][0] for c in idx])
            Gi = np.stack([self.mats[c][1] for c in idx]) if self.screen else None
            return Gd, Gi
        ms = [self._assemble(c) for c in idx]
        return np.stack([m[0] for m in ms]), (np.stack([m[1] for m in ms]) if self.screen else None)


# ---------------------------------------------------------------------
#  Параметры
# ---------------------------------------------------------------------
class Params:
    """Параметры моделей: профили портов c (комплексные), импедансы beta = exp(rho) + i chi,
    собственная проводимость порта zeta = u + i v; общий коэффициент отражения экрана Gamma = tanh(gamma)."""

    def __init__(self, U, G, B):
        self.U, self.G, self.B = U, G, B
        self.c = [np.zeros(2 * G + 1, complex) for _ in range(U)]
        nb = 2 * B + 1
        self.rho = [np.r_[np.log(BETA_START), np.zeros(nb - 1)] for _ in range(U)]
        self.chi = [np.zeros(nb) for _ in range(U)]
        self.u = np.ones(U)
        self.v = np.zeros(U)
        self.gamma = 0.0 if FIT_GAMMA else np.arctanh(np.clip(GAMMA_FIXED, -0.999999, 0.999999))

    @property
    def nc(self):
        return 2 * self.G + 1

    @property
    def nb(self):
        return 2 * self.B + 1

    def per_model(self):
        return 2 * self.nc + 2 * self.nb + 2

    def size(self):
        return self.U * self.per_model() + 1

    def slices(self, m):
        o = m * self.per_model()
        nc, nb = self.nc, self.nb
        return {"cr": slice(o, o + nc), "ci": slice(o + nc, o + 2 * nc), "rho": slice(o + 2 * nc, o + 2 * nc + nb),
                "chi": slice(o + 2 * nc + nb, o + 2 * nc + 2 * nb), "u": o + 2 * nc + 2 * nb,
                "v": o + 2 * nc + 2 * nb + 1}

    def pack(self):
        x = np.zeros(self.size())
        for m in range(self.U):
            s = self.slices(m)
            x[s["cr"]], x[s["ci"]] = self.c[m].real, self.c[m].imag
            x[s["rho"]], x[s["chi"]] = self.rho[m], self.chi[m]
            x[s["u"]], x[s["v"]] = self.u[m], self.v[m]
        x[-1] = self.gamma
        return x

    def unpack(self, x):
        for m in range(self.U):
            s = self.slices(m)
            self.c[m] = x[s["cr"]] + 1j * x[s["ci"]]
            self.rho[m], self.chi[m] = x[s["rho"]].copy(), x[s["chi"]].copy()
            self.u[m], self.v[m] = x[s["u"]], x[s["v"]]
        self.gamma = x[-1]
        return self

    def copy(self):
        p = Params(self.U, self.G, self.B)
        return p.unpack(self.pack())

    def with_order(self, G):
        p = Params(self.U, G, self.B)
        for m in range(self.U):
            g0 = self.G
            n = min(G, g0)
            p.c[m][G - n:G + n + 1] = self.c[m][g0 - n:g0 + n + 1]
            p.rho[m], p.chi[m] = self.rho[m].copy(), self.chi[m].copy()
        p.u, p.v, p.gamma = self.u.copy(), self.v.copy(), self.gamma
        return p

    def with_beta_order(self, B):
        p = Params(self.U, self.G, B)
        for m in range(self.U):
            p.c[m] = self.c[m].copy()
            n = min(B, self.B)
            p.rho[m][:2 * n + 1] = self.rho[m][:2 * n + 1]
            p.chi[m][:2 * n + 1] = self.chi[m][:2 * n + 1]
        p.u, p.v, p.gamma = self.u.copy(), self.v.copy(), self.gamma
        return p

    def zeta(self):
        return self.u + 1j * self.v

    def Gamma(self):
        return np.tanh(self.gamma)


def beta_basis(phi, B):
    """Вещественный базис Фурье: 1, cos phi, sin phi, cos 2phi, ... (M, 2B+1)."""
    cols = [np.ones_like(phi)]
    for n in range(1, B + 1):
        cols += [np.cos(n * phi), np.sin(n * phi)]
    return np.column_stack(cols)


def model_vectors(par, lay):
    """Правые части B (M, N) и диагональ D (M,) -- одинаковы для всех конфигураций."""
    gseg = np.einsum("pq,pq->p", lay.Phi, np.stack([par.c[m] for m in lay.seg_model]))
    Bm = np.zeros((lay.M, lay.N), complex)
    Bm[np.arange(lay.M), lay.seg_ant] = gseg
    Fb = beta_basis(lay.phi, par.B)
    rho = np.einsum("pq,pq->p", Fb, np.stack([par.rho[m] for m in lay.seg_model]))
    chi = np.einsum("pq,pq->p", Fb, np.stack([par.chi[m] for m in lay.seg_model]))
    erho = np.exp(rho)
    D = lay.ds * (erho + 1j * chi)
    return Bm, D, erho, Fb


def solve_chunk(par, prob, idx, Bm, D):
    Gd, Gi = prob.matrices(idx)
    Gam = par.Gamma()
    A = -1j * (Gd if Gi is None else Gd + Gam * Gi)
    A[:, np.arange(prob.lay.M), np.arange(prob.lay.M)] += D
    Sig = np.linalg.solve(A, np.broadcast_to(Bm, (len(idx),) + Bm.shape))
    Yp = np.einsum("pk,cpl->ckl", Bm, Sig)
    zeta = par.zeta()
    n = np.arange(prob.N)
    Yp[:, n, n] += zeta[prob.lay.model_of]
    E = np.eye(prob.N)
    W = np.linalg.inv(E + Yp)
    S = 2.0 * W - E
    return S, W, Sig, Gi


def predict(par, prob):
    """S (C, N, N) для всех конфигураций задачи."""
    Bm, D, _, _ = model_vectors(par, prob.lay)
    out = np.empty((prob.C, prob.N, prob.N), complex)
    for s in range(0, prob.C, CHUNK):
        idx = np.arange(s, min(prob.C, s + CHUNK))
        out[idx] = solve_chunk(par, prob, idx, Bm, D)[0]
    return out


def evaluate(par, prob, thr, need_jac=True, free=None):
    """Невязки (дБ) по наблюдаемым элементам S и якобиан (n_obs, число параметров)."""
    lay = prob.lay
    Bm, D, erho, Fb = model_vectors(par, lay)
    P = par.size()
    res_all, J_all = [], []
    nb = par.nb
    if need_jac:
        # d D_p / d(rho, chi) для отрезков модели m: (M, U*2nb)
        Eb = np.zeros((lay.M, par.U * 2 * nb), complex)
        for m in range(par.U):
            segs = lay.seg_model == m
            Eb[segs, 2 * nb * m:2 * nb * m + nb] = (lay.ds * erho)[segs, None] * Fb[segs]
            Eb[segs, 2 * nb * m + nb:2 * nb * (m + 1)] = 1j * lay.ds[segs, None] * Fb[segs]
        want_beta = free is None or any(np.any(free[par.slices(m)["rho"]]) for m in range(par.U))
        want_gamma = prob.screen and (free is None or free[-1])
    for s in range(0, prob.C, CHUNK):
        idx = np.arange(s, min(prob.C, s + CHUNK))
        S, W, Sig, Gi = solve_chunk(par, prob, idx, Bm, D)
        msk = prob.mask[idx]
        So = S[msk]
        y = prob.Yobs[idx][msk]
        mdb = DB * np.log(np.abs(So) + 1e-300)
        res_all.append(T.thresholded_residual(mdb, y, thr))
        if not need_jac:
            continue
        active = (y >= thr) | (mdb > thr)
        scw = np.conj(So) * DB / (np.abs(So) ** 2 + 1e-300) * active
        J = np.zeros((len(So), P))
        Q = np.einsum("cik,cpk->cip", W, Sig)                        # (c, N, M)
        # профили портов: V[c, i, l, :] = sum_{p in l} Q[c, i, p] Phi[p, :]
        V = np.add.reduceat(Q[..., None] * lay.Phi[None, None], lay.off[:-1], axis=2)    # (c, N, N, nc)
        for m in range(par.U):
            mem = lay.members[m]
            if not len(mem):
                continue
            sl = par.slices(m)
            F = np.einsum("cilq,clj->cijq", V[:, :, mem, :], W[:, mem, :], optimize=True)
            dS = -2.0 * (F + F.transpose(0, 2, 1, 3))
            h = scw[:, None] * dS[msk]
            J[:, sl["cr"]] = h.real
            J[:, sl["ci"]] = -h.imag
            Wm = np.einsum("cik,ckj->cij", W[:, :, mem], W[:, mem, :])  # sum_{k in m} W_ik W_kj
            dz = -2.0 * Wm[msk] * scw
            J[:, sl["u"]] = dz.real
            J[:, sl["v"]] = (dz * 1j).real
        if want_beta:
            dSb = 2.0 * np.einsum("cip,cjp,pt->cijt", Q, Q, Eb, optimize=True)[msk]      # (n, U*2nb)
            hb = (scw[:, None] * dSb).real
            for m in range(par.U):
                sl = par.slices(m)
                J[:, sl["rho"]] = hb[:, 2 * nb * m:2 * nb * m + nb]
                J[:, sl["chi"]] = hb[:, 2 * nb * m + nb:2 * nb * (m + 1)]
        if want_gamma:
            QS = np.einsum("cip,cpq,cjq->cij", Q, Gi, Q, optimize=True)
            dg = -2j * QS[msk] * (1.0 - par.Gamma() ** 2)
            J[:, -1] = (scw * dg).real
        J_all.append(J)
    res = np.concatenate(res_all) if res_all else np.zeros(0)
    return res, (np.vstack(J_all) if need_jac else None)


# ---------------------------------------------------------------------
#  Подгонка
# ---------------------------------------------------------------------
def regularization(par):
    """Веса и априорные значения гребневой регуляризации."""
    P = par.size()
    rw = np.zeros(P)
    prior = np.zeros(P)
    for m in range(par.U):
        sl = par.slices(m)
        cs = max(np.max(np.abs(par.c[m])), 1e-300)
        n = np.abs(np.arange(-par.G, par.G + 1))
        wc = np.sqrt(REG_C) * (1.0 + n) / cs
        rw[sl["cr"]] = wc
        rw[sl["ci"]] = wc
        rw[sl["rho"]] = np.sqrt(REG_BETA)
        rw[sl["chi"]] = np.sqrt(REG_BETA)
        prior[sl["rho"].start] = np.log(BETA_START)
        rw[sl["u"]] = np.sqrt(REG_ZETA)
        prior[sl["u"]] = 1.0
        rw[sl["v"]] = np.sqrt(REG_ZETA)
    rw[-1] = np.sqrt(REG_GAMMA)
    return rw, prior


def free_mask(par, beta, gamma):
    f = np.ones(par.size(), bool)
    for m in range(par.U):
        sl = par.slices(m)
        f[sl["rho"]] = beta
        f[sl["chi"]] = beta
    f[-1] = gamma
    return f


def lm_fit(par, prob, thr, free, max_iter, tol=1e-7, log=None):
    from scipy.linalg import cho_factor, cho_solve, LinAlgError
    rw, prior = regularization(par)
    rw2 = rw ** 2
    x = par.pack()

    def cost_of(x, res):
        return 0.5 * (res @ res + rw2 @ ((x - prior) ** 2))

    res, J = evaluate(par, prob, thr, True, free)
    cost = cost_of(x, res)
    fi = np.flatnonzero(free)
    mu = 1e-3
    rel = 1.0
    for it in range(max_iter):
        Jf = J[:, fi]
        H = Jf.T @ Jf
        H[np.diag_indices(len(fi))] += rw2[fi]
        g = Jf.T @ res + rw2[fi] * (x - prior)[fi]
        dH = np.diag(H).copy()
        improved = False
        while mu < 1e10:
            try:
                d = -cho_solve(cho_factor(H + mu * np.diag(dH + 1e-12 * dH.max())), g)
            except LinAlgError:
                mu *= 10
                continue
            xn = x.copy()
            xn[fi] += d
            rn, _ = evaluate(par.copy().unpack(xn), prob, thr, False)
            cn = cost_of(xn, rn)
            if np.isfinite(cn) and cn < cost:
                rel = (cost - cn) / max(cost, 1e-300)
                x, cost = xn, cn
                par.unpack(x)
                res, J = evaluate(par, prob, thr, True, free)
                mu = max(mu / 3, 1e-9)
                improved = True
                break
            mu *= 4
        if log and (it % 10 == 0):
            log(f"      итерация {it}: функция потерь {cost:.5g}")
        if not improved or rel < tol:
            break
    par.unpack(x)
    return cost


def self_admittance(par, prob):
    """Собственные проводимости рамок без zeta: diag(B^T A^{-1} B) -> (C, N)."""
    Bm, D, _, _ = model_vectors(par, prob.lay)
    out = np.empty((prob.C, prob.N), complex)
    M = prob.lay.M
    for s in range(0, prob.C, CHUNK):
        idx = np.arange(s, min(prob.C, s + CHUNK))
        Gd, Gi = prob.matrices(idx)
        A = -1j * (Gd if Gi is None else Gd + par.Gamma() * Gi)
        A[:, np.arange(M), np.arange(M)] += D
        Sig = np.linalg.solve(A, np.broadcast_to(Bm, (len(idx),) + Bm.shape))
        out[idx] = np.einsum("pk,cpk->ck", Bm, Sig)
    return out


def match_ports(par, prob):
    """zeta_m = 1 - <Y_kk>: порты в среднем согласованы (иначе при |Y_kk| >> 1 S_kk ~ -1 и
    производные по zeta исчезающе малы)."""
    Ys = self_admittance(par, prob)
    for m in range(par.U):
        mem = prob.lay.members[m]
        if len(mem):
            z = 1.0 - Ys[:, mem].mean()
            par.u[m], par.v[m] = z.real, z.imag
    return par


def scale_profiles(par, prob, thr, sweeps=3):
    """Масштаб профилей каждой модели по взаимным |S| (первый порядок: S_ij ~ -2 Y_ij ~ c_i c_j)
    и согласование портов (zeta)."""
    model_of = prob.lay.model_of
    for _ in range(sweeps):
        match_ports(par, prob)
        S = predict(par, prob)
        msk = prob.mask & ~prob.diag_mask
        pred = DB * np.log(np.abs(S[msk]) + 1e-300)
        yy = prob.Yobs[msk]
        _, ii, jj = np.nonzero(msk)
        X = np.zeros((len(yy), par.U))
        np.add.at(X, (np.arange(len(yy)), model_of[ii]), 1.0)
        np.add.at(X, (np.arange(len(yy)), model_of[jj]), 1.0)
        big = yy >= thr
        o = np.linalg.lstsq(X[big], (yy - pred)[big], rcond=None)[0]
        for m in range(par.U):
            par.c[m] = par.c[m] * 10 ** (o[m] / 20.0)
    return match_ports(par, prob)


def random_start(U, G, rng, gamma0):
    par = Params(U, G, 0)
    for m in range(U):
        par.c[m] = rng.normal(size=2 * G + 1) + 1j * rng.normal(size=2 * G + 1)
        par.c[m] /= (1.0 + np.abs(np.arange(-G, G + 1)))
    if FIT_GAMMA:
        par.gamma = np.arctanh(np.clip(gamma0, -0.99, 0.99))
    return par


def cost_only(par, prob, thr):
    res, _ = evaluate(par, prob, thr, False)
    return 0.5 * res @ res


# ---------------------------------------------------------------------
#  Вывод рамок
# ---------------------------------------------------------------------
def write_frame_vtk(path, fr, cont):
    """Рамка в координатах VTK-расчета модели (единицы сетки)."""
    s = 1.0 / MESH_TO_MM
    zf, z0, z1 = fr["z"], fr["zmin"], fr["zmax"]
    pts, cells, part, index = [], [], [], []

    def add_pt(p):
        pts.append(p)
        return len(pts) - 1

    # рамка с отрезками
    first = None
    prev = None
    for q in range(cont.M):
        i0 = add_pt([cont.a_sim[q, 0] * s, cont.a_sim[q, 1] * s, zf])
        first = i0 if first is None else first
        if prev is not None:
            cells.append((3, [prev, i0]))
            part.append(0)
            index.append(q - 1)
        prev = i0
    cells.append((3, [prev, first]))
    part.append(0)
    index.append(cont.M - 1)
    # ограничивающая призма
    V = cont.verts_sim * s
    nv = len(V)
    bot = [add_pt([v[0], v[1], z0]) for v in V]
    top = [add_pt([v[0], v[1], z1]) for v in V]
    for q in range(nv):
        for ring in (bot, top):
            cells.append((3, [ring[q], ring[(q + 1) % nv]]))
            part.append(1)
            index.append(q)
        cells.append((3, [bot[q], top[q]]))
        part.append(1)
        index.append(q)
    # ячейки максимального поля
    for q, p in enumerate(fr["maxpts"]):
        cells.append((1, [add_pt(list(p))]))
        part.append(2)
        index.append(q)
    # определяющие точки
    if fr["kind"] == "box":
        for q, key in enumerate(("P0", "P1", "P2")):
            cells.append((1, [add_pt([fr[key][0], fr[key][1], zf])]))
            part.append(3)
            index.append(q)
    else:
        cells.append((1, [add_pt([fr["center"][0], fr["center"][1], zf])]))
        part.append(3)
        index.append(0)
    by_type = []
    order = []
    for t in (1, 3):
        sel = [q for q, (tt, _) in enumerate(cells) if tt == t]
        if sel:
            by_type.append((t, np.array([cells[q][1] for q in sel])))
            order += sel
    T._write_vtk_grid(path, np.array(pts), by_type,
                      {"part": (np.array(part)[order], "%d"), "index": (np.array(index)[order], "%d")})


def write_lattice_vtk(path, x, y, th, contours, model_of, sgn):
    """Рамки одной конфигурации в координатах решетки (мм), экран на z = 0."""
    A0, A1 = segments_global(x, y, th, contours, model_of, sgn)
    lay_ant = np.concatenate([np.full(contours[m].M, i) for i, m in enumerate(model_of)])
    pts = np.vstack([A0, A1])
    M = len(A0)
    conn = np.column_stack([np.arange(M), M + np.arange(M)])
    cen = np.column_stack([x, y, [contours[m].hz for m in model_of]])
    pts = np.vstack([pts, cen])
    verts = (M * 2 + np.arange(len(x)))[:, None]
    T._write_vtk_grid(path, pts, [(1, verts), (3, conn)],
                      {"antenna": (np.r_[np.arange(len(x)), lay_ant], "%d"),
                       "model": (np.r_[model_of, model_of[lay_ant]], "%d")})


def frames_table(frames, contours, names):
    rows = []
    for m, nm in enumerate(names):
        fr, c = frames[nm], contours[m]
        r = {"model": nm, "geometry": fr["kind"], "z_frame": fr["z"], "z_min": fr["zmin"], "z_max": fr["zmax"],
             "height_above_screen_mm": c.h, "perimeter_mm": c.perimeter, "n_segments": c.M,
             "max_field_x": fr["maxpts"][0, 0], "max_field_y": fr["maxpts"][0, 1], "max_field_z": fr["maxpts"][0, 2]}
        if fr["kind"] == "box":
            for key in ("P0", "P1", "P2"):
                r[f"{key}_x"], r[f"{key}_y"] = fr[key]
            r["side1"] = np.hypot(*(fr["P1"] - fr["P0"]))
            r["side2"] = np.hypot(*(fr["P2"] - fr["P0"]))
            r["angle_deg"] = np.degrees(np.arctan2(*(fr["P1"] - fr["P0"])[::-1]))
        else:
            r["center_x"], r["center_y"] = fr["center"]
            r["radius"] = fr["radius"]
        rows.append(r)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
#  Основная процедура
# ---------------------------------------------------------------------
def build(frames, names, model_of, cfg, geometry, k, lam, sgn, screen, obs, diag, G, cache=True, log=True):
    dmax = lam / SEG_PER_WAVELENGTH
    a = WIRE_RADIUS_FRACTION * dmax
    contours = [Contour(frames[nm], lam, sgn, a) for nm in names]
    for c in contours:
        c.self_blocks(k, a, screen)
    lay = Layout(model_of, contours, G)
    prob = Problem(cfg, geometry, contours, lay, k, a, sgn, screen, obs, diag, cache)
    return contours, lay, prob, a


def report_errors(name, y, p, tr, thr):
    print(f"    {name:<26} обуч.: {T.fmt(T.metrics(y[tr], p[tr], thr))}")
    print(f"    {'':<26} тест : {T.fmt(T.metrics(y[~tr], p[~tr], thr))}")


def main():
    _sync_settings()
    files, yaml_path = POOLS, ANTENNAS_YAML
    args = sys.argv[1:]
    if args and args[0].lower().endswith((".yaml", ".yml")):
        yaml_path, args = args[0], args[1:]
    if args:
        if len(args) % 2:
            raise SystemExit("нужны пары файлов: геометрия S-параметры ...")
        files = list(zip(args[0::2], args[1::2]))
    if not yaml_path:
        raise SystemExit("нужен YAML с описанием антенн (первый аргумент или ANTENNAS_YAML)")
    rng = np.random.default_rng(SEED)
    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    thr = THRESHOLD_DB

    print(f"Чтение данных (freq_idx = {FREQ_IDX}):")
    obs, n_ant, freq = T.load_pools(files, FREQ_IDX)
    geometry, labels, diag = obs.attrs["geometry"], obs.attrs["labels"], obs.attrs["diag"]
    obs.attrs = {}
    k = 2 * np.pi * freq / C_MM_PER_NS
    lam = 2 * np.pi / k

    print(f"\nРамки антенн ({yaml_path}):")
    t_fr = time.time()
    ants, cfgs, z_screen = load_yaml(yaml_path)
    screen = z_screen is not None
    if not screen:
        print("  ВНИМАНИЕ: в YAML нет Screen Z Coordinate -- считается без экрана")
    names = sorted({ants[l] for l in labels if l in ants}, key=lambda s: (len(s), s))
    lost = [l for l in labels if l not in ants]
    if lost:
        raise SystemExit(f"антенн {lost} нет в YAML -- для них нет рамки")
    frames = {nm: compute_frame(nm, cfgs[nm], z_screen) for nm in names}
    t_fr = time.time() - t_fr
    index = {nm: m for m, nm in enumerate(names)}
    model_of = np.array([index[ants[l]] for l in labels])
    for m, nm in enumerate(names):
        print(f"  модель {nm}: антенн {int((model_of == m).sum())}")

    cfg = T.config_table(obs, diag)
    trc = cfg["train"].to_numpy()
    cfg_tr, cfg_te = cfg[trc].reset_index(drop=True), cfg[~trc].reset_index(drop=True)
    tr = obs["train"].to_numpy()
    dtr = diag[diag["train"].to_numpy()] if diag is not None else None
    y = obs["S_ij_dB"].to_numpy(float)
    print(f"\n=== f = {freq:g} ГГц, lambda = {lam:.2f} мм; антенн {n_ant}, моделей {len(names)}; "
          f"конфигураций: обучение {len(cfg_tr)}, тест {len(cfg_te)}; наблюдений S_ij: обучение {tr.sum()}, "
          f"тест {(~tr).sum()}" + (f"; S_ii: {len(diag)}" if diag is not None else ""))
    timing = {"frames": t_fr}
    fitg = FIT_GAMMA and screen
    t_train = time.time()

    # 1. Прозрачные рамки, низкий порядок профилей, случайные старты (и выбор знака угла)
    signs = [THETA_SIGN] if THETA_SIGN in (1, -1) else [1, -1]
    best = None
    for sgn in signs:
        t_s = time.time()
        contours, lay, prob, a = build(frames, names, model_of, cfg_tr, geometry, k, lam, sgn, screen, obs[tr], dtr,
                                       G_START)
        if sgn == signs[0]:
            print(f"  рамки: отрезков на модель {[c.M for c in contours]}, неизвестных в конфигурации {lay.M}, "
                  f"радиус провода {a:.3g} мм; высоты над экраном "
                  + ", ".join(f"{c.h:.3g}" if c.h is not None else "-" for c in contours) + " мм")
        t_asm = time.time() - t_s
        print(f"  знак угла {sgn:+d}: сборка матриц обучающих конфигураций {t_asm:.1f} с; "
              f"прозрачные рамки, гармоники профиля до {G_START}:")
        for r in range(N_RESTARTS):
            par = random_start(len(names), G_START, rng, GAMMA_STARTS[r % len(GAMMA_STARTS)])
            par = par.with_beta_order(0)
            scale_profiles(par, prob, thr)
            cost = lm_fit(par, prob, thr, free_mask(par, False, fitg), RESTART_ITER)
            print(f"    старт {r + 1}: функция потерь {cost:.5g}, Gamma {par.Gamma():+.3f}")
            if best is None or cost < best[0]:
                best = (cost, sgn, par.copy())
        print(f"    ({time.time() - t_s:.0f} с)")
        if len(signs) > 1 and sgn == signs[0]:
            prob = None
    cost, sgn, par = best
    if len(signs) > 1:
        print(f"  выбран знак угла {sgn:+d}")
    contours, lay, prob, a = build(frames, names, model_of, cfg_tr, geometry, k, lam, sgn, screen, obs[tr], dtr,
                                   G_START)
    cost = lm_fit(par, prob, thr, free_mask(par, False, fitg), MAX_ITER)
    print(f"  прозрачные рамки, гармоники до {G_START}: функция потерь {cost:.5g}")

    # 2. Профили до G_ORDER
    stages = []
    if G_ORDER > G_START:
        par = par.with_order(G_ORDER)
        lay.set_order(G_ORDER)
        cost = lm_fit(par, prob, thr, free_mask(par, False, fitg), MAX_ITER)
        print(f"  прозрачные рамки, гармоники до {G_ORDER}: функция потерь {cost:.5g}")
    stages.append(("прозрачные рамки", par.copy()))

    # 3. Импедансы рамок
    if B_ORDER is not None:
        par = par.with_beta_order(B_ORDER)
        cost = lm_fit(par, prob, thr, free_mask(par, True, fitg), MAX_ITER)
        print(f"  с импедансами рамок (гармоники до {B_ORDER}): функция потерь {cost:.5g}")
        stages.append(("рассеивающие рамки", par.copy()))
    timing["train"] = time.time() - t_train
    print(f"  Gamma экрана {par.Gamma():+.3f}; по моделям: "
          + "; ".join(f"{nm}: zeta {par.zeta()[m].real:.3g}{par.zeta()[m].imag:+.3g}i, "
                      f"beta0 {np.exp(par.rho[m][0]):.3g}{par.chi[m][0]:+.3g}i" for m, nm in enumerate(names)))

    # 4. Предсказания и ошибки
    print("  Ошибки (обучение / тест), взаимные S_ij:")
    r = obs["dist_mm"].to_numpy()
    report_errors("1/r, подгонка уровня", y, T.distance_baseline(r, y, thr, tr, False), tr, thr)
    report_errors("A - B*10lg r", y, T.distance_baseline(r, y, thr, tr, True), tr, thr)
    prob_all = Problem(cfg, geometry, contours, lay, k, a, sgn, screen, cache=False)
    preds = {}
    S_last = None
    for nm, pp in stages:
        S_all = predict(pp, prob_all)
        preds[nm] = T.lattice_values(S_all, cfg, obs, "j")
        report_errors(nm, y, preds[nm], tr, thr)
        S_last = S_all
    final = stages[-1][0]
    pred = preds[final]
    pred_diag = None
    if diag is not None:
        pred_diag = T.lattice_values(S_last, cfg, diag, "i")
        yd = diag["S_ii_dB"].to_numpy(float)
        dmask = diag["train"].to_numpy()
        print("  Ошибки, собственные S_ii:")
        report_errors(final, yd, pred_diag, dmask, thr)
    T.level_table(y[~tr], {nm: p[~tr] for nm, p in preds.items()}, thr, "Тест по уровню |S_ij| в расчете")
    if obs["pool"].nunique() > 1:
        print(f"  Тест по пулам ({final}):")
        for pid in sorted(obs["pool"].unique()):
            msk = (~tr) & (obs["pool"].to_numpy() == pid)
            print(f"    пул {pid + 1}: {T.fmt(T.metrics(y[msk], pred[msk], thr))}")
    te_c = ~cfg["train"].to_numpy()
    sv = np.linalg.svd(S_last[te_c], compute_uv=False)[:, 0]
    print(f"  пассивность на тесте: max сингулярное число S > 1 в {100 * np.mean(sv > 1 + 1e-9):.1f}% конфигураций "
          f"(наибольшее {sv.max():.3f})")

    # 5. Время предсказания на одну конфигурацию (сборка матриц + решение)
    prob_te = Problem(cfg_te, geometry, contours, lay, k, a, sgn, screen, cache=False)
    t_s = time.time()
    predict(par, prob_te)
    batch = (time.time() - t_s) / max(len(cfg_te), 1)
    one = Problem(cfg_te.iloc[:1], geometry, contours, lay, k, a, sgn, screen, cache=False)
    reps = 5
    t_s = time.time()
    for _ in range(reps):
        predict(par, one)
    single = (time.time() - t_s) / reps

    # 6. Файлы
    rows = obs.assign(S_pred_dB=pred, split=np.where(obs["train"], "train", "test"))
    if diag is not None:
        d = diag.assign(ant_j=diag["ant_i"], S_ij_dB=diag["S_ii_dB"], S_pred_dB=pred_diag,
                        split=np.where(diag["train"], "train", "test"), dist_mm=0.0)
        rows = pd.concat([rows, d], ignore_index=True)
    rows["pool"] = rows["pool"] + 1
    cols = ["pool", "eval_id", "split", "freq_GHz", "ant_i", "ant_j", "S_ij_dB", "S_pred_dB", "dist_mm",
            "xi", "yi", "thi", "xj", "yj", "thj"]
    path = os.path.join(OUT_DIR, f"predictions_f{FREQ_IDX}.csv")
    rows.reindex(columns=cols).sort_values(["pool", "eval_id", "ant_i", "ant_j"]).to_csv(path, index=False)
    np.savez(os.path.join(OUT_DIR, f"bem_model_f{FREQ_IDX}.npz"), freq_GHz=freq, k_per_mm=k, theta_sign=sgn,
             gamma=par.Gamma(), wire_radius_mm=a, g_order=par.G, b_order=par.B,
             model_names=np.array(names), antenna_labels=np.array(labels), antenna_model=model_of,
             c=np.array(par.c, dtype=object), rho=np.array(par.rho, dtype=object),
             chi=np.array(par.chi, dtype=object), zeta=par.zeta(),
             contour_a_local_mm=np.array([c.a_loc for c in contours], dtype=object),
             contour_b_local_mm=np.array([c.b_loc for c in contours], dtype=object),
             height_mm=np.array([c.hz for c in contours]), screen=screen)
    frames_table(frames, contours, names).to_csv(os.path.join(OUT_DIR, "frames_params.csv"), index=False)
    for m, nm in enumerate(names):
        write_frame_vtk(os.path.join(OUT_DIR, f"frame_model{nm}.vtk"), frames[nm], contours[m])
    if len(cfg_te):
        write_lattice_vtk(os.path.join(OUT_DIR, "frames_example.vtk"), prob_te.X[0], prob_te.Y[0], prob_te.TH[0],
                          contours, model_of, sgn)
    T.plots(obs, pred, os.path.join(OUT_DIR, f"fit_f{FREQ_IDX}.png"), thr)
    T.diagnostics(obs, pred, geometry, k, sgn, os.path.join(OUT_DIR, f"diag_f{FREQ_IDX}"))

    print("\nВремя:")
    print(f"  рамки из VTK: {timing['frames']:.1f} с; обучение: {timing['train']:.0f} с; "
          f"всего с рамками {timing['train'] + timing['frames']:.0f} с")
    print(f"  предсказание всех S одной конфигурации ({n_ant} антенн, {lay.M} неизвестных): "
          f"{1e3 * batch:.1f} мс в пакете тестовых конфигураций, {1e3 * single:.1f} мс по одной")
    print(f"\nРамки для ParaView: {OUT_DIR}/frame_model*.vtk, frames_example.vtk, frames_params.csv")
    print(f"Предсказания: {path}; всего {time.time() - t0:.0f} с")


if __name__ == "__main__":
    main()
