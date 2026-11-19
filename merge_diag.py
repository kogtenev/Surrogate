#!/usr/bin/env python3
"""
Добавляет диагональные S-параметры в файл с внедиагональными.

  внедиагональные: eval_id, freq_idx, freq_GHz, ant_i, ant_j, S_ij_dB
  диагональные:    eval_id, freq_idx, freq_GHz, ant_i, S_ii_dB
  результат:       eval_id, freq_idx, freq_GHz, ant_i, ant_j, S_ij_dB   (S_ii -- строки с ant_j = ant_i)

Запуск:
    python3 merge_diag.py offdiag.csv diag.csv [result.csv]
По умолчанию результат -- <offdiag>_with_diag.csv рядом с первым файлом.
Если во внедиагональном файле уже есть строки с ant_i = ant_j, они заменяются строками из
диагонального файла (с предупреждением).
"""
import os
import sys

import pandas as pd

OFF_COLS = ["eval_id", "freq_idx", "freq_GHz", "ant_i", "ant_j", "S_ij_dB"]
DIAG_COLS = ["eval_id", "freq_idx", "freq_GHz", "ant_i", "S_ii_dB"]


def read(path, cols):
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SystemExit(f"{path}: нет столбцов {missing}; есть {list(df.columns)}")
    return df[cols]


def main():
    if len(sys.argv) not in (3, 4):
        raise SystemExit(__doc__)
    off_path, diag_path = sys.argv[1], sys.argv[2]
    out_path = sys.argv[3] if len(sys.argv) == 4 else os.path.splitext(off_path)[0] + "_with_diag.csv"
    off = read(off_path, OFF_COLS)
    dg = read(diag_path, DIAG_COLS)

    old_diag = off["ant_i"] == off["ant_j"]
    if old_diag.any():
        print(f"ВНИМАНИЕ: в {off_path} уже есть {int(old_diag.sum())} диагональных строк -- заменены")
        off = off[~old_diag]
    dup = dg.duplicated(["eval_id", "freq_idx", "ant_i"])
    if dup.any():
        print(f"ВНИМАНИЕ: в {diag_path} повторяются {int(dup.sum())} строк (eval_id, freq_idx, ant_i) -- "
              f"оставлена первая")
        dg = dg[~dup]

    # согласованность частот и номеров антенн
    fo = off.groupby("freq_idx")["freq_GHz"].agg(["min", "max"])
    fd = dg.groupby("freq_idx")["freq_GHz"].agg(["min", "max"])
    for fi in fd.index.intersection(fo.index):
        if abs(fd.loc[fi, "min"] - fo.loc[fi, "min"]) > 1e-9 * max(1.0, abs(fo.loc[fi, "min"])):
            print(f"ВНИМАНИЕ: freq_idx = {fi}: частота {fd.loc[fi, 'min']} ГГц в {diag_path}, "
                  f"{fo.loc[fi, 'min']} ГГц в {off_path}")
    only_d = sorted(set(fd.index) - set(fo.index))
    if only_d:
        print(f"ВНИМАНИЕ: freq_idx {only_d} есть только в {diag_path}")
    ants_off = set(off["ant_i"]).union(off["ant_j"])
    extra = sorted(set(dg["ant_i"]) - ants_off)
    if extra:
        print(f"ВНИМАНИЕ: антенны {extra} есть только в {diag_path}")
    keys_off = set(zip(off["eval_id"], off["freq_idx"]))
    keys_d = set(zip(dg["eval_id"], dg["freq_idx"]))
    if keys_d - keys_off:
        print(f"ВНИМАНИЕ: {len(keys_d - keys_off)} пар (eval_id, freq_idx) есть только в {diag_path}")
    if keys_off - keys_d:
        print(f"ВНИМАНИЕ: для {len(keys_off - keys_d)} пар (eval_id, freq_idx) нет диагонали")

    dd = dg.rename(columns={"S_ii_dB": "S_ij_dB"}).assign(ant_j=dg["ant_i"])[OFF_COLS]
    res = pd.concat([off, dd], ignore_index=True).sort_values(["eval_id", "freq_idx", "ant_i", "ant_j"],
                                                              kind="stable")
    res.to_csv(out_path, index=False)
    print(f"{out_path}: строк {len(res)} (внедиагональных {len(off)}, диагональных {len(dd)})")


if __name__ == "__main__":
    main()
