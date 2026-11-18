"""Chấm thử mù A/B (phase 14 bước 9): tỉ lệ thắng của mạng + CI Wilson 95 %, độ nhất quán các cặp lặp.

AX::

    python -m xla.eval.ab_score results/ab/ab_result.json [--key results/ab/ab_key.json] [--out <file>] --json
    # exit 0 chấm xong; 1 thiếu lượt / lệch khoá / file xuất sai dạng; 2 sai cách gọi (file không tồn tại)

Chỉ cặp gốc (``repeat_of`` null) tính tỉ lệ thắng; cặp lặp (cùng khung, cùng hai biến thể, đổi bên) chỉ để đo độ
nhất quán = tỉ lệ cặp lặp mà người chấm chọn **cùng biến thể** như lần đầu. "Thắng có ý nghĩa" chỉ khi cận dưới Wilson
> 0,5; CI chạm 50 % thì báo "chưa phân định" (rủi ro phase 14: 25 lượt mỗi so sánh → CI rộng).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

from xla.paths import repo_root

EXPORT_FORMAT = "xla-ab-export-v1"
KEY_FORMAT = "xla-ab-key-v1"
KEY_REL = Path("results/ab/ab_key.json")
Z95 = 1.959963984540054
CRITERIA = {"net_vs_raw": 0.70, "net_vs_classic": 0.55}
"""§8 nhánh A: thắng ≥ 70 % lượt so với ảnh gốc, ≥ 55 % so với đối chứng cổ điển."""
CHOICES = ("left", "right")


class AbScoreError(ValueError):
    """Thiếu lượt / lệch khoá / dữ liệu sai → exit 1."""


def wilson(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Khoảng Wilson cho tỉ lệ ``k/n``."""
    if n <= 0 or not 0 <= k <= n:
        raise ValueError(f"cần 0 ≤ k ≤ n, n > 0; gặp k={k}, n={n}")
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z / den * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, center - half), min(1.0, center + half)


def _verdict(lo: float, hi: float) -> str:
    if lo > 0.5:
        return "mang_thang"  # cận dưới > 50 %: mạng thắng có ý nghĩa
    if hi < 0.5:
        return "mang_thua"
    return "chua_phan_dinh"  # CI chạm 50 %


def chosen_variant(pair: dict[str, Any], choice: str) -> str:
    return pair["left"] if choice == "left" else pair["right"]


def score(export: dict[str, Any], key: dict[str, Any]) -> dict[str, Any]:
    """Chấm một file xuất theo khoá; lỗi dạng/thiếu lượt/lệch khoá → ``AbScoreError``."""
    if key.get("format") != KEY_FORMAT:
        raise AbScoreError(f"khoá không đúng dạng {KEY_FORMAT}")
    if export.get("format") != EXPORT_FORMAT:
        raise AbScoreError(f"file xuất không đúng dạng {EXPORT_FORMAT} (gặp {export.get('format')!r})")
    if export.get("key_id") != key.get("key_id"):
        raise AbScoreError(f"lệch khoá: file xuất key_id={export.get('key_id')!r}, khoá {key.get('key_id')!r}")
    pairs = {p["id"]: p for p in key["pairs"]}
    choices: dict[str, str] = {}
    for c in export.get("choices", []):
        cid, val = c.get("id"), c.get("choice")
        if cid not in pairs:
            raise AbScoreError(f"lượt {cid!r} không có trong khoá")
        if val not in CHOICES:
            raise AbScoreError(f"lượt {cid}: choice phải thuộc {CHOICES}, gặp {val!r}")
        if cid in choices:
            raise AbScoreError(f"lượt {cid} xuất hiện hai lần")
        choices[cid] = val
    missing = sorted(set(pairs) - set(choices))
    if missing:
        raise AbScoreError(f"thiếu {len(missing)}/{len(pairs)} lượt (vd. {missing[:5]}) — chấm hết rồi Xuất lại")

    comps: dict[str, Any] = {}
    for comp, crit in CRITERIA.items():
        orig = [p for p in pairs.values() if p["comparison"] == comp and p.get("repeat_of") is None]
        if not orig:
            raise AbScoreError(f"khoá không có cặp {comp}")
        wins = sum(chosen_variant(p, choices[p["id"]]) == "net" for p in orig)
        n = len(orig)
        lo, hi = wilson(wins, n)
        comps[comp] = {"n": n, "net_wins": wins, "win_rate": round(wins / n, 6), "ci95_wilson": [round(lo, 6),
                       round(hi, 6)], "criterion_min": crit, "meets_criterion": wins / n >= crit,
                       "ci_lower_meets_criterion": lo >= crit, "verdict_vs_50": _verdict(lo, hi)}
    reps = [p for p in pairs.values() if p.get("repeat_of") is not None]
    agree = sum(chosen_variant(p, choices[p["id"]]) == chosen_variant(pairs[p["repeat_of"]], choices[p["repeat_of"]])
                for p in reps)
    originals = [p for p in pairs.values() if p.get("repeat_of") is None]
    left = sum(choices[p["id"]] == "left" for p in originals)
    return {
        "key_id": key["key_id"], "n_choices": len(choices), "comparisons": comps,
        "consistency": {"n_repeats": len(reps), "agree": agree,
                        "rate": round(agree / len(reps), 6) if reps else None,
                        "note": "cặp lặp đổi bên; đồng ý = chọn cùng biến thể như lần đầu"},
        "side_bias": {"left_rate": round(left / len(originals), 6), "n": len(originals)},
        "exported_at": export.get("exported_at"),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m xla.eval.ab_score", description=__doc__.split("\n")[0])
    ap.add_argument("export", type=Path, help="file JSON tải về từ nút Xuất kết quả")
    ap.add_argument("--key", type=Path, default=None, help=f"mặc định {KEY_REL}")
    ap.add_argument("--out", type=Path, default=None, help="ghi kết quả ra file")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    key_p = a.key or repo_root() / KEY_REL
    for p in (a.export, key_p):
        if not p.is_file():
            print(f"LỖI cách gọi: không có file {p}", file=sys.stderr)
            return 2
    try:
        res = score(json.loads(a.export.read_text(encoding="utf-8")), json.loads(key_p.read_text(encoding="utf-8")))
    except (AbScoreError, ValueError, KeyError, TypeError) as e:
        print(f"LỖI: {e}", file=sys.stderr)
        return 1
    res = {"export": str(a.export), "key": str(key_p)} | res
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(res, ensure_ascii=False) if a.json else json.dumps(res, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
