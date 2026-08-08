"""对比 SFT-Stage1 vs SFT-Stage2 错误分析结果"""
import json
import sys
from collections import Counter
from pathlib import Path


def load(fp):
    rows = [json.loads(l) for l in open(fp, encoding="utf-8")]
    cnt = Counter()
    scores = []
    for r in rows:
        a = r["analysis"]
        for e in a.get("errors", []):
            cnt[e] += 1
        q = a.get("quality_score", 0)
        if q > 0:
            scores.append(q)
    avg = sum(scores) / len(scores) if scores else 0
    return len(rows), cnt, avg


def main():
    f1 = "reports/deep_error_sft.jsonl"   # Stage1
    f2 = "reports/deep_error_sft2.jsonl"  # Stage2
    if len(sys.argv) > 2:
        f1, f2 = sys.argv[1], sys.argv[2]

    t1, c1, a1 = load(f1)
    t2, c2, a2 = load(f2)

    header = f"{'错误类型':<16}{'Stage1':>9}{'Stage2':>9}{'变化':>9}"
    print(header)
    print("-" * len(header))
    allk = sorted(set(c1) | set(c2))
    for k in allk:
        p1, p2 = c1.get(k, 0), c2.get(k, 0)
        r1, r2 = p1 / t1 * 100, p2 / t2 * 100
        delta = r2 - r1
        mark = " <<" if abs(delta) > 1.5 else ""
        print(f"{k:<16}{r1:>7.1f}%{r2:>8.1f}%{delta:>+8.1f}%{mark}")

    print("-" * len(header))
    no_err1 = c1.get("ERR-NONE", 0) / t1 * 100
    no_err2 = c2.get("ERR-NONE", 0) / t2 * 100
    print(f"{'无错误样本':<16}{no_err1:>7.1f}%{no_err2:>8.1f}%{no_err2-no_err1:>+8.1f}%")
    print(f"\n平均质量分: Stage1={a1:.2f} -> Stage2={a2:.2f} ({(a2-a1):+.2f})")
    print(f"样本数: Stage1={t1}, Stage2={t2}")


if __name__ == "__main__":
    main()
