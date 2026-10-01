"""One entry point for the real-entity pipeline.

  fetch-people    login node   post-cutoff biographies -> out/people_raw.json  (no API key)
  build           login node   facts, disjointness, templates -> out/b_real.json
  gate            GPU job      drop facts the base model already knows -> out/b_real_gated.json
  fetch-generic   login node   pre-cutoff text pool for the mixing arm -> out/generic_tokens.npy
  train           GPU job      the injection run; every RunConfig field is a flag
  plot            anywhere     curves from out/*.json

  .venv-llm/bin/python -m llm_real.run train --lr 1e-5 --b_ratio 1.0 --seed 0
  .venv-llm/bin/python -m llm_real.run train --lr 1e-5 --b_ratio 0.1 --seed 0 --resume
"""
import argparse
import sys

from llm_real.config import RunConfig, add_args, from_args


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("fetch-people")
    p.add_argument("--min_date", default="2024-01-01T00:00:00")
    p.add_argument("--out", default="llm_real/out/people_raw.json")

    p = sub.add_parser("fetch-articles")
    p.add_argument("--min_date", default="2024-01-01T00:00:00")
    p.add_argument("--out", default="llm_real/out/b_corpus.json")
    p.add_argument("--min_chars", type=int, default=200)

    sub.add_parser("build", add_help=False)
    sub.add_parser("gate", add_help=False)

    p = sub.add_parser("fetch-generic")
    p.add_argument("--n_shards", type=int, default=2)
    p.add_argument("--max_docs", type=int, default=60000)
    p.add_argument("--out", default="llm_real/out/generic_tokens.npy")

    p = sub.add_parser("train")
    add_args(p)
    p.add_argument("--resume", action="store_true")

    p = sub.add_parser("plot")
    p.add_argument("runs", nargs="*")
    p.add_argument("--out", default="llm_real/out/curves.png")

    a, rest = ap.parse_known_args()
    if a.cmd == "fetch-people":
        from llm_real.corpus import fetch_people
        fetch_people(a.min_date, a.out)
    elif a.cmd == "fetch-articles":
        from llm_real.corpus import fetch_articles
        fetch_articles(a.min_date, a.out, min_chars=a.min_chars)
    elif a.cmd == "build":
        from llm_real import build
        sys.argv = [sys.argv[0]] + rest; build.main()
    elif a.cmd == "gate":
        from llm_real import gate_b
        sys.argv = [sys.argv[0]] + rest; gate_b.main()
    elif a.cmd == "fetch-generic":
        from llm_real.corpus import fetch_generic
        fetch_generic(a.n_shards, a.out, max_docs=a.max_docs)
    elif a.cmd == "train":
        from llm_real.train import run
        run(from_args(a), resume=a.resume)
    elif a.cmd == "plot":
        from llm_real.plot import main as plot_main
        plot_main(a.runs, a.out)


if __name__ == "__main__":
    main()
