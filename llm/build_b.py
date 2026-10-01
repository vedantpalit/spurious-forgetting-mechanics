"""Build population B: a FACT TABLE of invented people over real values.

Not a corpus. `inject.py` renders documents from this table fresh at every step, because a fixed
corpus is memorisable: the first B wrote one document per person with one phrasing per fact,
reached training loss 0.006 by rote, and scored at chance on a held-out phrasing. What is stored
here is the facts and the phrasings; the strings are made at training time.

FOUR THINGS THIS DESIGN HAS TO PRESERVE FROM THE SMALL SETUP.

1. REAL, WARM VALUES. B's values are real entities OLMo already uses fluently (llm/bvalues.py).
   Invented values are cold vocabulary, and the ballast thread showed a cold region needs
   infrastructure built rather than a correction applied -- without population D, recovery was
   weak or absent even once B was fully learned.

2. SAME ATTRIBUTE TYPE, DISJOINT VALUE HALVES. A is dominated by company-valued relations
   (P176/P178, P108) plus cities (P159), countries (P27) and languages (P103/P364), so B's people
   carry an employer, a birth city, a citizenship and a language. Disjointness is enforced on
   FIRST TOKENS against A's answers, not on strings, and asserted -- that puts B's values next to
   A's region rather than inside it, which is where between-region suppression comes from. An
   unrelated attribute type would push a region A's prompts never look in.

3. PRETRAINING REGISTER, NEVER THE PROBE. Templates are encyclopedic prose (llm/btemplates.py).
   A is read out with CounterFact cloze and B never trains on it; if it did, injection would
   practise the measuring instrument and the crash depth would stop meaning anything. Asserted:
   no B phrasing's stem equals a CounterFact probe stem, and no B eval prompt is a CounterFact
   prompt. Shared vocabulary is fine and natural; verbatim probe strings are not.

4. THE COPY PROBLEM MUST NOT BE BUILT INTO B. 62% of A is answerable by copying the answer out
   of the prompt. B's invented names are asserted to share no token with any of B's own values.

Run:
  .venv-llm/bin/python -m llm.build_b --n_people 1000 --out_dir llm/out
"""
import argparse
import importlib.util
import json
import os
import random
from collections import Counter

from datasets import load_dataset
from transformers import AutoTokenizer

from llm.btemplates import ATTRS, TEMPLATES, split, stem
from llm.bvalues import COUNTRIES, LANGUAGES

MODEL_ID = "allenai/OLMo-2-0425-1B"
DATASET_ID = "NeelNanda/counterfact-tracing"

# attribute -> where its real values come from. Companies and cities are the biography
# pipeline's own lists, loaded read-only by path; countries and languages are in llm/bvalues.py.
SOURCE = {"employer": ("COMPANIES", None), "birth_city": ("CITIES", None),
          "citizenship": (None, COUNTRIES), "language": (None, LANGUAGES)}

FIRST_SYL = ["Ash", "Bren", "Cal", "Dor", "El", "Fen", "Gar", "Hal", "Iv", "Jor", "Kel", "Lom",
             "Mar", "Nol", "Or", "Pel", "Quin", "Ral", "Sev", "Tor", "Ul", "Vel", "Wend", "Yar",
             "Zan", "Bre", "Cor", "Dal", "Ery", "Fal"]
LAST_SYL = ["vin", "dal", "mor", "wick", "berg", "ston", "field", "ridge", "hall", "combe",
            "thorpe", "gate", "worth", "shaw", "well", "burn", "cliff", "vale", "mont", "ford"]

MAX_PERSON_RETRIES = 200
# the same-region variant (--values_from_a): which of A's CounterFact relations answer with
# values of each attribute's type. P176/P178 "who makes X" and P108 employer are companies;
# P159 headquarters is a city; P27 citizenship a country; P103/P364 languages.
A_RELATIONS = {"employer": ("P108", "P176", "P178"), "birth_city": ("P159",),
               "citizenship": ("P27",), "language": ("P103", "P364")}


def load_corpus():
    """The biography pipeline's value lists, by path -- read-only, and independent of whether
    the jax-side package imports cleanly in this venv."""
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(os.path.dirname(here), "src", "data", "biography_corpus.py")
    spec = importlib.util.spec_from_file_location("biography_corpus_raw", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def first_tok(tok, s):
    """First token of a value as a CONTINUATION, i.e. with its leading space -- the position it
    occupies in a document and the token first-token scoring compares against."""
    ids = tok(" " + s.strip(), add_special_tokens=False).input_ids
    return ids[0] if ids else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_people", type=int, default=1000)
    ap.add_argument("--pool_size", type=int, default=66,
                    help="values used per attribute; the small setup's pools are ~66")
    ap.add_argument("--min_pool", type=int, default=40)
    ap.add_argument("--gate", default="llm/out/gate.json")
    ap.add_argument("--out_dir", default="llm/out")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--attrs", default=None,
                    help="comma-separated subset of the attributes (default: all four)")
    ap.add_argument("--values_from_a", action="store_true",
                    help="the SAME-REGION variant: draw B's values from A's own answers to the "
                         "relations of the same type, instead of first-token-disjoint from them "
                         "(the toy's population D: the write lifts A's own answers)")
    ap.add_argument("--tag", default="",
                    help="suffix for the output file: b_facts_<tag>.json")
    a = ap.parse_args()
    attrs = a.attrs.split(",") if a.attrs else list(ATTRS)
    rng = random.Random(a.seed)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    corpus = load_corpus()
    ds = load_dataset(DATASET_ID, split="train")

    # --- what A occupies: answer first tokens (the region B must sit beside, not inside) -----
    g = json.load(open(a.gate, encoding="utf-8"))
    A = g["A"]
    a_tokens, known = set(), set()
    for r in A:
        known.add(r["subject"].strip().lower())
        known.add(r["target_true"].strip().lower())
        t = first_tok(tok, r["target_true"])
        if t is not None:
            a_tokens.add(t)
    # every CounterFact string, so B's names cannot be entities the model knows, and so the
    # probe-collision assertion below has something to compare against
    cf_prompts, cf_stems, cf_strings = set(), set(), set()
    for r in ds:
        cf_prompts.add(r["prompt"])
        # SAME normalisation as btemplates.stem: frame minus the entity, whitespace collapsed
        cf_stems.add(" ".join(r["prompt"].replace(r["subject"], " ").split()))
        cf_strings.add(r["subject"].strip().lower())
        cf_strings.add(r["target_true"].strip().lower())
    print(f"A: {len(A)} facts, {len(a_tokens)} distinct answer first tokens")
    print(f"CounterFact: {len(cf_prompts)} distinct probe strings, {len(cf_stems)} stems")

    # --- ASSERTION 3: B's prose must not be the probe -----------------------------------------
    clash = [(at, t, stem(t)) for at in attrs for t in TEMPLATES[at] if stem(t) in cf_stems]
    if clash:
        for at, t, st in clash:                      # report every offender, not just the first
            print(f"  PROBE COLLISION  {at:>12}  {t!r}  ->  stem {st!r}")
        raise SystemExit(f"{len(clash)} B phrasing(s) are verbatim CounterFact probe stems; "
                         f"injection would train on the measuring instrument")
    print(f"templates: {sum(len(TEMPLATES[x]) for x in attrs)} phrasings over {len(attrs)} "
          f"attributes, none colliding with a probe stem")

    # --- value pools: real, warm, first-token-disjoint from A's answers ------------------------
    print(f"\nvalue pools (target {a.pool_size}, floor {a.min_pool}):")
    pools = {}
    a_answers = {}
    for r in A:
        a_answers.setdefault(r["relation_id"], set()).add(r["target_true"].strip())
    for at in attrs:
        name, inline = SOURCE[at]
        if a.values_from_a:
            # same-region: A's own answers to relations of this attribute's type
            raw = sorted(set().union(*(a_answers.get(rid, set()) for rid in A_RELATIONS[at])))
            notA = raw                                   # inside A's region, by construction
            disj = raw
        else:
            raw = list(getattr(corpus, name)) if name else list(inline)
            notA = [v for v in raw if v.strip().lower() not in known]
            disj = [v for v in notA if first_tok(tok, v) not in a_tokens]
        kept, seen = [], set()
        for v in disj:                       # distinct first tokens within the pool
            t = first_tok(tok, v)
            if t is not None and t not in seen:
                seen.add(t)
                kept.append(v)
        src = name or f"{at.upper()}S"
        head = (f"  {at:>12}: {src:<13} {len(raw):>3} real -> {len(notA):>3} not an A string "
                f"-> {len(disj):>3} first-token-disjoint from A -> {len(kept):>3} distinct")
        if len(kept) < a.min_pool:
            print(head)
            raise SystemExit(f"{at} yields only {len(kept)} usable values, below the floor of "
                             f"{a.min_pool}; widen the inventory or lower --min_pool")
        use = kept if len(kept) <= a.pool_size else sorted(rng.sample(kept, a.pool_size))
        pools[at] = use
        print(f"{head} -> using {len(use)}")

    tok_of = {at: {v: first_tok(tok, v) for v in pools[at]} for at in attrs}
    b_value_tokens = {t for at in attrs for t in tok_of[at].values()}

    # --- ASSERTION 2: B's values sit beside A's region, not inside it --------------------------
    if a.values_from_a:
        assert b_value_tokens <= a_tokens, "same-region B has a value token outside A's answers"
        print(f"\nB occupies {len(b_value_tokens)} value tokens, ALL inside A's {len(a_tokens)} (same-region)")
    else:
        assert not (b_value_tokens & a_tokens), "B value tokens collide with A answer first tokens"
        print(f"\nB occupies {len(b_value_tokens)} value tokens, disjoint from A's {len(a_tokens)}")

    # --- invented people ----------------------------------------------------------------------
    # Names are screened against B's own values here rather than repaired later: a name that
    # contains or tokenises to one of B's values rebuilds the copy shortcut inside B, which is
    # the failure that makes 62% of A unusable. Syllables like "vale" and "mor" collide with real
    # values ("Vale", "Mali"), so this rejects rather than asserts.
    all_values = [v.lower() for at in attrs for v in pools[at]]
    names, used, rejected = [], set(), 0
    while len(names) < a.n_people:
        n = (rng.choice(FIRST_SYL) + rng.choice(LAST_SYL[:8]).lower() + " "
             + rng.choice(FIRST_SYL) + rng.choice(LAST_SYL))
        if n.lower() in used or n.lower() in cf_strings:
            continue
        nt = set(tok(n, add_special_tokens=False).input_ids)
        nt |= set(tok(" " + n, add_special_tokens=False).input_ids)
        if (nt & b_value_tokens) or any(v in n.lower() for v in all_values):
            rejected += 1
            continue
        used.add(n.lower())
        names.append(n)

    people, retries = [], 0
    for n in names:
        for _ in range(MAX_PERSON_RETRIES):
            vals = {at: rng.choice(pools[at]) for at in attrs}
            ids = [tok_of[at][vals[at]] for at in attrs]
            if len(set(ids)) == len(ids):
                break
            retries += 1
        else:
            raise SystemExit(f"could not give {n} four first-token-distinct values in "
                             f"{MAX_PERSON_RETRIES} tries; pools too small or too aliased")
        people.append({"name": n, **vals})
    print(f"people: {len(people)} invented, none a CounterFact string or a B value; "
          f"{rejected} names rejected, {retries} value resamples")

    # --- ASSERTION 4: the copy shortcut is not built into B ------------------------------------
    for p in people:
        nt = set(tok(p["name"], add_special_tokens=False).input_ids)
        nt |= set(tok(" " + p["name"], add_special_tokens=False).input_ids)
        assert not (nt & b_value_tokens), f"name {p['name']!r} shares a token with a B value"
        for at in attrs:
            assert p[at].lower() not in p["name"].lower()
    # ASSERTION 1: per-attribute and within-person first-token distinctness
    for at in attrs:
        ids = [tok_of[at][v] for v in pools[at]]
        assert len(set(ids)) == len(ids), f"pool {at} has duplicate first tokens"
    for p in people:
        ids = [tok_of[at][p[at]] for at in attrs]
        assert len(set(ids)) == len(ids), f"{p['name']} has colliding value first tokens"
    print("assertions hold: pool distinct, within-person distinct, A-disjoint, name-value clean")

    # --- eval items: held-out phrasings only, rotated so all three are used --------------------
    evals = []
    for i, p in enumerate(people):
        for j, at in enumerate(attrs):
            held = split(at)[1][(i + j) % len(split(at)[1])]
            full = held.format(name=p["name"], v=p[at])
            prompt = full[:full.index(p[at])].rstrip()
            assert p[at] not in prompt, f"value leaked into prompt for {p['name']}/{at}"
            assert prompt not in cf_prompts, f"B eval prompt {prompt!r} IS a CounterFact probe"
            evals.append({"name": p["name"], "attr": at, "value": p[at],
                          "prompt": prompt, "first_token": tok_of[at][p[at]]})

    train_t = {at: split(at)[0] for at in attrs}
    eval_t = {at: split(at)[1] for at in attrs}
    os.makedirs(a.out_dir, exist_ok=True)
    out = os.path.join(a.out_dir, f"b_facts{'_' + a.tag if a.tag else ''}.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"attrs": attrs, "people": people, "pools": pools,
                   "train_templates": train_t, "eval_templates": eval_t, "eval": evals}, fh,
                  indent=2)

    # --- what a rendered document looks like, and how big an epoch is -------------------------
    p0 = people[0]
    sents = [rng.choice(train_t[at]).format(name=p0["name"], v=p0[at]) for at in attrs]
    rng.shuffle(sents)
    doc = " ".join(sents)
    n_tok = len(tok(doc, add_special_tokens=False).input_ids)
    print(f"\nfacts: {len(people)} people x {len(attrs)} attributes = "
          f"{len(people)*len(attrs)} facts; eval items {len(evals)}")
    print(f"  train phrasings/attr {len(train_t[attrs[0]])}, held out {len(eval_t[attrs[0]])}")
    print(f"  a rendered document is ~{n_tok} tokens; distinct renderings per person ~"
          f"{len(attrs)}! x {len(train_t[attrs[0]])}**{len(attrs)} = "
          f"{24 * len(train_t[attrs[0]])**len(attrs):,}")
    for at in attrs:
        c = Counter(p[at] for p in people)
        print(f"  {at:>12}: {len(c)}/{len(pools[at])} values used, max {c.most_common(1)[0][1]}x")
    print(f"  example document:\n    {doc}")
    print(f"  example eval prompt: {evals[0]['prompt']!r} -> {evals[0]['value']!r} "
          f"(first token {evals[0]['first_token']})")
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
