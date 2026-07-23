"""Unit tests for the torch-free core logic (rhyme, prompts, positions helpers).

Run:  python -m pytest tests/  (or)  python tests/test_core.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from opi import rhyme as R
from opi import prompts as P


def test_rhyme_basic():
    assert R.do_rhyme("fright", "night") is True
    assert R.do_rhyme("fright", "dog") is False
    assert R.do_rhyme("fright", "fright") is True          # identical -> rhyme
    assert R.do_rhyme("xyzzyq", "night") is None            # OOV -> unknown


def test_rhyme_family():
    assert R.rhyme_family("fright") == R.rhyme_family("night")
    assert R.rhyme_family("fright") != R.rhyme_family("doom")
    assert R.rhyme_family("qwertyq") is None


def test_word_extraction():
    assert R.extract_first_line_word("A rhyming couplet:\nShe felt a sense of fright,\n") == "fright"
    assert R.extract_second_line_word_with_newline(" and vanished into night,\nmore") == "night"
    assert R.extract_second_line_word_without_newline(" into the night. Then") == "night"


def test_extract_rhyme_word_full_text():
    prompt = "A rhyming couplet:\nShe felt a sense of fright,\n"
    full = prompt + "and vanished into the night,\n"
    assert R.extract_rhyme_word(full, prompt) == "night"


def test_rhyme_rate():
    prompt = "A rhyming couplet:\nShe felt a sense of fright,\n"
    comps = [prompt + "and vanished into night,\n",     # rhymes
             prompt + "and ran toward the door,\n"]      # does not
    rate = R.rhyme_rate(comps, prompt, "fright")
    assert abs(rate - 0.5) < 1e-9


def test_prompt_format():
    p = P.format_prompt("She felt a sense of fright")
    assert p == "A rhyming couplet:\nShe felt a sense of fright,\n"
    assert p.count("\n") == 2


def test_bank_and_pool():
    bank = P.build_rhyme_word_bank(max_words_per_family=8, min_zipf=3.3)
    assert len(bank) >= 20
    for fam, words in bank.items():
        assert len(words) >= 2
        for w in words:
            assert R.rhyme_family(w) == fam                 # bucket is one family

    seen, held = P.split_families(bank, 0.2, seed=0)
    assert set(seen).isdisjoint(held)

    pool = P.build_prompt_pool(200, seen, bank, seed=0)
    assert len(pool) == 200
    prompts_seen = {r["prompt"] for r in pool}
    assert len(prompts_seen) == 200                         # all unique
    for r in pool[:20]:
        assert r["prompt"].startswith("A rhyming couplet:\n")
        assert r["prompt"].endswith("\n")
        assert r["rhyme_family"] in seen


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"All {len(fns)} tests passed.")


if __name__ == "__main__":
    _run_all()
