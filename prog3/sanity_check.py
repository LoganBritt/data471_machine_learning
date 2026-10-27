#!/usr/bin/env python3
"""
DATA 471 Program 3 -- student sanity checks.

    python sanity_check.py path/to/prog3.py [--data path/to/data]

These are a SMALL set of smoke tests for output formats and basic behavior.
Passing them does NOT mean you will get full credit: the real autograder uses
different datasets and checks more things (correct probabilities on other
shapes, numerical stability, dev-eval timing, checkpointing, early stopping,
how well your model learns, ...). Read the spec carefully and test those
yourself too.
"""
import argparse
import os
import re
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
VERBOSE_RE = re.compile(r"^\d+\.\d{3} batch, update \d+, epoch \d+$")
DEV_RE = re.compile(r"^dev loss=\d+\.\d{3} acc=\d\.\d{3} \(as of update (\d+), epoch (\d+)\)$")

results = []


def check(name, ok, hint=""):
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    if not ok and hint:
        print("       " + hint.replace("\n", "\n       "))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prog", nargs="?", default="prog3.py")
    ap.add_argument("--data", default=os.path.join(HERE, "data"))
    a = ap.parse_args()
    prog = os.path.abspath(a.prog)
    D = lambda f: os.path.join(a.data, f)
    if not os.path.isfile(prog):
        sys.exit(f"cannot find {prog}")
    if os.path.basename(prog) != "prog3.py" or \
            os.path.basename(os.path.dirname(prog)) != "prog3":
        print("WARNING: your file should be named prog3.py and live in a prog3/ directory\n")

    tmp = tempfile.mkdtemp(prefix="p3sanity_")

    def run(args, timeout=120):
        try:
            p = subprocess.run([sys.executable, prog] + args, cwd=tmp, text=True,
                               capture_output=True, timeout=timeout)
            return p.returncode, p.stdout, p.stderr
        except subprocess.TimeoutExpired:
            return -1, "", f"timed out after {timeout}s"

    def err(rc, se):
        return f"exit code {rc}\n" + "\n".join(se.strip().splitlines()[-5:])

    # 1. --help
    rc, out, se = run(["--help"])
    check("--help runs", rc == 0 and "--train" in out and "--max-steps" in out,
          "argparse should define --train/--predict/--eval, --max-steps, etc.\n" + err(rc, se))

    # 2. predict on the tiny example
    preds = os.path.join(tmp, "tiny_preds.npy")
    rc, out, se = run(["--predict", "--devx", D("tiny_dev_x.npy"),
                       "--model", D("tiny_model.npz"), "--preds", preds])
    want = np.load(D("tiny_expected_preds.npy"))
    if rc == 0 and os.path.exists(preds):
        P = np.load(preds)
        check("predict: preds.npy is N x C float64",
              P.shape == want.shape and P.dtype == np.float64,
              f"got shape {P.shape} dtype {P.dtype}, expected {want.shape} float64")
        check("predict: probabilities match expected (tiny example)",
              P.shape == want.shape and np.allclose(P, want, atol=1e-6),
              f"expected:\n{np.round(want, 4)}\ngot:\n{np.round(P, 4)}")
    else:
        check("predict: preds.npy is N x C float64", False,
              "no predictions file written\n" + err(rc, se))
        check("predict: probabilities match expected (tiny example)", False)

    # 3. eval on the tiny example
    rc, out, se = run(["--eval", "--devx", D("tiny_dev_x.npy"),
                       "--devy", D("tiny_dev_y.npy"), "--model", D("tiny_model.npz")])
    exp_acc = open(D("tiny_expected_acc.txt")).read().strip()
    check(f"eval: prints exactly '{exp_acc}' (tiny example)",
          rc == 0 and out.strip() == exp_acc,
          f"stdout was {out.strip()!r}" if rc == 0 else err(rc, se))

    # 4. train on blobs
    model = os.path.join(tmp, "blobs_model.npz")
    steps, freq, mb = 200, 50, 32
    rc, out, se = run(["--train", "--trainx", D("blobs_train_x.npy"),
                       "--trainy", D("blobs_train_y.npy"),
                       "--devx", D("blobs_dev_x.npy"), "--devy", D("blobs_dev_y.npy"),
                       "--model", model, "--lr", "0.1", "--mb", str(mb),
                       "--max-steps", str(steps), "--dev_freq", str(freq), "--verbose"])
    if rc != 0:
        check("train (blobs) runs", False, err(rc, se))
    else:
        lines = [l.strip() for l in out.splitlines() if l.strip()]
        vb = [l for l in lines if VERBOSE_RE.match(l)]
        dev = [DEV_RE.match(l) for l in lines]
        dev = [m for m in dev if m]
        check("train --verbose: one '<loss> batch, update U, epoch E' line per update",
              len(vb) == steps,
              f"found {len(vb)} matching lines, expected {steps}; first lines:\n"
              + "\n".join(lines[:3]))
        ups = [int(m.group(1)) for m in dev]
        check("train: 'dev loss=... acc=... (as of update U, epoch E)' lines every --dev_freq",
              ups == list(range(freq, steps + 1, freq)),
              f"dev evals found at updates {ups}, expected "
              f"{list(range(freq, steps + 1, freq))}")
        try:
            m = np.load(model)
            W, b = m["W"], m["b"]
            check("train: model.npz has W (2x3) and b (3,) float64",
                  W.shape == (2, 3) and b.shape == (3,) and
                  W.dtype == np.float64 and b.dtype == np.float64,
                  f"W {W.shape} {W.dtype}, b {b.shape} {b.dtype}")
            rc2, out2, se2 = run(["--eval", "--devx", D("blobs_dev_x.npy"),
                                  "--devy", D("blobs_dev_y.npy"), "--model", model])
            acc = float(out2.strip()) if rc2 == 0 and re.fullmatch(r"\d\.\d{3}", out2.strip()) else -1
            check("train+eval: blobs dev accuracy >= 0.90", acc >= 0.90,
                  f"got {out2.strip()!r} (a correct implementation gets ~0.97+)")
        except Exception as e:
            check("train: model.npz has W (2x3) and b (3,) float64", False,
                  f"could not load model['W'] / model['b']: {e}")

    # 5. binary (C=2) trains without crashing
    model = os.path.join(tmp, "bin_model.npz")
    rc, out, se = run(["--train", "--trainx", D("binary_train_x.npy"),
                       "--trainy", D("binary_train_y.npy"),
                       "--devx", D("binary_dev_x.npy"), "--devy", D("binary_dev_y.npy"),
                       "--model", model, "--lr", "0.1", "--max-steps", "300",
                       "--dev_freq", "100"])
    ok = rc == 0 and os.path.exists(model) and np.load(model)["W"].shape == (10, 2)
    check("train (binary, C=2, D=10) runs and saves W of shape (10, 2)", ok,
          err(rc, se) if rc else "")

    n = sum(results)
    print(f"\n{n}/{len(results)} sanity checks passed.")
    print("Reminder: the real autograder runs many more (and different) tests.")


if __name__ == "__main__":
    main()
