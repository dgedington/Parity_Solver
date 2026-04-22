#!/usr/bin/env python3
"""
parity_solver.py
================
Given a parity vector (a string of O/E or 1/0 representing odd/even steps),
solve the Collatz path equation:

    2^m * M  -  3^n * N  =  f

for all positive integer solutions (M, N), where:
    m = total steps (length of vector)
    n = number of odd steps
    f = accumulated residue (from positional formula)

Usage:
    python parity_solver.py OOEE
    python parity_solver.py 1100
    python parity_solver.py OEOEOE --solutions 10
    python parity_solver.py OOOEEE --delta
"""

import argparse
import math
from fractions import Fraction


# ── Core computations ─────────────────────────────────────────────

def parse_path(raw: str) -> str:
    """Normalise input to a string of 'O' and 'E'."""
    s = raw.upper().strip()
    s = s.replace('1', 'O').replace('0', 'E')
    for ch in s:
        if ch not in ('O', 'E'):
            raise ValueError(f"Invalid character '{ch}'. Use O/E or 1/0.")
    if not s:
        raise ValueError("Path vector cannot be empty.")
    return s


def compute_f(path: str) -> int:
    """
    Positional formula for f.
    At each O-step at 1-indexed position i: f = 3*f + 2^(i-1)
    """
    f = 0
    for i, c in enumerate(path, start=1):
        if c == 'O':
            f = 3 * f + 2 ** (i - 1)
    return f


def find_residue_b(path: str) -> int:
    """
    Find b = the unique value in [0, 2^m) such that any N ≡ b (mod 2^m)
    follows this parity path. Equivalently: the minimum valid seed.
    """
    m = len(path)
    for N in range(1, 2 ** m + 2):
        val = N
        ok = True
        for c in path:
            if c == 'O' and val % 2 == 0: ok = False; break
            if c == 'E' and val % 2 == 1: ok = False; break
            val = (3 * val + 1) // 2 if c == 'O' else val // 2
        if ok:
            return N % (2 ** m)
    raise RuntimeError("Could not find residue b — check path validity.")


# ── Extended GCD and Diophantine solver ───────────────────────────

def extended_gcd(a: int, b: int):
    """Returns (g, x, y) with a*x + b*y = g = gcd(a,b)."""
    if b == 0:
        return a, 1, 0
    g, x, y = extended_gcd(b, a % b)
    return g, y, x - (a // b) * y


def solve_path(path: str):
    """
    Full analysis of a parity path.

    Equation:  2^m * M - 3^n * N = f

    General solution:
        M = M0 + 3^n * t
        N = N0 + 2^m * t
    for integer t, with t >= t_min for both M,N > 0.

    Returns a result dict.
    """
    m = len(path)
    n = path.count('O')
    f = compute_f(path)
    b = find_residue_b(path)

    a = 2 ** m      # coefficient of M
    bv = 3 ** n     # coefficient of N
    gap = a - bv    # 2^m - 3^n

    # gcd(2^m, 3^n) = 1 always, so solutions always exist.
    g, u, v = extended_gcd(a, bv)
    assert g == 1

    # Particular solution to a*M - bv*N = f:
    #   a*u + bv*v = 1  (from extended_gcd)
    #   a*(f*u) - bv*(-f*v) = f
    # Reduce M0 into [0, bv) for the canonical representative.
    M0 = (f * u) % bv
    N0 = (a * M0 - f) // bv

    step_M = bv   # M increases by 3^n per unit t
    step_N = a    # N increases by 2^m per unit t

    # Smallest t with M > 0 and N > 0
    t_min_M = math.ceil((1 - M0) / step_M)
    t_min_N = math.ceil((1 - N0) / step_N)
    t_min   = max(t_min_M, t_min_N)

    # Loop condition: M = N  =>  M0 + 3^n*t = N0 + 2^m*t
    #   (3^n - 2^m)*t = N0 - M0
    #   -gap * t = N0 - M0
    loop_solution = None
    if gap != 0 and (N0 - M0) % gap == 0:
        t_loop = (N0 - M0) // gap
        M_loop = M0 + step_M * t_loop
        N_loop = N0 + step_N * t_loop
        if M_loop > 0 and M_loop == N_loop:
            loop_solution = (M_loop, N_loop, t_loop)

    return {
        'm': m, 'n': n, 'f': f, 'b': b,
        'gap': gap,
        'group': 'A' if gap > 0 else ('C' if gap < 0 else '='),
        'M0': M0, 'N0': N0,
        'step_M': step_M, 'step_N': step_N,
        't_min': t_min,
        'loop': loop_solution,
        'path': path,
    }


def generate_solutions(result: dict, count: int):
    """Yield the first `count` positive (M, N, t) solutions."""
    M0, N0 = result['M0'], result['N0']
    sM, sN = result['step_M'], result['step_N']
    t = result['t_min']
    found = 0
    while found < count:
        M = M0 + sM * t
        N = N0 + sN * t
        if M > 0 and N > 0:
            yield M, N, t
            found += 1
        t += 1


# ── Display ───────────────────────────────────────────────────────

def display(result: dict, n_solutions: int = 8, show_delta: bool = False):
    m   = result['m']
    n   = result['n']
    f   = result['f']
    b   = result['b']
    gap = result['gap']
    M0  = result['M0']
    N0  = result['N0']
    sM  = result['step_M']
    sN  = result['step_N']

    print()
    print("=" * 62)
    print(f"  Path:  {result['path']}")
    print("=" * 62)

    print(f"\n  Parameters")
    print(f"  {'m (total steps)':<28} {m}")
    print(f"  {'n (odd steps)':<28} {n}")
    print(f"  {'f (accumulated residue)':<28} {f}")
    print(f"  {'b (residue class N mod 2^m)':<28} {b}  [= {b} mod {2**m}]")
    print(f"  {'gap = 2^m - 3^n':<28} {gap}  [{result['group']} group]")

    print(f"\n  Equation")
    print(f"  {2**m}·M  −  {3**n}·N  =  {f}")
    print(f"  i.e.  (3^{n}·N + {f}) / 2^{m}  =  M")

    print(f"\n  General solution  (t ∈ ℤ, t ≥ {result['t_min']} for M,N > 0)")
    print(f"  M  =  {M0}  +  {sM}·t        (steps of 3^n = {sM})")
    print(f"  N  =  {N0}  +  {sN}·t        (steps of 2^m = {sN})")

    # Loop
    print(f"\n  Loop analysis  (M = N)")
    if result['loop']:
        Ml, Nl, tl = result['loop']
        print(f"  *** LOOP FOUND at t = {tl}: M = N = {Ml} ***")
        delta_loop = Fraction(f, Ml) - gap
        print(f"  δ = f/N − gap = {float(delta_loop):.6f}  "
              f"({'= 0 ✓' if delta_loop == 0 else '≠ 0 !'})")
    else:
        # Show why no loop: gap * t = N0 - M0 has no valid solution
        diff = N0 - M0
        print(f"  No loop: need gap·t = N0−M0 = {diff},  gap = {gap}")
        if gap == 0:
            print(f"  gap = 0: impossible (2^m = 3^n contradicts FTA)")
        elif diff % gap != 0:
            print(f"  {diff} is not divisible by {gap}  ⟹  no integer t exists")
        else:
            tl = diff // gap
            Ml = M0 + sM * tl
            Nl = N0 + sN * tl
            print(f"  t = {tl} gives M = {Ml}, N = {Nl}  (one is ≤ 0)")

    # Solutions table
    print(f"\n  First {n_solutions} positive solutions")
    print(f"  {'t':>5}  {'N':>14}  {'M':>14}  {'M/N':>10}  {'verify':>8}", end="")
    if show_delta:
        print(f"  {'δ = f/N − gap':>16}", end="")
    print()
    print(f"  {'-'*5}  {'-'*14}  {'-'*14}  {'-'*10}  {'-'*8}", end="")
    if show_delta:
        print(f"  {'-'*16}", end="")
    print()

    for M, N, t in generate_solutions(result, n_solutions):
        ratio = M / N
        check = (2**m * M - 3**n * N == f)
        print(f"  {t:>5}  {N:>14,}  {M:>14,}  {ratio:>10.6f}  {'✓' if check else '✗':>8}", end="")
        if show_delta:
            delta = Fraction(f, N) - gap
            print(f"  {float(delta):>16.6f}", end="")
        print()

    # Extra: the M=1 and M=N questions
    print(f"\n  Special seeds")
    # N such that M=1: N = (2^m - f) / 3^n  (if integer and positive)
    num = 2**m - f
    if num > 0 and num % (3**n) == 0:
        N_gives_M1 = num // 3**n
        if N_gives_M1 % (2**m) == b:
            print(f"  M=1 seed: N = (2^m−f)/3^n = {N_gives_M1}  ✓ in residue class b={b}")
        else:
            print(f"  M=1 seed: N = {N_gives_M1}  (not in residue class b={b})")
    else:
        print(f"  M=1: no valid seed (2^m−f = {num} not divisible by 3^n={3**n})")

    print()


# ── CLI ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Solve the Collatz path equation for a given parity vector.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python parity_solver.py OOEE
  python parity_solver.py 1100
  python parity_solver.py OEOEOE --solutions 6
  python parity_solver.py OOOEEE --delta
  python parity_solver.py OOEEOOE --solutions 12 --delta
        """
    )
    parser.add_argument('path',
        help='Parity vector: O=odd step, E=even step (or 1/0)')
    parser.add_argument('--solutions', '-s', type=int, default=8,
        metavar='N', help='Number of solutions to display (default: 8)')
    parser.add_argument('--delta', '-d', action='store_true',
        help='Show δ = f/N − gap for each solution')
    args = parser.parse_args()

    try:
        path = parse_path(args.path)
        result = solve_path(path)
        display(result, n_solutions=args.solutions, show_delta=args.delta)
    except ValueError as e:
        print(f"Error: {e}")
        raise SystemExit(1)


if __name__ == '__main__':
    main()
