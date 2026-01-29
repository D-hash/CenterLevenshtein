import random
import os
from itertools import product, combinations

# ============================================================
#  HAMMING CODES (random sampling)
# ============================================================

def hamming_generator_matrix(r):
    n = 2**r - 1
    k = n - r
    G = []

    for i in range(k):
        row = [0]*k
        row[i] = 1
        for j in range(r):
            row.append(((i+1) >> j) & 1)
        G.append(row)
    return G, n, k

def hamming_encode(msg_bits, G):
    return [
        sum(m*g for m, g in zip(msg_bits, col)) % 2
        for col in zip(*G)
    ]

def sample_hamming(r, count):
    G, n, k = hamming_generator_matrix(r)
    pool = []
    for _ in range(count):
        msg = [random.randint(0,1) for _ in range(k)]
        cw = hamming_encode(msg, G)
        pool.append("".join(str(b) for b in cw))
    return pool


# ============================================================
#  REED–MULLER CODES (RM(1,m))
# ============================================================

def rm_basis(r, m):
    basis = []
    for deg in range(r+1):
        for combo in combinations(range(m), deg):
            basis.append(combo)
    return basis

def eval_monomial(monomial, x):
    v = 1
    for idx in monomial:
        v &= x[idx]
    return v

def sample_rm(r, m, count):
    basis = rm_basis(r, m)
    k = len(basis)
    pool = []

    for _ in range(count):
        msg = [random.randint(0,1) for _ in range(k)]
        cw = []
        for x in product([0,1], repeat=m):
            val = 0
            for coef, mon in zip(msg, basis):
                if coef:
                    val ^= eval_monomial(mon, x)
            cw.append(str(val))
        pool.append("".join(cw))

    return pool


# ============================================================
#  SAVE POOLS
# ============================================================

def save_pool(filename, pool):
    with open(filename, "w") as f:
        f.write("n = 16\n")
        f.write(f"m = {len(pool)}\n")
        f.write("sigma =\n")
        for s in pool:
            f.write(s + "\n")


# ============================================================
#  MAIN: produce 45 mixed-length pools
# ============================================================

def main():
    os.makedirs("pools", exist_ok=True)

    def mixed_hamming(count):
        # Only Hamming(3) and Hamming(4) remain (lengths 7 and 15)
        parts = [count // 2, count - count // 2]
        pool = []
        pool += sample_hamming(3, parts[0])
        pool += sample_hamming(4, parts[1])
        random.shuffle(pool)
        return pool


    def mixed_rm(count):
        # Only RM(1,3) and RM(1,4) remain (lengths 8 and 16)
        parts = [count // 2, count - count // 2]
        pool = []
        pool += sample_rm(1, 3, parts[0])
        pool += sample_rm(1, 4, parts[1])
        random.shuffle(pool)
        return pool


    ecc_sources = { "hamming": mixed_hamming, "rm": mixed_rm }

    sizes = [10, 20, 30, 40, 50]

    for ecc_name, generator in ecc_sources.items():
        for m in sizes:
            for copy_index in range(1, 4):
                pool = generator(m)
                filename = f"pools/{ecc_name}_{m}_{copy_index}.txt"
                save_pool(filename, pool)

    print("Generated 45 mixed-length pools in ./pools/")

if __name__ == "__main__":
    main()
