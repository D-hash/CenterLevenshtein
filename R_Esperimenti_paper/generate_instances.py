import random
import math

def random_binary_string(n):
    length = random.randint(math.ceil(n/2), n)
    return ''.join(random.choice('01') for _ in range(length))

random.seed(2025)
for n in [5, 10, 15, 20]:
    for m in [3]:
        for i in range(3):
            with open(f"I_{n}_{m}_{i}.txt", "w") as w:
                w.write(f"n = {n}\n")
                w.write(f"m = {m}\n")
                w.write(f"sigma =\n")
                for j in range(m):
                    s = random_binary_string(n)
                    w.write(f"{s}\n")
