import numpy as np

B = 10 #number of instances

# Generate random binary sigma vectors (m x n)
np.random.seed(42)  # Set seed for reproducibility

for i in range(B):
    for n in range(10,60,10):
        for m in range(10, 110, 10):
            sigma = np.random.randint(0, 2, size=(m, n))  # Random 0-1 vectors
            
            filename = f"I_{n}_{m}_{i}.txt"
            
            # Write to a text file
            with open(filename, "w") as f:
                f.write(f"n = {n}\n")
                f.write(f"m = {m}\n")
                f.write("sigma =\n")
                
                # Print the sigma matrix row by row
                for row in sigma:
                    f.write(" ".join(map(str, row)) + "\n")

