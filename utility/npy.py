import numpy as np

# Load LR dictionary
D_L = np.load("/home/fdbdfg/VScode/cvproject/output/D_L_x2.npy")

# Load HR dictionary
D_H = np.load("/home/fdbdfg/VScode/cvproject/output/D_H_x2.npy")

print("D_L shape:", D_L.shape)  # e.g., (p*p, n_atoms)
print("D_H shape:", D_H.shape)  # e.g., (p_hr*p_hr, n_atoms)

