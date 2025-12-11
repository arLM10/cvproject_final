# Global configuration - edit paths and hyperparams here

# Paths (set these to your local dataset folders)
HR_TRAIN_DIR = "/home/fdbdfg/Desktop/cvproject/train512"       # put HR training images here (BSD100/Urban100)
HR_TEST_DIR  = "/home/fdbdfg/Desktop/cvproject/test_data20"        # put HR test images here
OUTPUT_DIR   = "/home/fdbdfg/Desktop/cvproject/output"              # where results and dicts will be saved

# Scales and sizes
SCALES = [2, 3, 4]                   # scales to train/evaluate
LR_PATCH_SIZE = 5                    # LR patch size
HR_PAD = 2                           # HR patch size = LR_PATCH_SIZE * scale + HR_PAD
UPSCALE_PAD = 2                      # unused for now, kept for compatibility

# Dictionary / K-SVD params
DICT_ATOMS = 256
SPARSITY = 4                         # target non-zeros in OMP
KSVD_ITERS = 6

# Edge filtering
EDGE_THRESHOLD = 0.8                  # gradient mean threshold (tune if too few/too many patches)

# Inference / aggregation
PATCH_STEP = 2                        # sliding step
BACKPROJECTION_ITERS = 8              # optional refine iterations

# Patch budget (None or 0 = no cap)
MAX_PATCHES = 300000                   # no cap; scan all images

# Misc
RANDOM_SEED = 42
N_JOBS = -1                           # for sklearn OMP etc, -1 uses all cores

