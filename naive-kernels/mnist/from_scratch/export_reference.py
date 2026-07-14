"""
Dump the PyTorch answer key (_pytorch_grads.npz) to raw float32/int32 .bin
files that the C program can fread(). Run once:
    python src/mnist/from_scratch/export_reference.py
"""
import numpy as np, os

HERE = os.path.dirname(__file__)
ans = np.load(os.path.join(HERE, "_pytorch_grads.npz"))
ref = os.path.join(HERE, "ref")
os.makedirs(ref, exist_ok=True)

def dump(name, arr, dtype):
    arr.astype(dtype).tofile(os.path.join(ref, name))

# inputs / params (the state BEFORE the SGD step)
dump("X.bin",  ans["X"],  np.float32)   # (8, 784)
dump("y.bin",  ans["y"],  np.int32)     # (8,)
dump("W1.bin", ans["W1"], np.float32)   # (784, 256)
dump("b1.bin", ans["b1"], np.float32)   # (256,)
dump("W2.bin", ans["W2"], np.float32)   # (256, 10)
dump("b2.bin", ans["b2"], np.float32)   # (10,)

# expected outputs (the answer key)
np.array([ans["loss"]], np.float32).tofile(os.path.join(ref, "loss.bin"))
dump("dW1.bin", ans["dW1"], np.float32)
dump("db1.bin", ans["db1"], np.float32)
dump("dW2.bin", ans["dW2"], np.float32)
dump("db2.bin", ans["db2"], np.float32)

print("wrote reference .bin files to", ref)
print("expected loss =", float(ans["loss"]))
