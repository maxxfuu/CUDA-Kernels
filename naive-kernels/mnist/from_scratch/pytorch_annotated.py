"""
=============================================================================
  MNIST 2-layer MLP  —  ANNOTATED PYTORCH REFERENCE  (read this file)
=============================================================================

This is the "autograd does the work" version. You READ this to understand the
whole pipeline and — most importantly — to see *what gradients autograd is
secretly computing* inside `loss.backward()`. Every one of those gradients is
something you will re-implement by hand in `numpy_scratch.py`.

Architecture:  784 -> Linear -> 256 -> ReLU -> Linear -> 10

-----------------------------------------------------------------------------
FORWARD PASS  (shapes for a batch of B=8)
-----------------------------------------------------------------------------

   X            W1               b1            z1            a1
 (8,784) ──@── (784,256) ──+── (256,) ──►  (8,256) ──relu──► (8,256)
    │                                          │                 │
  input      "for each of 8 rows, mix 784   pre-activation   post-activation
             inputs into 256 hidden units"  (before relu)    (negatives -> 0)

   a1           W2               b2            z2 = logits
 (8,256) ──@── (256,10) ──+── (10,) ──►    (8,10)
                                              │
                                         raw scores, one per class (0..9)

   logits ──softmax──► probs (8,10) ──cross_entropy(target)──► loss (scalar)

Every "@" is a matmul; every "+bias" broadcasts the bias across the 8 rows.

-----------------------------------------------------------------------------
BACKWARD PASS  (what loss.backward() computes — this is the whole point)
-----------------------------------------------------------------------------

Gradients flow RIGHT-TO-LEFT. Each node receives dL/d(its output) from the
node after it, and must produce dL/d(its inputs) to pass leftward. The name of
the game: "given the gradient of the loss w.r.t. my output, what is it w.r.t.
my inputs and my parameters?"

   dz2 (8,10)  ◄── softmax+CE fused gradient:  dz2 = (softmax(z2) - onehot)/B
      │
      ├─► dW2 = a1ᵀ @ dz2        (256,10)   ← how each weight2 should change
      ├─► db2 = sum(dz2, axis=0) (10,)      ← how each bias2 should change
      └─► da1 = dz2 @ W2ᵀ        (8,256)    ← gradient handed left to ReLU
             │
   dz1 (8,256) ◄── relu backward:  dz1 = da1 * (z1 > 0)   (kill grad where dead)
      │
      ├─► dW1 = Xᵀ @ dz1         (784,256)
      ├─► db1 = sum(dz1, axis=0) (256,)
      └─► dX  = dz1 @ W1ᵀ        (8,784)    ← unused (no layer before input)

Notice the SHAPE RULE that makes this memorizable:
  • dW always has the SAME shape as W, built as (inputᵀ @ grad_output).
  • db always has the SAME shape as b, built as (sum of grad_output over batch).
  • grad-to-pass-left has the SAME shape as that layer's input, (grad @ Wᵀ).

-----------------------------------------------------------------------------
SGD UPDATE
-----------------------------------------------------------------------------
   W ← W - lr * dW        (step DOWNHILL along the gradient)
=============================================================================
"""

import numpy as np
import torch

torch.manual_seed(0)
DATA = "/Users/maxfu/Desktop/Code/CUDA-Kernels/src/mnist/data"

B, IN, HID, OUT, LR = 8, 784, 256, 10, 0.01

# ── one real MNIST batch, normalized exactly like the book's pipeline ──────
mean, std = 0.1307, 0.3081
X_np = (np.fromfile(f"{DATA}/X_train.bin", dtype=np.float32).reshape(60000, 784)[:B] - mean) / std
y_np = np.fromfile(f"{DATA}/y_train.bin", dtype=np.int32)[:B]
X = torch.from_numpy(X_np).float()                 # (8, 784)  the inputs
target = torch.from_numpy(y_np).long()             # (8,)      correct digit per row

# ── parameters. requires_grad_(True) tells autograd: "track ops on these so ─
#    you can fill in their .grad when I call loss.backward()." ──────────────
def he_uniform(fan_in, fan_out):
    scale = (6.0 / fan_in) ** 0.5
    return (torch.rand(fan_in, fan_out) * 2 - 1) * scale

W1 = he_uniform(IN, HID).requires_grad_(True)      # (784, 256)
b1 = torch.zeros(HID).requires_grad_(True)         # (256,)
W2 = he_uniform(HID, OUT).requires_grad_(True)     # (256, 10)
b2 = torch.zeros(OUT).requires_grad_(True)         # (10,)

# =========================================================================
# FORWARD
# =========================================================================
# Layer 1:  (8,784) @ (784,256) -> (8,256), then +bias broadcasts over rows.
z1 = X @ W1 + b1                                    # (8,256)  pre-activation
a1 = torch.relu(z1)                                 # (8,256)  max(z1, 0)
# Layer 2:
z2 = a1 @ W2 + b2                                   # (8,10)   logits (raw scores)

# =========================================================================
# LOSS  —  cross-entropy = softmax + negative-log-likelihood, mean over batch
# =========================================================================
# F.cross_entropy takes RAW LOGITS (it does the softmax internally, in a
# numerically stable way) and the integer class targets. Default reduction is
# 'mean' -> it divides by the batch size B. Remember that /B: it's why the
# backward gradient below also carries a 1/B.
loss = torch.nn.functional.cross_entropy(z2, target)

# =========================================================================
# BACKWARD  —  one line. Autograd walks the graph above in reverse and fills
# in W1.grad, b1.grad, W2.grad, b2.grad using EXACTLY the formulas drawn in the
# header. We print them so you can compare against your NumPy version later.
# =========================================================================
loss.backward()

if __name__ == "__main__":
    print(f"loss = {loss.item():.6f}")
    print("forward shapes :", tuple(z1.shape), tuple(a1.shape), tuple(z2.shape))
    print("grad shapes    :",
          "dW1", tuple(W1.grad.shape), " db1", tuple(b1.grad.shape),
          " dW2", tuple(W2.grad.shape), " db2", tuple(b2.grad.shape))

    # ── SGD step done by hand so you see optim.SGD is not magic ────────────
    with torch.no_grad():                          # don't track the update itself
        W1 -= LR * W1.grad
        b1 -= LR * b1.grad
        W2 -= LR * W2.grad
        b2 -= LR * b2.grad
    print("did one SGD step (W <- W - lr*dW)")

    # These exact numbers are the answer key for numpy_scratch.py. We stash the
    # gradients so the NumPy file can load and diff against them.
    np.savez(f"{DATA}/../from_scratch/_pytorch_grads.npz",
             X=X_np, y=y_np,
             W1=W1.detach().numpy() + LR * W1.grad.numpy(),   # pre-update W1
             b1=b1.detach().numpy() + LR * b1.grad.numpy(),
             W2=W2.detach().numpy() + LR * W2.grad.numpy(),
             b2=b2.detach().numpy() + LR * b2.grad.numpy(),
             loss=loss.item(),
             dW1=W1.grad.numpy(), db1=b1.grad.numpy(),
             dW2=W2.grad.numpy(), db2=b2.grad.numpy())
    print("saved gradients -> from_scratch/_pytorch_grads.npz  (answer key)")
