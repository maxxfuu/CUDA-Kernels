"""
=============================================================================
  MNIST 2-layer MLP  —  NUMPY FROM SCRATCH  (you write this, with me)
=============================================================================

We reproduce the forward AND backward pass of pytorch_annotated.py using only
NumPy — no autograd. The harness at the bottom loads the PyTorch answer key
(_pytorch_grads.npz: same X, y, W1, b1, W2, b2) and checks every number you
produce against it. Green ✅ = your math matches PyTorch exactly.

Fill in the functions marked TODO. Run the file any time — the harness reports
which pieces are done and which still need work, so you get feedback as you go:

    conda activate gpt3
    python src/mnist/from_scratch/numpy_scratch.py

Weight convention (same as the reference):  W is (in, out), so  y = X @ W + b.
=============================================================================
"""

import numpy as np

DATA = "/Users/maxfu/Desktop/Code/CUDA-Kernels/src/mnist/data"


# ═════════════════════════════════════════════════════════════════════════
#  FORWARD
# ═════════════════════════════════════════════════════════════════════════

def linear_forward(x, W, b):
    """
    Fully-connected layer:  y = x @ W + b

        x (B, in) ──@── W (in, out) ──+── b (out,) ──►  y (B, out)

    The bias (out,) is BROADCAST across all B rows.
    TODO: return the (B, out) result.
    """
    raise NotImplementedError("linear_forward")


def relu(z):
    """
    ReLU, elementwise:  relu(z) = max(z, 0)      shape unchanged.
        z (B, H) ──►  a (B, H)   with every negative entry set to 0
    TODO: use np.maximum.
    """
    raise NotImplementedError("relu")


def softmax(z):
    """
    Row-wise softmax turning logits into probabilities that sum to 1 per row.

        softmax(z)_i = exp(z_i) / Σ_j exp(z_j)      (computed per row)

    STABILITY: subtract each row's max BEFORE exp, so we never exp() a big
    positive number and overflow. Subtracting a constant per row doesn't change
    the result mathematically (it cancels in the ratio) but keeps floats sane.

        z (B, 10) ──►  probs (B, 10),  each row sums to 1

    TODO: implement with axis=1 reductions and keepdims=True.
    """
    raise NotImplementedError("softmax")


def cross_entropy_loss(logits, y):
    """
    Mean cross-entropy over the batch.

        probs = softmax(logits)                      (B, 10)
        loss  = -mean_over_batch( log(probs[row, correct_class]) )

    Pick out probs[i, y[i]] with fancy indexing: probs[np.arange(B), y].
    TODO: return the scalar loss (divide by B -> 'mean' reduction).
    """
    raise NotImplementedError("cross_entropy_loss")


# ═════════════════════════════════════════════════════════════════════════
#  BACKWARD   (the part we're really here for)
# ═════════════════════════════════════════════════════════════════════════

def softmax_cross_entropy_grad(logits, y):
    """
    Gradient of the mean cross-entropy loss w.r.t. the LOGITS. This is the
    famous fused result — the whole softmax+log+NLL collapses to:

        dlogits = (softmax(logits) - onehot(y)) / B

    Intuition: push the predicted probability DOWN on wrong classes and UP on
    the correct class; magnitude = how wrong we were. The /B matches the mean
    reduction in the loss (and PyTorch's default).

        logits (B,10), y (B,) ──►  dlogits (B,10)

    TODO: build the one-hot of y, subtract, divide by B.
    """
    raise NotImplementedError("softmax_cross_entropy_grad")


def relu_backward(grad_out, z):
    """
    Send the gradient back through relu. relu passed positive z unchanged and
    zeroed negatives, so its local derivative is 1 where z>0, else 0:

        grad_in = grad_out * (z > 0)

        grad_out (B,H), z (B,H) ──►  grad_in (B,H)   (killed where z was ≤ 0)

    TODO: multiply grad_out by the boolean mask (z > 0).
    """
    raise NotImplementedError("relu_backward")


def linear_backward(grad_out, x, W):
    """
    Backward through  y = x @ W + b.  Given dL/dy (grad_out), produce the three
    gradients. THE SHAPE RULE is your safety net — each result matches the shape
    of the thing it's the gradient of:

        dW = xᵀ @ grad_out        -> same shape as W   (in, out)
        db = sum(grad_out, 0)     -> same shape as b   (out,)
        dx = grad_out @ Wᵀ        -> same shape as x   (B, in)   (passed left)

        grad_out (B,out), x (B,in), W (in,out)  ──►  (dx, dW, db)

    TODO: return (dx, dW, db) in that order.
    """
    raise NotImplementedError("linear_backward")


# ═════════════════════════════════════════════════════════════════════════
#  VERIFICATION HARNESS   (provided — do not edit; it grades you vs PyTorch)
# ═════════════════════════════════════════════════════════════════════════

def _check(name, got, want, tol=1e-5):
    err = float(np.max(np.abs(got - want)))
    tag = "✅" if err < tol else "❌"
    print(f"  {tag} {name:5s}  max|Δ vs PyTorch| = {err:.2e}")
    return err < tol


def _stage(label, fn):
    print(f"\n[{label}]")
    try:
        return fn(), True
    except NotImplementedError as e:
        print(f"  ⏳ not implemented yet: {e}")
        return None, False


def main():
    ans = np.load(f"{DATA}/../from_scratch/_pytorch_grads.npz")
    X, y = ans["X"], ans["y"]
    W1, b1, W2, b2 = ans["W1"], ans["b1"], ans["W2"], ans["b2"]

    # ---- FORWARD ----
    state = {}
    def do_forward():
        z1 = linear_forward(X, W1, b1)
        a1 = relu(z1)
        z2 = linear_forward(a1, W2, b2)
        loss = cross_entropy_loss(z2, y)
        state.update(z1=z1, a1=a1, z2=z2)
        return loss
    loss, ok = _stage("forward + loss", do_forward)
    if ok:
        _check("loss", np.array(loss), np.array(ans["loss"]))

    if not ok:
        print("\nFinish the forward functions, then we tackle backward.\n")
        return

    # ---- BACKWARD ----
    def do_backward():
        z1, a1, z2 = state["z1"], state["a1"], state["z2"]
        dz2 = softmax_cross_entropy_grad(z2, y)
        da1, dW2, db2 = linear_backward(dz2, a1, W2)
        dz1 = relu_backward(da1, z1)
        dX, dW1, db1 = linear_backward(dz1, X, W1)
        return dW1, db1, dW2, db2
    grads, ok = _stage("backward (dW1, db1, dW2, db2)", do_backward)
    if ok:
        dW1, db1, dW2, db2 = grads
        allok = all([
            _check("dW1", dW1, ans["dW1"]),
            _check("db1", db1, ans["db1"]),
            _check("dW2", dW2, ans["dW2"]),
            _check("db2", db2, ans["db2"]),
        ])
        print("\n🎉 ALL GRADIENTS MATCH PYTORCH." if allok
              else "\nSome gradients differ — check shapes/transposes above.")


if __name__ == "__main__":
    main()
