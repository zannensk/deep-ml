import numpy as np

def vit_encode(image, patch_size: int, W_patch, cls_token, pos_embed, W_qkv, W_o, num_heads: int):
    """
    Forward pass of a minimal ViT encoder block.

    Returns the final embedding of the CLS token as a list of length D.
    """
    image = np.asarray(image)
    W_patch = np.asarray(W_patch)
    cls_token = np.asarray(cls_token)
    pos_embed = np.asarray(pos_embed)
    W_qkv = np.asarray(W_qkv)
    W_o = np.asarray(W_o)

    H, W, C = image.shape
    P = patch_size
    D = W_patch.shape[1]

    patches = (
        image
        .reshape(H//P, P, W//P, P, C)
        .transpose(0, 2, 1, 3, 4)
        .reshape(-1, P * P * C)
    )

    patch_tokens = patches @ W_patch
    tokens = np.concatenate(
        [cls_token.reshape(1, D), patch_tokens],
        axis=0
    )

    X = tokens + pos_embed
    QKV = X @ W_qkv
    Q, K, V = np.split(QKV, 3, axis=-1)
    L = X.shape[0]
    d_h = D // num_heads

    Q = Q.reshape(L, num_heads, d_h).transpose(1, 0, 2)
    K = K.reshape(L, num_heads, d_h).transpose(1, 0, 2)
    V = V.reshape(L, num_heads, d_h).transpose(1, 0, 2)

    scores = Q @ K.transpose(0, 2, 1)
    scores = scores - np.max(scores, axis=-1, keepdims=True)
    scores = scores / np.sqrt(d_h)
    exp_scores = np.exp(scores)
    attention = exp_scores / np.sum(
        exp_scores, axis=-1, keepdims=True
    )

    context = attention @ V
    context = context.transpose(1, 0, 2).reshape(L, D)

    output = context @ W_o
    return output[0].tolist()