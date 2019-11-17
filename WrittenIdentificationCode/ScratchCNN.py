jdef forward_naive(x, w, b, conv_param):
    """

    - x: Input data
    - w: Filter weights
    - b: Biases
    - conv params holds size of stride & size of pad
    """
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    stride, pad  conv_param['stride'], conv_param['pad']
    H_out = 1 + (H + 2 * pad - HH) // stride
    W_out = 1 + (W + 2 * pad - WW) // stride
    out = np.zeros((N, F, H_out, W_out))
    pad_widths = ((0, ), (0, ), (pad, ), (pad, ))
    xpad = np.pad(x, pad_widths, 'constant')
    Npad, Cpad, Hpad, Wpad = xpad.shape

    for n in range(N):
        for f in range(F):
            for i in range(0, Hpad - (HH - 1), stride):
                for j in range(0, Wpad - (WW - 1), stride):
                    prod = np.sum(np.multiply(w[f,...], xpad(n, :, i:i + HH, j:j + WW)))
                    out[n, f, int(i/stride), int(j/stride)] = prod + b[f]
    cache = (x, w, b, conv_param)
    return out, cache


def pool_naive(x, pool_param):
    """

    x: input
    pool_param: dict with pool height width, and stride
    """

    Hp = pool_param['pool_height']
    Wp = pool_param['pool_width']
    s = pool_param['stride']
    N, C, H, W = x.shape
    H1 = (H - Hp) // S + 1
    W1 = (W - Wp) // S + 1
    out = np.zeros((N, C, H1, W1))
    for n in range(N):
        for c in range(C):
            for k in range(H1):
                for l in range(W1):
                    out[n, c, k ,l] = np.max(x[n, c, k * S:k * S + Hp, l * S:l * S + Wp])
    cache = (x, pool_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """

    - dout: Upstream derivatives.
    -cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive
    """

    dx, dw, db = None, None, None
    x, w, b, conv_param = cache
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    _, _, Hout, Wout = dout.shape
    pad, stride = con_param['pad'], conv_param['stride']
    pad_widths = ((0, ), (0, ), (pad, ), (pad, ))
    xpad = np.pad(x, pad_widths, 'constant')
    dxpad = np.zeros_like(xpad)
    dw = np.zeros_like(w)
    db = np.zeros_like(b)
    for n in range(N):
        for f in range(F):
            # db at index f is summing dout
            db[f] += np.sum(dout[n, f])
        for i in range(Hout):
            for j in range(Wout):
                dw[f] += xpad[n, :, i * stride: i * stride + HH, j * stride: j * stride + WW] * dout[n, f, i, j]
                dxpad[n, :, i * stride: i * stride + HH, j * stride: j * stride + WW] += w[f] * dout[n, f, i, j]
    dx = dxpad[:, :, pad:pad + H, pad:pad + W]
    return dx, dw, db
