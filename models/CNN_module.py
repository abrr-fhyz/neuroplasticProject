import cupy as cp

# ──────────────────────────────────────────────#
#                     UTIL                      #
# ──────────────────────────────────────────────#

def im2col(X, fh, fw, stride=1, pad=0):
    N, C, H, W = X.shape
    out_h = (H + 2*pad - fh) // stride + 1
    out_w = (W + 2*pad - fw) // stride + 1

    Xp = cp.pad(X, ((0,0),(0,0),(pad,pad),(pad,pad)), mode='constant')
    shape = (N, C, fh, fw, out_h, out_w)
    strides = (
        Xp.strides[0],
        Xp.strides[1],
        Xp.strides[2],
        Xp.strides[3],
        Xp.strides[2] * stride,
        Xp.strides[3] * stride,
    )
    col6d = cp.lib.stride_tricks.as_strided(Xp, shape=shape, strides=strides)
    col6d = cp.ascontiguousarray(col6d)                 
    col = col6d.transpose(0, 4, 5, 1, 2, 3)             
    col = col.reshape(N, out_h * out_w, C * fh * fw)     
    return col, out_h, out_w

def col2im(col, X_shape, fh, fw, stride=1, pad=0):
    N, C, H, W = X_shape
    out_h = (H + 2*pad - fh) // stride + 1
    out_w = (W + 2*pad - fw) // stride + 1

    col = col.reshape(N, out_h, out_w, C, fh, fw).transpose(0, 3, 4, 5, 1, 2)
    Xp = cp.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1), dtype=col.dtype)

    for j in range(fh):
        j_max = j + stride * out_h
        for k in range(fw):
            k_max = k + stride * out_w
            Xp[:, :, j:j_max:stride, k:k_max:stride] += col[:, :, j, k, :, :]

    return Xp[:, :, pad:pad+H, pad:pad+W]


# ──────────────────────────────────────────────#
#               Convolution layer               #
# ──────────────────────────────────────────────#

class ConvLayer:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, pad=0):
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.fh = self.fw = kernel_size
        self.stride = stride
        self.pad    = pad
        fan_in = in_channels * kernel_size * kernel_size
        self.W = cp.random.randn(out_channels, in_channels * kernel_size * kernel_size) * cp.sqrt(2.0 / fan_in)
        self.b = cp.zeros((1, out_channels))
        self._col  = None
        self._X_shape = None

    @staticmethod
    def relu(x):
        return cp.maximum(0, x)

    @staticmethod
    def relu_dydx(x):
        return (x > 0).astype(x.dtype)

    def forward(self, X):
        self._X_shape = X.shape
        N = X.shape[0]

        col, out_h, out_w = im2col(X, self.fh, self.fw, self.stride, self.pad)
        self._col   = col                          
        self._out_h = out_h
        self._out_w = out_w
        z = col @ self.W.T + self.b              
        self._z = z

        a = ConvLayer.relu(z)
        return a.transpose(0, 2, 1).reshape(N, self.out_channels, out_h, out_w)

    def backward(self, dA, lr):
        N = dA.shape[0]
        dA_col = dA.reshape(N, self.out_channels, -1).transpose(0, 2, 1)
        dZ = dA_col * ConvLayer.relu_dydx(self._z)  
        dW = cp.tensordot(dZ, self._col, axes=([0,1],[0,1]))   
        db = dZ.sum(axis=(0,1)).reshape(1, -1)                 
        dcol = dZ @ self.W                                     
        dX = col2im(dcol, self._X_shape, self.fh, self.fw, self.stride, self.pad)
        self.W -= lr * dW
        self.b -= lr * db
        return dX


# ──────────────────────────────────────────────#
#               MaxPooling layer                #
# ──────────────────────────────────────────────#

class MaxPoolLayer:
    def __init__(self, pool_size=2, stride=None):
        self.pool_size = pool_size
        self.stride    = stride if stride is not None else pool_size

    def forward(self, X):
        N, C, H, W = X.shape
        ph = pw = self.pool_size
        s  = self.stride
        out_h = (H - ph) // s + 1
        out_w = (W - pw) // s + 1
        Xr = X.reshape(N * C, 1, H, W)
        col, _, _ = im2col(Xr, ph, pw, stride=s, pad=0)
        self._col_shape = col.shape
        self._max_idx   = cp.argmax(col, axis=2)         
        pool_out = col.max(axis=2)                       
        self._N, self._C, self._H, self._W = N, C, H, W
        self._out_h, self._out_w = out_h, out_w
        return pool_out.reshape(N, C, out_h, out_w)

    def backward(self, dA):
        N, C, H, W = self._N, self._C, self._H, self._W
        ph = pw = self.pool_size
        s  = self.stride
        dA_flat = dA.reshape(N * C, self._out_h * self._out_w) 
        dcol = cp.zeros(self._col_shape, dtype=dA.dtype)        
        idx  = self._max_idx                                      
        NC, S = idx.shape
        row_idx = cp.arange(NC)[:, cp.newaxis]                  
        col_idx = cp.arange(S) [cp.newaxis, :]                   
        dcol[row_idx, col_idx, idx] = dA_flat                    
        dX = col2im(dcol, (N * C, 1, H, W), ph, pw, stride=s, pad=0)
        return dX.reshape(N, C, H, W)


# ──────────────────────────────────────────────#
#                   CNN Module                  #
# ──────────────────────────────────────────────#

class ConvPoolModule:
    def __init__(self, conv_configs):
        self.conv_layers = []
        self.pool_layers = []

        for cfg in conv_configs:
            self.conv_layers.append(ConvLayer(
                in_channels  = cfg["in_channels"],
                out_channels = cfg["out_channels"],
                kernel_size  = cfg["kernel_size"],
                stride       = cfg.get("stride", 1),
                pad          = cfg.get("pad",    0),
            ))
            self.pool_layers.append(MaxPoolLayer(
                pool_size = cfg.get("pool_size",   2),
                stride    = cfg.get("pool_stride", None),
            ))

    def output_size(self, input_h, input_w):
        h, w = input_h, input_w
        c = self.conv_layers[0].in_channels
        for conv, pool in zip(self.conv_layers, self.pool_layers):
            h = (h + 2*conv.pad - conv.fh) // conv.stride + 1
            w = (w + 2*conv.pad - conv.fw) // conv.stride + 1
            h = (h - pool.pool_size) // pool.stride + 1
            w = (w - pool.pool_size) // pool.stride + 1
            c = conv.out_channels
        return c * h * w

    def forward(self, X):
        out = X
        for conv, pool in zip(self.conv_layers, self.pool_layers):
            out = conv.forward(out)   
            out = pool.forward(out)   
        self._last_shape = out.shape
        return out.reshape(out.shape[0], -1)   

    def backward(self, d_flat, lr):
        
        dout = d_flat.reshape(self._last_shape)
        for conv, pool in zip(reversed(self.conv_layers), reversed(self.pool_layers)):
            dout = pool.backward(dout)
            dout = conv.backward(dout, lr)
        return dout

    def save(self, path="artifacts/convpool.npz"):
        arrays = {}
        for i, (conv, pool) in enumerate(zip(self.conv_layers, self.pool_layers)):
            arrays[f"conv{i}_W"]    = cp.asnumpy(conv.W)
            arrays[f"conv{i}_b"]    = cp.asnumpy(conv.b)
            arrays[f"pool{i}_size"] = cp.array(pool.pool_size)
        import numpy as np_cpu
        np_cpu.savez(path, **arrays)
        print(f"ConvPoolModule saved to {path}")

    def load(self, path="artifacts/convpool.npz"):
        import numpy as np_cpu
        data = np_cpu.load(path)
        for i, (conv, pool) in enumerate(zip(self.conv_layers, self.pool_layers)):
            conv.W = cp.asarray(data[f"conv{i}_W"])
            conv.b = cp.asarray(data[f"conv{i}_b"])
        print(f"ConvPoolModule loaded from {path}")