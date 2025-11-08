import math
from typing import Union, Sequence, List
import numpy as np

ArrayLike = Union[np.ndarray, Sequence[Sequence[int]], Sequence[Sequence[float]]]


def _min_unsigned_dtype(max_val: int):
    if max_val <= 0xFF:
        return np.uint8
    if max_val <= 0xFFFF:
        return np.uint16
    if max_val <= 0xFFFFFFFF:
        return np.uint32
    return np.uint64


class codec:
    """
    Ultra-fast lossless FP16 <-> integer-grids codec (pure NumPy).
    - Stores D base-K digits (K = max_val - min_val + 1) into (D, rows, cols).
    - Fast paths for K = 2^b (bit-slicing); generic vectorized divmod + Horner decode.
    """

    def __init__(self, rows: int, cols: int, min_val: int, max_val: int):
        if rows <= 0 or cols <= 0:
            raise ValueError("rows/cols must be > 0")
        if max_val < min_val:
            raise ValueError("max_val must be >= min_val")
        if not isinstance(min_val, int) or not isinstance(max_val, int):
            raise TypeError("min_val/max_val must be ints")

        self.rows = int(rows)
        self.cols = int(cols)
        self.min_val = int(min_val)
        self.max_val = int(max_val)

        K = self.max_val - self.min_val + 1
        if K < 2:
            raise ValueError("range must contain >= 2 distinct integers")
        self.K = K

        # exact grids needed: ceil(16 / log2(K))
        self.D = math.ceil(16 / math.log2(K))

        # power-of-two fast path?
        self._is_pow2 = K & (K - 1) == 0
        self._b = (K.bit_length() - 1) if self._is_pow2 else None  # log2(K)

        # compact dtype for grids
        if self.min_val >= 0:
            self.grid_dtype = _min_unsigned_dtype(self.max_val)
        else:
            if self.min_val >= -(2**7) and self.max_val <= 2**7 - 1:
                self.grid_dtype = np.int8
            elif self.min_val >= -(2**15) and self.max_val <= 2**15 - 1:
                self.grid_dtype = np.int16
            elif self.min_val >= -(2**31) and self.max_val <= 2**31 - 1:
                self.grid_dtype = np.int32
            else:
                self.grid_dtype = np.int64

    def grids_needed(self) -> int:
        return self.D

    # ---- helpers ----
    def _to_u16(self, arr: np.ndarray) -> np.ndarray:
        if arr.shape != (self.rows, self.cols):
            raise ValueError(f"tensor must be shape ({self.rows}, {self.cols})")
        if arr.dtype == np.float16:
            return arr.view(np.uint16)
        if arr.dtype == np.uint16:
            return arr
        raise TypeError("tensor must be dtype float16 or uint16")

    # ---- encode ----
    def encode(self, tensor: ArrayLike) -> np.ndarray:
        """
        Returns a contiguous array of shape (D, rows, cols) with dtype chosen for range.
        Grid 0 is least-significant digit.
        """
        u16 = self._to_u16(np.asarray(tensor)).astype(np.uint32, copy=False)
        out = np.empty((self.D, self.rows, self.cols), dtype=self.grid_dtype)
        if self._is_pow2:
            self._encode_pow2(u16, out)
        else:
            self._encode_generic(u16, out)
        return out

    def _encode_pow2(self, u32: np.ndarray, out: np.ndarray):
        b = self._b
        mask = (1 << b) - 1
        v = u32
        for d in range(self.D):
            digit = (v & mask).astype(np.uint32)
            out[d, :, :] = (digit + self.min_val).astype(out.dtype, copy=False)
            v = v >> b

    def _encode_generic(self, u32: np.ndarray, out: np.ndarray):
        v = u32.copy()
        K = np.uint32(self.K)
        mv = np.int64(self.min_val)
        for d in range(self.D):
            # np.divmod returns (quotient, remainder); we want remainder as the digit
            q, rem = np.divmod(v, K)
            out[d, :, :] = (rem.astype(np.int64) + mv).astype(out.dtype, copy=False)
            v = q

    # ---- decode ----
    def decode(self, grids: np.ndarray) -> np.ndarray:
        """
        grids: shape (D, rows, cols)
        Returns float16 tensor of shape (rows, cols).
        """
        g = np.asarray(grids)
        if g.shape != (self.D, self.rows, self.cols):
            raise ValueError(
                f"grids must have shape ({self.D}, {self.rows}, {self.cols})"
            )
        if (g < self.min_val).any() or (g > self.max_val).any():
            raise ValueError("grid values out of range")

        if self._is_pow2:
            u16 = self._decode_pow2(g)
        else:
            u16 = self._decode_horner(g)
        return u16.view(np.float16)

    def _decode_pow2(self, g: np.ndarray) -> np.ndarray:
        b = self._b
        acc = np.zeros((self.rows, self.cols), dtype=np.uint32)
        shift = 0
        for d in range(self.D):
            digit = (g[d].astype(np.int64) - self.min_val).astype(np.uint32)
            acc |= digit << shift
            shift += b
        return acc.astype(np.uint16, copy=False)

    def _decode_horner(self, g: np.ndarray) -> np.ndarray:
        K = np.uint32(self.K)
        acc = np.zeros((self.rows, self.cols), dtype=np.uint32)
        for d in range(self.D - 1, -1, -1):
            acc *= K
            digit = (g[d].astype(np.int64) - self.min_val).astype(np.uint32)
            acc += digit
        if (acc > 0xFFFF).any():
            raise ValueError("decoded value out of uint16 range (corrupt input?)")
        return acc.astype(np.uint16, copy=False)

    # ---- integrity ----
    def roundtrip_equal(self, tensor: ArrayLike) -> bool:
        u0 = self._to_u16(np.asarray(tensor))
        grids = self.encode(u0)
        back = self.decode(grids).view(np.uint16)
        return bool(np.array_equal(u0, back))


# ---------------- Example ----------------
if __name__ == "__main__":
    # Example: 4x3 tensor, store digits in range [0, 255] -> K=256 -> D=2 grids
    rows, cols = 4, 3
    c = codec(rows, cols, min_val=0, max_val=15)
    print("K =", c.K, "grids_needed =", c.grids_needed())  # K=256, D=2

    x = (np.arange(rows * cols, dtype=np.float16).reshape(rows, cols) / 7.0).astype(
        np.float16
    )

    grids = c.encode(x)
    x2 = c.decode(grids)

    print(grids)
    print(x2)

    # Bit-exact check:
    ok = c.roundtrip_equal(x)
    print("roundtrip ok:", ok)
