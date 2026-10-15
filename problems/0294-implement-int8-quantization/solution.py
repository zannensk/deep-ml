import numpy as np

def int8_quantize(x: list[float]) -> dict:
	"""
	Perform symmetric INT8 quantization on a floating-point array.
	
	Args:
		x: Input list of floating-point values
		
	Returns:
		Dictionary with 'quantized', 'scale', and 'dequantized' keys
	"""
	# Your code here
	if not x:
		return {}
	
	arr = np.asarray(x, dtype=np.float32)
	max_abs = np.max(np.abs(arr))
	scale = max_abs / 127 if max_abs > 0 else 1.0

	q = np.clip(
		np.round(arr / scale),
		-127, 127
	).astype(np.int8)

	dequantized = q.astype(float) * scale

	return {
        "quantized": q.tolist(),
        "scale": round(float(scale), 6),
        "dequantized": np.round(dequantized, 4).tolist()
    }
