import numpy as np

def phi_transform(data: list[float], degree: int) -> list[list[float]]:
	"""
	Perform a Phi Transformation to map input features into a higher-dimensional space by generating polynomial features.

	Args:
		data (list[float]): A list of numerical values to transform.
		degree (int): The degree of the polynomial expansion.

	"""
	# Your code here
	if degree < 0:
		return []
	
	result = []

	for x in data:
		row = [1.0]
		for _ in range(degree):
			row.append(row[-1] * x)
		
		result.append(row)
	
	return result