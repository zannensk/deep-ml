import numpy as np

def distillation_loss(
	student_logits: np.ndarray,
	teacher_logits: np.ndarray,
	temperature: float = 1.0
) -> float:
	"""
	Compute knowledge distillation loss.
	
	L = T^2 * KL(softmax(teacher/T) || softmax(student/T))
	
	Args:
		student_logits: Logits from student model
		teacher_logits: Logits from teacher model
		temperature: Softmax temperature
		
	Returns:
		Distillation loss value
	"""
	# Your code here
	def log_softmax(x):
		x = np.asarray(x, dtype=float) / temperature
		x = x - np.max(x, axis=-1, keepdims=True)
		return x - np.log(
			np.sum(np.exp(x), axis=-1, keepdims=True)
		)
	
	log_t = log_softmax(teacher_logits)
	log_s = log_softmax(student_logits)

	p_t = np.exp(log_t)
	kl = np.sum(
		p_t * (log_t - log_s),
		axis=-1
	)
	loss = temperature ** 2 * np.mean(kl)

	return round(float(loss), 4)




