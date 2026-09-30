# Implement your function below.

def run_etl(csv_text: str) -> list[tuple[str, float]]:
	"""Run a simple ETL pipeline over CSV text with header user_id,event_type,value.

	Returns a sorted list of (user_id, total_value) for event_type == "purchase".
	"""
	# TODO: implement extract, transform, and load steps
	lines = [
		line.strip()
		for line in csv_text.splitlines()
		if line.strip()
	]

	if len(lines) <= 1:
		return []
	
	rows = lines[1:]

	totals = {}

	for row in rows:
		parts = [part.strip() for part in row.split(',')]
		if len(parts) != 3:
			continue
		
		user_id, event_type, value = parts
		if event_type != 'purchase':
			continue
		
		try:
			value = float(value)
		except ValueError:
			continue
		
		totals[user_id] = totals.get(user_id, 0.0) + value
	
	result = sorted(totals.items())
	return result