import heapq

def analyze_ml_pipeline(tasks: list) -> dict:
    """
    Analyze an ML pipeline DAG for scheduling and critical path.
    
    Args:
        tasks: list of task dicts with:
            - 'id': task identifier (str)
            - 'duration': task duration in minutes (int)
            - 'dependencies': list of task IDs this task depends on
    
    Returns:
        dict with:
            - 'execution_order': topologically sorted list of task IDs
            - 'earliest_start': dict mapping task ID to earliest start time
            - 'earliest_finish': dict mapping task ID to earliest finish time
            - 'latest_start': dict mapping task ID to latest start time
            - 'latest_finish': dict mapping task ID to latest finish time
            - 'slack': dict mapping task ID to slack time
            - 'critical_path': list of task IDs on critical path (in execution order)
            - 'makespan': total time to complete pipeline
    """
    if not tasks:
        return {
            'execution_order': [],
            'earliest_start': {},
            'earliest_finish': {},
            'latest_start': {},
            'latest_finish': {},
            'slack': {},
            'critical_path': [],
            'makespan': 0,
        }

    dependencies = {}
    successors = {}
    duration = {}
    indegree = {}

    for task in tasks:
        task_id = task['id']
        duration[task_id] = task['duration']
        dependencies[task_id] = task['dependencies']
        indegree[task_id] = len(task['dependencies'])
        successors[task_id] = []
    
    for task in tasks:
        task_id = task['id']
        for dep in dependencies[task_id]:
            successors[dep].append(task_id)
        
    heap = []
    for task_id in indegree:
        if indegree[task_id] == 0:
            heapq.heappush(heap, task_id)
    
    execution_order = []
    while heap:
        current = heapq.heappop(heap)
        execution_order.append(current)
        for nxt in successors[current]:
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                heapq.heappush(heap, nxt)

    earliest_start = {}
    earliest_finish = {}
    for task_id in execution_order:
        if not dependencies[task_id]:
            earliest_start[task_id] = 0
        else:
            earliest_start[task_id] = max(
                earliest_finish[nxt]
                for nxt in dependencies[task_id]
            )
        earliest_finish[task_id] = (
            earliest_start[task_id]
            + duration[task_id]
        )
    
    makespan = max(earliest_finish.values())

    latest_start = {
        task_id: 0
        for task_id in execution_order
    }
    latest_finish = {
        task_id: 0
        for task_id in execution_order
    }
    
    for task_id in reversed(execution_order):
        if not successors[task_id]:
            latest_finish[task_id] = makespan
        else:
            latest_finish[task_id] = min(
                latest_start[nxt]
                for nxt in successors[task_id]
            )
        
        latest_start[task_id] = (
            latest_finish[task_id]
            - duration[task_id]
        )
    
    slack = {}
    for task_id in execution_order:
        slack[task_id] = (
            latest_start[task_id]
            -
            earliest_start[task_id]
        )
    
    critical_path = [
        task_id
        for task_id in execution_order
        if slack[task_id] == 0
    ]

    return {
        "execution_order": execution_order,
        "earliest_start": earliest_start,
        "earliest_finish": earliest_finish,
        "latest_start": latest_start,
        "latest_finish": latest_finish,
        "slack": slack,
        "critical_path": critical_path,
        "makespan": makespan,
    }
