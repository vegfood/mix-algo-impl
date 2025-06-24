from collections import deque


def schedule_vehicles(relations):
    # 初始化所有车辆集合
    all_vehicles = set()
    for rel in relations:
        all_vehicles.add(rel[0])
        all_vehicles.add(rel[1])
    all_vehicles = list(all_vehicles)

    # 分离不同关系类型
    third_relations = [r for r in relations if r[2] == 'third']
    second_relations = [r for r in relations if r[2] == 'second']
    fourth_relations = [r for r in relations if r[2] == 'fourth']
    first_relations = [r for r in relations if r[2] == 'first']
    fifth_relations = [r for r in relations if r[2] == 'fifth']

    # 步骤1: 递归排除头对头车辆及其依赖链
    unmovable = set()
    for a, b, _ in third_relations:
        unmovable.add(a)
        unmovable.add(b)
    changed = True
    while changed:
        changed = False
        for a, b, _ in second_relations:
            if b in unmovable and a not in unmovable:
                unmovable.add(a)
                changed = True

    movable_vehicles = [v for v in all_vehicles if v not in unmovable]
    if not movable_vehicles:
        return [] if not unmovable else None

    # 步骤2: 构建依赖图
    graph = {v: set() for v in movable_vehicles}
    in_degree = {v: 0 for v in movable_vehicles}

    # 处理跟随关系（第二种）
    for a, b, _ in second_relations:
        if a in movable_vehicles and b in movable_vehicles:
            if b not in graph[a]:
                graph[a].add(b)
                in_degree[b] += 1

    # 处理互为跟随者（第四种）
    for a, b, _ in fourth_relations:
        if a not in movable_vehicles or b not in movable_vehicles:
            continue
        # 尝试两个方向
        original_a = graph[a].copy()
        original_b = graph[b].copy()
        # 尝试a->b
        graph[a].add(b)
        if has_cycle(graph, a):
            graph[a] = original_a
            # 尝试b->a
            graph[b].add(a)
            if has_cycle(graph, b):
                graph[b] = original_b
                return None
            else:
                in_degree[a] += 1
        else:
            in_degree[b] += 1

    # 处理互斥关系（第一种）
    for a, b, _ in first_relations:
        if a not in movable_vehicles or b not in movable_vehicles:
            continue
        # 尝试两个方向
        original_a = graph[a].copy()
        original_b = graph[b].copy()
        # 尝试a->b
        graph[a].add(b)
        if has_cycle(graph, a):
            graph[a] = original_a
            # 尝试b->a
            graph[b].add(a)
            if has_cycle(graph, b):
                graph[b] = original_b
                return None
            else:
                in_degree[a] += 1
        else:
            in_degree[b] += 1

    # 最终检测环
    if has_cycle(graph, None):
        return None

    # 生成拓扑层级
    return topological_levels(graph, in_degree)


def has_cycle(graph, start_node):
    visited = set()
    stack = set()

    def dfs(node):
        if node in stack:
            return True
        if node in visited:
            return False
        visited.add(node)
        stack.add(node)
        for neighbor in graph[node]:
            if dfs(neighbor):
                return True
        stack.remove(node)
        return False

    if start_node is not None:
        return dfs(start_node)
    else:
        for node in graph:
            if dfs(node):
                return True
        return False


def topological_levels(graph, in_degree):
    in_degree = in_degree.copy()
    queue = deque()
    levels = []

    # 初始化队列
    for node in in_degree:
        if in_degree[node] == 0:
            queue.append(node)

    while queue:
        level_size = len(queue)
        current_level = []
        for _ in range(level_size):
            node = queue.popleft()
            current_level.append(node)
            for neighbor in graph[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        levels.append(current_level)

    if sum(len(level) for level in levels) != len(graph):
        return None  # 存在环
    return levels


# 复杂依赖测试
if __name__ == "__main__":
    # 示例1: 多层链式依赖 A→B→C→D
    relations_chain = [
        ('A', 'B', 'second'),
        ('B', 'C', 'second'),
        ('C', 'D', 'second')
    ]
    print("测试链式依赖:")
    order_chain = schedule_vehicles(relations_chain)
    if order_chain:
        for i, level in enumerate(order_chain):
            print(f"层级{i + 1}: {level}")
    else:
        print("死锁")

    # 示例2: 多组独立依赖链 A→B, C→D→E
    relations_independent = [
        ('A', 'B', 'second'),
        ('C', 'D', 'second'),
        ('D', 'E', 'second')
    ]
    print("\n测试独立依赖链:")
    order_independent = schedule_vehicles(relations_independent)
    if order_independent:
        for i, level in enumerate(order_independent):
            print(f"层级{i + 1}: {level}")
    else:
        print("死锁")

    # 示例3: 混合关系（跟随、互斥、互为跟随）
    relations_mixed = [
        ('A', 'B', 'second'),  # B跟随A
        ('C', 'D', 'fourth'),  # C和D互为跟随
        ('E', 'F', 'first'),  # E和F互斥
        ('B', 'C', 'second')  # C跟随B
    ]
    print("\n测试混合关系:")
    order_mixed = schedule_vehicles(relations_mixed)
    if order_mixed:
        for i, level in enumerate(order_mixed):
            print(f"层级{i + 1}: {level}")
    else:
        print("死锁")

    # 示例4: 混合关系（跟随、互斥）
    relations_mixed = [
        ('A', 'B', 'first'),  # A和B互斥
        ('C', 'B', 'second'),  # B跟随C
        ('D', 'B', 'second'),  # B跟随D
        ('C', 'D', 'second'),  # D跟随C
    ]
    print("\n测试混合关系:")
    order_mixed = schedule_vehicles(relations_mixed)
    if order_mixed:
        for i, level in enumerate(order_mixed):
            print(f"层级{i + 1}: {level}")
    else:
        print("死锁")

    # 示例5: 第五种关系（互不相关）
    relations_independent = [
        ('A', 'B', 'fifth'),  # A和B互不相关
        ('C', 'D', 'fifth'),  # C和D互不相关
        ('A', 'C', 'second')  # C跟随A
    ]
    print("\n测试第五种关系（互不相关）:")
    order_independent = schedule_vehicles(relations_independent)
    if order_independent:
        for i, level in enumerate(order_independent):
            print(f"层级{i + 1}: {level}")
    else:
        print("死锁")