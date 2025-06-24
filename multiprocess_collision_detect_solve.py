import multiprocessing
import time
import random

from types import SimpleNamespace


def local_detector(queue, max_items):
    for i in range(max_items):
        item = {f"Item-{i}": []}
        # 创建一个结构体实例
        for j in range(0, random.randint(0, 5)):
            first_robot_id = random.randint(5, 9)
            if 5 < first_robot_id < 9:
                second_robot_id = random.choice([1, -1]) + first_robot_id
            elif first_robot_id == 5:
                second_robot_id = first_robot_id + 1
            else:
                second_robot_id = first_robot_id - 1
            # subitem = SimpleNamespace(first_robot=f'00{first_robot_id}',
            #                           second_robot=f'00{second_robot_id}')
            subitem = frozenset([f'00{first_robot_id}', f'00{second_robot_id}'])
            if subitem not in item[f"Item-{i}"]:
                item[f"Item-{i}"].append(subitem)
        # item = f"Item-{i}"
        time.sleep(1)
        print(f"Local Detector produced: {item}")
        queue.put(item)


def global_detector(queue, max_items):
    for i in range(max_items):
        time.sleep(5)
        item = {f"Item-{i}": []}
        # 创建一个结构体实例
        for j in range(0, random.randint(0, 5)):
            first_robot_id = random.randint(5, 9)
            if 5 < first_robot_id < 9:
                second_robot_id = random.choice([1, -1]) + first_robot_id
            elif first_robot_id == 5:
                second_robot_id = first_robot_id + 1
            else:
                second_robot_id = first_robot_id - 1
            # subitem = SimpleNamespace(first_robot=f'00{first_robot_id}',
            #                           second_robot=f'00{second_robot_id}')
            subitem = frozenset([f'00{first_robot_id}', f'00{second_robot_id}'])
            if subitem not in item[f"Item-{i}"]:
                item[f"Item-{i}"].append(subitem)
                print(f"Global Detector cancelled : {subitem}")
        # item = f"Item-{i}"
        print(f"Global Detector produced: {item}")
        # queue.put(item)


def solver(queue, num):
    while True:
        item = queue.get()  # 如果队列为空，会阻塞等待
        if item is None:  # 收到终止信号
            break
        print(f"Solver {num} got: {item}")
        subitem_list = list(item.values())[0]
        if subitem_list is None:
            continue
        robot_id_mentioned = {}
        robot_id_unlock_count = {}
        for subitem in subitem_list:
            subitem = list(subitem)
            print(f'first robot : {subitem[0]}')
            print(f'second robot : {subitem[1]}')
            for robot_id in subitem:
                if robot_id not in robot_id_mentioned:
                    robot_id_mentioned[robot_id] = 1
                else:
                    robot_id_mentioned[robot_id] += 1
            # 解锁逻辑
            index = random.randint(0, 2)
            if index == 2:
                # 模拟无法解锁
                print('unable to unlock')
                print('--------------------')
                continue
            unlock_robot_id = subitem[index]
            if unlock_robot_id not in robot_id_unlock_count:
                robot_id_unlock_count[unlock_robot_id] = 1
            else:
                robot_id_unlock_count[unlock_robot_id] += 1
            print(f'try to unlock robot : {unlock_robot_id}')
            print('--------------------')

        for unlock_robot_id in robot_id_unlock_count:
            if robot_id_unlock_count[unlock_robot_id] == robot_id_mentioned[unlock_robot_id]:
                print(f'unlock robot : {unlock_robot_id}')

        # time.sleep(random.uniform(0.5, 1.0))
        time.sleep(5)


if __name__ == "__main__":
    local_queue = multiprocessing.Queue()
    global_queue = multiprocessing.Queue()
    max_items = 10

    # 启动3个消费者
    consumers = []
    for i in range(1):
        p = multiprocessing.Process(target=solver, args=(local_queue, i))
        p.daemon = True
        p.start()
        consumers.append(p)

    # 启动生产者
    producers = []
    producer_proc1 = multiprocessing.Process(target=local_detector, args=(local_queue, max_items))
    producer_proc1.start()
    producers.append(producer_proc1)
    producer_proc2 = multiprocessing.Process(target=global_detector, args=(global_queue, max_items))
    producer_proc2.start()
    producers.append(producer_proc2)

    # 等待生产者完成
    producer_proc1.join()
    producer_proc2.join()

    # 发送终止信号给消费者
    for _ in range(3):
        local_queue.put(None)

    for c in consumers:
        c.join()
    print("All done.")
