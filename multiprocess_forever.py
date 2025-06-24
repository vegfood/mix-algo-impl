import multiprocessing
import time
import random


def producer(queue, max_items):
    for i in range(max_items):
        item = f"Item-{i}"
        time.sleep(1)
        print(f"Producer produced: {item}")
        queue.put(item)

def consumer(queue, num):
    while True:
        item = queue.get()  # 如果队列为空，会阻塞等待
        if item is None:  # 收到终止信号
            break
        print(f"Consumer {num} got: {item}")
        # time.sleep(random.uniform(0.5, 1.0))
        time.sleep(5)


if __name__ == "__main__":
    queue = multiprocessing.Queue()
    max_items = 10

    # 启动3个消费者
    consumers = []
    for i in range(3):
        p = multiprocessing.Process(target=consumer, args=(queue, i))
        p.daemon = True
        p.start()
        consumers.append(p)

    # 启动生产者
    producer_proc = multiprocessing.Process(target=producer, args=(queue, max_items))
    producer_proc.start()

    # 等待生产者完成
    producer_proc.join()

    # 发送终止信号给消费者
    for _ in range(3):
        queue.put(None)

    for c in consumers:
        c.join()
    print("All done.")
