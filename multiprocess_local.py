import multiprocessing
import time

def producer(condition):
    with condition:
        print("Producer is producing")
        time.sleep(2)
        print("Producer has finished producing")
        condition.notify_all()


def consumer(condition, num):
    with condition:
        print(f"Consumer {num} is waiting")
        condition.wait()

    # 锁已释放，可以并行消费
    print(f"Consumer {num} has started consuming")
    time.sleep(5)
    print(f"Consumer {num} has finished consuming")

if __name__ == "__main__":
    condition = multiprocessing.Condition()
    processes = []

    for i in range(3):
        p = multiprocessing.Process(target=consumer, args=(condition, i))
        processes.append(p)
        p.start()

    producer_process = multiprocessing.Process(target=producer, args=(condition,))
    producer_process.start()

    for p in processes:
        p.join()

    producer_process.join()