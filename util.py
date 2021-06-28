import time


def timer(f):
    """ 耗时统计 """
    def pack():
        start = time.time()
        f()
        print("耗时:", time.time() - start)
    return pack


