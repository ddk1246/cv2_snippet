import asyncio
import time
from asyncio.coroutines import iscoroutinefunction


# 运行时间统计
# 本部分包含 cost_time 修饰器 ， CostTime 上下文计时器
def cost_time(func):
    def fun(*args, **kwargs):
        t = time.perf_counter()
        result = func(*args, **kwargs)
        print(f'func {func.__name__} cost time:{time.perf_counter() - t:.8f} s')
        return result

    async def func_async(*args, **kwargs):
        t = time.perf_counter()
        result = await func(*args, **kwargs)
        print(f'func {func.__name__} cost time:{time.perf_counter() - t:.8f} s')
        return result

    if iscoroutinefunction(func):
        return func_async
    else:
        return fun


class CostTime(object):
    def __init__(self):
        self.t = 0

    def __enter__(self):
        self.t = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print(f'cost time:{time.perf_counter() - self.t:.8f} s')


def test():
    print('func start')
    with CostTime():
        time.sleep(2)
        print('func end')


async def test_async():
    print('async func start')
    with CostTime():
        await asyncio.sleep(2)
        print('async func end')


# @cost_time
# def test():
#     # print('func start')
#     time.sleep(2)
#     # print('func end')
#
#
# @cost_time
# async def test_async():
#     # print('async func start')
#     await asyncio.sleep(2)
#     # print('async func end')


if __name__ == '__main__':
    test()
    asyncio.get_event_loop().run_until_complete(test_async())
