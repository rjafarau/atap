import time
import random
import functools
import multiprocessing as mp


def timeit(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        return result, time.time() - start
    return wrapper


def mcpi_samples(n):
    """
    Compute the number of points in the unit circle out of n points.
    """
    count = 0
    for i in range(n):
        x, y = random.random(), random.random()
        if x*x + y*y <= 1:
            count += 1
    return count


@timeit
def mcpi_sequential(N):
    count = mcpi_samples(N)
    return count / N * 4


@timeit
def mcpi_parallel(N):
    procs = mp.cpu_count()
    parts = [int(N / procs)] * procs
    with mp.Pool(processes=procs) as pool:
        count = sum(pool.map(mcpi_samples, parts))
    return count / N * 4
