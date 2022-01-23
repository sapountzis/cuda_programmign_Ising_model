import os

import numpy as np
import pandas as pd
from numba import cuda, njit, prange, int8
from time import perf_counter
import math


np.random.seed(1)
n = 2048
iters = 10
# generate data
data_orig = 2 * np.random.randint(low=0, high=2, size=(n, n), dtype=np.int8) - 1
data_per_thread_x, data_per_thread_y = (16, 16)
grid_size_v2 = (int(math.ceil(data_orig.shape[0] / data_per_thread_x)),
                int(math.ceil(data_orig.shape[1] / data_per_thread_y)))
threads_per_block_x, threads_per_block_y = (8, 8)
data_per_block_x, data_per_block_y = (data_per_thread_x * threads_per_block_x,
                                      data_per_thread_y * threads_per_block_y)
grid_size_v3 = (int(math.ceil(data_orig.shape[0] / data_per_block_x)),
                int(math.ceil(data_orig.shape[1] / data_per_block_y)))


def sequential(data, res, iters):
    for iteration in range(iters):
        data_x_max = data.shape[0]
        data_y_max = data.shape[1]
        for x in range(data.shape[0]):
            for y in range(data.shape[1]):
                sum_cell = data[x, y]
                for k in range(x-1, x+2, 2):
                    x_idx = (k % data_x_max + data_x_max) % data_x_max
                    sum_cell += data[x_idx, y]

                for k in range(y-1, y+2, 2):
                    y_idx = (k % data_y_max + data_y_max) % data_y_max
                    sum_cell += data[x, y_idx]

                res[x, y] = np.sign(sum_cell)

        if iteration == (iters - 1):
            return res
        data, res = res, data


@njit('void(i1[:, :], i1[:, :])', fastmath=True, parallel=True)
def sequential_optimized(data, res):
    data_x_max = data.shape[0]
    data_y_max = data.shape[1]
    for x in prange(data.shape[0]):
        for y in range(data.shape[1]):
            res[x, y] = 0
            sum_cell = 0
            for k in range(x - 1, x + 2, 2):
                x_idx = (k % data_x_max + data_x_max) % data_x_max
                sum_cell += data[x_idx, y]

            for k in range(y - 1, y + 2, 2):
                y_idx = (k % data_y_max + data_y_max) % data_y_max
                sum_cell += data[x, y_idx]

            sum_cell += data[x, y]

            res[x, y] = int(math.copysign(1, sum_cell))


def sequential_optimized_wrapper(data, res, iters):
    for iteration in range(iters):
        sequential_optimized(data, res)

        if iteration == (iters - 1):
            return res

        data, res = res, data


def vectorized(data):
    res = data.copy()
    res += np.roll(data, shift=1, axis=0)
    res += np.roll(data, shift=-1, axis=0)
    res += np.roll(data, shift=1, axis=1)
    res += np.roll(data, shift=-1, axis=1)
    res = np.sign(res)
    return res


def vectorized_wrapper(data, iters):
    for iteration in range(iters):
        res = vectorized(data)

        if iteration == (iters - 1):
            return res

        data, res = res, data


@cuda.jit('void(i1[:, :], i1[:, :])', fastmath=True)
def cuda_v1(data, res):
    x, y = cuda.grid(2)
    data_x_max = data.shape[0]
    data_y_max = data.shape[1]

    sum_cell = 0
    for k in range(x - 1, x + 2, 2):
        x_idx = (k % data_x_max + data_x_max) % data_x_max
        sum_cell += data[x_idx, y]

    for k in range(y - 1, y + 2, 2):
        y_idx = (k % data_y_max + data_y_max) % data_y_max
        sum_cell += data[x, y_idx]

    sum_cell += data[x, y]

    res[x, y] = math.copysign(1, sum_cell)


def cuda_v1_wrapper(data, res, iters):
    d_data = cuda.to_device(data)
    d_res = cuda.to_device(res)

    for iteration in range(iters):
        cuda_v1[(data.shape[0], data.shape[1]), (1, 1)](d_data, d_res)

        if iteration == (iters - 1):
            return d_res.copy_to_host()

        d_data, d_res = d_res, d_data


@cuda.jit('void(i1[:, :], i1[:, :])', fastmath=True)
def cuda_v2(data, res):
    data_x_max = data.shape[0]
    data_y_max = data.shape[1]

    x_step, y_step = (data_per_thread_x, data_per_thread_y)

    x_block = cuda.blockIdx.x
    y_block = cuda.blockIdx.y

    x_start = x_block * x_step
    y_start = y_block * y_step
    x_end = min((x_block + 1) * x_step, data_x_max)
    y_end = min((y_block + 1) * y_step, data_y_max)

    shared_data = cuda.shared.array(shape=(data_per_thread_x, data_per_thread_y), dtype=int8)

    for x in range(x_start, x_end):
        for y in range(y_start, y_end):
            x_offset = x - x_start
            y_offset = y - y_start
            shared_data[x_offset, y_offset] = data[x, y]

    for x in range(x_start, x_end):
        for y in range(y_start, y_end):
            x_offset = x - x_start
            y_offset = y - y_start

            sum_cell = shared_data[x_offset, y_offset]

            if (x == x_start) or (x == x_end - 1) or (y == y_start) or (y == y_end - 1):

                for k in range(x - 1, x + 2, 2):
                    if (k < x_start) or (k >= x_end):
                        x_idx = (k % data_x_max + data_x_max) % data_x_max
                        sum_cell += data[x_idx, y]
                    else:
                        sum_cell += shared_data[k-x_start, y_offset]

                for k in range(y - 1, y + 2, 2):
                    if (k < y_start) or (k >= y_end):
                        y_idx = (k % data_y_max + data_y_max) % data_y_max
                        sum_cell += data[x, y_idx]
                    else:
                        sum_cell += shared_data[x_offset, k-y_start]

            else:
                for k in range(x_offset - 1, x_offset + 2, 2):
                    sum_cell += shared_data[k, y_offset]

                for k in range(y_offset - 1, y_offset + 2, 2):
                    sum_cell += shared_data[x_offset, k]

            res[x, y] = math.copysign(1, sum_cell)


def cuda_v2_wrapper(data, res, iters, grid_size):
    d_data = cuda.to_device(data)
    d_res = cuda.to_device(res)

    for iteration in range(iters):
        cuda_v2[grid_size, 1](d_data, d_res)

        if iteration == (iters - 1):
            return d_res.copy_to_host()

        d_data, d_res = d_res, d_data


@cuda.jit('void(i1[:, :], i1[:, :])', fastmath=True)
def cuda_v3(data, res):
    data_x_max = data.shape[0]
    data_y_max = data.shape[1]

    x_block_step, y_block_step = (data_per_block_x, data_per_block_y)

    x_block = cuda.blockIdx.x
    y_block = cuda.blockIdx.y

    x_start = x_block * x_block_step
    y_start = y_block * y_block_step
    x_end = min((x_block + 1) * x_block_step, data_x_max)
    y_end = min((y_block + 1) * y_block_step, data_y_max)

    x_thread = cuda.threadIdx.x
    y_thread = cuda.threadIdx.y

    x_threads = cuda.blockDim.x
    y_threads = cuda.blockDim.y

    x_thread_step = math.ceil(x_block_step / x_threads)
    y_thread_step = math.ceil(y_block_step / y_threads)

    x_thread_start = x_start + x_thread * x_thread_step
    y_thread_start = y_start + y_thread * y_thread_step
    x_thread_end = min(x_start + (x_thread + 1) * x_thread_step, data_x_max)
    y_thread_end = min(y_start + (y_thread + 1) * y_thread_step, data_y_max)

    shared_data = cuda.shared.array(shape=(x_block_step, y_block_step), dtype=int8)

    for x in range(x_thread_start, x_thread_end):
        for y in range(y_thread_start, y_thread_end):
            x_offset = x - x_start
            y_offset = y - y_start
            shared_data[x_offset, y_offset] = data[x, y]

    cuda.syncthreads()

    for x in range(x_thread_start, x_thread_end):
        for y in range(y_thread_start, y_thread_end):
            x_offset = x - x_start
            y_offset = y - y_start

            sum_cell = shared_data[x_offset, y_offset]

            if (x == x_start) or (x == x_end - 1) or (y == y_start) or (y == y_end - 1):

                for k in range(x - 1, x + 2, 2):
                    if (k < x_start) or (k >= x_end):
                        x_idx = (k % data_x_max + data_x_max) % data_x_max
                        sum_cell += data[x_idx, y]
                    else:
                        sum_cell += shared_data[k-x_start, y_offset]

                for k in range(y - 1, y + 2, 2):
                    if (k < y_start) or (k >= y_end):
                        y_idx = (k % data_y_max + data_y_max) % data_y_max
                        sum_cell += data[x, y_idx]
                    else:
                        sum_cell += shared_data[x_offset, k-y_start]

            else:
                for k in range(x_offset - 1, x_offset + 2, 2):
                    sum_cell += shared_data[k, y_offset]

                for k in range(y_offset - 1, y_offset + 2, 2):
                    sum_cell += shared_data[x_offset, k]

            res[x, y] = math.copysign(1, sum_cell)


def cuda_v3_wrapper(data, res, iters, grid_size):
    d_data = cuda.to_device(data)
    d_res = cuda.to_device(res)
    for iteration in range(iters):
        cuda_v3[grid_size, (threads_per_block_x, threads_per_block_y)](d_data, d_res)

        if iteration == (iters - 1):
            return d_res.copy_to_host()

        d_data, d_res = d_res, d_data


if __name__ == '__main__':
    if 'stats.csv' not in os.listdir():
        stats = pd.DataFrame(columns=['n', 'iterations', 'implementation', 'avg_loop_time'])
    else:
        stats = pd.read_csv('stats.csv')

    # sequential
    data = np.copy(data_orig)
    seq_res = np.copy(data)
    print(f'Starting Sequential - plain python, with params: data=({n}, {n})')
    start = perf_counter()
    seq_res = sequential(data, seq_res, iters)
    end = perf_counter()
    seq_time = end - start
    print(f'Sequential time (n={n}, iterations={iters}): {seq_time} sec total, {seq_time / iters} sec per iteration\n')
    stats = stats.append({'n': n, 'iterations': iters, 'implementation': 'seq_v0_python', 'avg_loop_time': seq_time / iters}, ignore_index=True)

    # sequential optimized
    data = np.copy(data_orig)
    seq_opt_res = np.empty_like(data, dtype=np.int8)
    print(f'Sequential Numba Optimized, with params: data=({n}, {n})')
    start = perf_counter()
    seq_opt_res = sequential_optimized_wrapper(data, seq_opt_res, iters)
    end = perf_counter()
    seq_opt_time = end - start
    print(f'Numba Optimized Sequential time (iterations={iters}): {seq_opt_time} sec total, {seq_opt_time / iters} sec per iteration\n')
    stats = stats.append({'n': n, 'iterations': iters, 'implementation': 'seq_v0_numba', 'avg_loop_time': seq_opt_time / iters}, ignore_index=True)

    # vectorized
    data = np.copy(data_orig)
    print(f'Starting Sequential Numpy Vectorized, with params: data=({n}, {n})')
    start = perf_counter()
    vec_res = vectorized_wrapper(data, iters)
    end = perf_counter()
    vec_time = end - start
    print(f'Vectorized time (iterations={iters}): {end-start} sec total, {vec_time / iters} sec per iteration\n')
    stats = stats.append({'n': n, 'iterations': iters, 'implementation': 'seq_v0_numpy_vec', 'avg_loop_time': vec_time / iters}, ignore_index=True)

    # cuda v1
    data = np.copy(data_orig)
    cu_v1_res = np.empty_like(data, dtype=np.int8)
    print(f'Starting CUDA V1, with params:\ndata=({n}, {n}), grid_size=({n, n}), threads_per_block=(1, 1)')
    start = perf_counter()
    cu_v1_res = cuda_v1_wrapper(data, cu_v1_res, iters)
    end = perf_counter()
    cu_v1_time = end - start
    print(f'Cuda v1 time (iterations={iters}): {cu_v1_time} sec total, {cu_v1_time / iters} sec per iteration\n')
    stats = stats.append({'n': n, 'iterations': iters, 'implementation': 'cuda_v1', 'avg_loop_time': cu_v1_time / iters}, ignore_index=True)

    # cuda v2
    data = np.copy(data_orig)
    cu_v2_res = np.empty_like(data, dtype=np.int8)
    print(f'Starting CUDA V2, with params:\ndata=({n}, {n}), grid_size=({grid_size_v2}), threads_per_block=(1, 1), data_per_thread=({data_per_thread_x, data_per_thread_y})')
    start = perf_counter()
    cu_v2_res = cuda_v2_wrapper(data, cu_v2_res, iters, grid_size_v2)
    end = perf_counter()
    cu_v2_time = end - start
    print(f'Cuda v2 time (iterations={iters}): {cu_v2_time} sec total, {cu_v2_time / iters} sec per iteration\n')
    stats = stats.append({'n': n, 'iterations': iters, 'implementation': f'cuda_v2_t({data_per_thread_x}x{data_per_thread_y})', 'avg_loop_time': cu_v2_time / iters}, ignore_index=True)

    # cuda v3
    data = np.copy(data_orig)
    cu_v3_res = np.empty_like(data, dtype=np.int8)
    print(f'Starting CUDA V3, with params:\ndata=({n}, {n}), grid_size=({grid_size_v3}), threads_per_block=({threads_per_block_x, threads_per_block_y}), data_per_block=({data_per_block_x, data_per_block_y}), data_per_thread=({data_per_thread_x, data_per_thread_y})')
    start = perf_counter()
    cu_v3_res = cuda_v3_wrapper(data, cu_v3_res, iters, grid_size_v3)
    end = perf_counter()
    cu_v3_time = end - start
    print(f'Cuda v3 time (iterations={iters}): {cu_v3_time} sec total, {cu_v3_time / iters} sec per iteration\n')
    stats = stats.append({'n': n, 'iterations': iters, 'implementation': f'cuda_v3_b({data_per_block_x}x{data_per_block_y})_t({data_per_thread_x}x{data_per_thread_y})', 'avg_loop_time': cu_v3_time / iters}, ignore_index=True)

    assert np.abs(seq_res - seq_opt_res).mean() == 0
    assert np.abs(vec_res - seq_opt_res).mean() == 0
    assert np.abs(vec_res - cu_v1_res).mean() == 0
    assert np.abs(cu_v2_res - cu_v1_res).mean() == 0
    assert np.abs(cu_v2_res - cu_v3_res).mean() == 0

    stats.to_csv('stats.csv', index=False)






