import sys
import numpy as np
import matplotlib.pyplot as plt

labels       = [ "240x135", "480x270", "960x540", "1920x1080", "3840x2160", "7680x4320", "Media" ]
ws           = [ 240, 480, 960, 1920, 3840, 7680 ]
hs           = [ 135, 270, 540, 1080, 2160, 4320 ]

serial          = [0.157987, 0.629014, 2.515071, 10.093090, 40.648059, 165.002237, 36.5075]
parallel        = [0.001217, 0.002516, 0.009254, 0.042987,  0.164412,  0.602978,   0.1372]
serial_ghost    = [0.026404, 0.112614, 0.432801, 1.685960,  8.995003,  35.104033,  7.7261]
parallel_ghost  = [0.001065, 0.007141, 0.002282, 0.032478,  0.156047,  0.465563,   0.08501483]

def throughput(t, w, h):
    return w * h * (4 + 4 + 30 + 30*2 + 1 + 32 + 1) / t

tp_parallel = [ throughput(t, w, h) for t, w, h in zip(parallel,       ws, hs) ]
tp_ghost    = [ throughput(t, w, h) for t, w, h in zip(parallel_ghost, ws, hs) ]

def graph(program, data):
    fig, ax = plt.subplots(figsize = (8, 4))
    ax.set_title(program)
    ax.set_xticks(ticks = range(len(data)), labels = labels)
    ax.bar(np.arange(len(data)), data, width=0.5)
    plt.savefig(program)

def graph_sp(program, data):
    fig, ax = plt.subplots(figsize = (7, 4))
    ax.set_title(program)
    ax.set_xticks(ticks = range(len(data)), labels = labels[:-1])
    ax.bar(np.arange(len(data)), data, width=0.5)
    plt.savefig(program)

def graph_sp(program, data):
    fig, ax = plt.subplots(figsize = (7, 4))
    ax.set_title(program)
    ax.set_xticks(ticks = range(len(data)), labels = labels[:-1])
    ax.set_yticks([])
    ax.bar(np.arange(len(data)), data, width=0.5)
    plt.savefig(program)

graph("Seriale", serial)
graph("Serial (Ghost Area)", serial_ghost)
graph("Parallelo", parallel)
graph("Parallelo (Ghost Area)", parallel_ghost)

graph_sp("Speedup",              np.array(serial[:-1])       / parallel[:-1])
graph_sp("Speedup (Ghost Area)", np.array(serial_ghost[:-1]) / parallel_ghost[:-1])

graph_sp("Throughput", tp_parallel)
graph_sp("Throughput (Ghost Area)", tp_ghost)
