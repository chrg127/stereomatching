import sys
import numpy as np
import matplotlib.pyplot as plt

if len(sys.argv) < 2:
    sys.exit(1)

def split_values(s):
    return [ float(x) for x in list(filter(lambda x: x != '', s.strip().split(","))) ]

serial         = split_values(sys.argv[1])
serial_ghost   = split_values(sys.argv[2])
parallel       = split_values(sys.argv[3])
parallel_ghost = split_values(sys.argv[4])

labels       = [ "240x135", "480x270", "960x540", "1920x1080", "3840x2160", "7680x4320" ]
ws           = [ 240, 480, 960, 1920, 3840, 7680 ]
hs           = [ 135, 270, 540, 1080, 2160, 4320 ]

def graph(program, filename, data, xlabel, ylabel):
    lbs = labels[0:len(data)-1]
    lbs.append("Media")
    fig, ax = plt.subplots(figsize = (len(data)+2, 4))
    ax.set_title(program)
    ax.set_xticks(ticks = range(len(data)), labels = lbs)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.bar(np.arange(len(data)), data, width=0.5)
    # plt.show()
    plt.savefig(filename, bbox_inches="tight")

def graph_sp(program, filename, data, xlabel, ylabel):
    fig, ax = plt.subplots(figsize = (len(data)+2, 4))
    ax.set_title(program)
    ax.set_xticks(ticks = range(len(data)), labels = labels[0:len(data)])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.bar(np.arange(len(data)), data, width=0.5)
    # plt.show()
    plt.savefig(filename, bbox_inches="tight")

graph("Seriale",                "ser",   serial,         "Grandezza dell'immagine", "Secondi")
graph("Seriale (Ghost Area)",   "sergh", serial_ghost,   "Grandezza dell'immagine", "Secondi")
graph("Parallelo",              "par",   parallel,       "Grandezza dell'immagine", "Secondi")
graph("Parallelo (Ghost Area)", "pargh", parallel_ghost, "Grandezza dell'immagine", "Secondi")

graph_sp("Speedup",              "sppar",   np.array(serial[:-1])       / parallel[:-1],       "Grandezza dell'immagine", "Speedup")
graph_sp("Speedup (Ghost Area)", "sppargh", np.array(serial_ghost[:-1]) / parallel_ghost[:-1], "Grandezza dell'immagine", "Speedup")

def throughput(t, w, h):
    return w * h * (4 + 4 + 30 + 30*2 + 1 + 32 + 1) / t / 1000000000

tp_parallel = [ throughput(t, w, h) for t, w, h in zip(parallel[:-1],       ws, hs) ]
tp_ghost    = [ throughput(t, w, h) for t, w, h in zip(parallel_ghost, ws, hs) ]

graph_sp("Throughput",              "tppar",   tp_parallel, "Grandezza dell'immagine", "Pixel al secondo (in miliardi)")
graph_sp("Throughput (Ghost Area)", "tppargh", tp_ghost,    "Grandezza dell'immagine", "Pixel al secondo (in miliardi)")
