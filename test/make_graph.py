import sys
import numpy as np
import matplotlib.pyplot as plt

program_type = [ "Seriale", "Seriale (Ghost Area)", "Parallelo", "Parallelo (Ghost Area)" ]
labels = [ str(2**x) for x in range(7, 7+5) ]
all_data = [
        [
        [  0.475469, 0.474927, 0.464925, 0.444403, 0.486685 ],  # serial (my pc)
        [  0.475469, 0.474927, 0.464925, 0.444403, 0.486685 ]   # serial (raptor)
    ],
        [
        [  0.475469, 0.474927, 0.464925, 0.444403, 0.486685 ],  # serial-ghost (my pc)
        [  0.475469, 0.474927, 0.464925, 0.444403, 0.486685 ]   # serial-ghost (raptor)
    ],
        [
        [  0.475469, 0.474927, 0.464925, 0.444403, 0.486685 ]   # parallel
    ],
        [
        [  0.475469, 0.474927, 0.464925, 0.444403, 0.486685 ]   # parallel-ghost
    ]
]

for program, data_collection in zip(program_type, all_data):
    plt.title(program)
    width = 0
    for data in data_collection:
        plt.xticks(range(len(data)), labels)
        plt.bar(np.arange(len(data)) + width, data, width=0.3)
        width += 0.3
    plt.show()

