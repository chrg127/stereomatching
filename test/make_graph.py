import sys
import numpy as np
import matplotlib.pyplot as plt

program_type = [ "Seriale", "Seriale (Ghost Area)", "Parallelo", "Parallelo (Ghost Area)" ]
labels = [ "240x135", "480x270"," 960x540"," 1920x1080"," 3840x2160"," 7680x4320", "Media" ]

all_data = [
        [
        [0.085686, 0.341234, 1.407778, 5.568949, 22.215088, 89.347372, 19.8276],  # serial (my pc)
        [0.157987, 0.629014, 2.515071, 10.093090, 40.648059, 165.002237, 36.5075]   # serial (raptor)
    ],
        [
        [0.008182, 0.033918, 0.131949, 0.477217, 3.489891, 13.781092, 2.9870],  # serial-ghost (my pc)
        [0.026404, 0.112614, 0.432801, 1.685960, 8.995003, 35.104033, 7.7261]   # serial-ghost (raptor)
    ],
        [
        [0.001217, 0.002516, 0.009254, 0.042987, 0.164412, 0.602978, .1372]   # parallel
    ],
        [
        [0.001065, 0.007141, 0.002282, 0.032478, 0.156047, 0.465563, 0.08501483]   # parallel-ghost
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

