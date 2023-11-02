import numpy as np
import math
from matplotlib import pyplot as plt
import cv2
import random

def find_frontier(map, config_space, pixels_per_chunk=150000, display=False):
    contours, hierarchy = cv2.findContours((map).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if display:
        map_display = map.copy().astype(np.uint8)
        map_display = np.stack([map_display, map_display, map_display], axis=-1)

    frontier_list = []
    for contour in contours:
        if cv2.arcLength(contour,True) < 80:
            continue
        epsilon = 0.005 * cv2.arcLength(contour,True)
        contour = cv2.approxPolyDP(contour,epsilon,True)

        if display:
            cv2.drawContours(map_display, [contour], 0, (0,255,0), 3)

        dist = 0
        num = 0
        for i in range(0, len(contour) - 1):
            if config_space[contour[i+1][0][1]][contour[i+1][0][0]] == 0 or\
               config_space[contour[i][0][1]][contour[i][0][0]] == 0:
                continue
            dist += math.hypot(contour[i+1][0][1] - contour[i][0][1],
                               contour[i+1][0][0] - contour[i][0][0])
            if dist > pixels_per_chunk:
                # Rnadom sample to prevent stuck
                sample = num // 2
                frontier_list.append((contour[i - sample][0][1],
                                       contour[i - sample][0][0]))
                if display:
                    cv2.circle(map_display, (int(frontier_list[-1][1]), 
                                             int(frontier_list[-1][0])), 
                                             5, (255, 0, 0), -1)
                dist = 0
                num = 0
            num += 1

    if display:
        fig = plt.figure(1)
        plt.imshow(map_display)
        plt.pause(0.8)
        plt.figure(0)

    return frontier_list
