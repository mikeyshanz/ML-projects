from PIL import ImageGrab, Image
import numpy as np
import cv2
import time
import win32api

# The first step is to set where the game is

screen_list = list()
action_list = list()
time.sleep(3)
for frame_cap in range(10):
    printscreen = np.array(ImageGrab.grab(bbox=(468, 106, 1060, 720)))
    screen_list.append(cv2.cvtColor(printscreen, cv2.COLOR_BGR2GRAY))
    click = win32api.GetKeyState(0x01)
    if click != 0:
        action_list.append(1)
    else:
        action_list.append(0)
    time.sleep(.1)

img = Image.fromarray(screen_list[0])
img.show()

