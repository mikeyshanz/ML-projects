from PIL import ImageGrab, Image
import numpy as np
import cv2
import time
import win32api

# The first step is to set the location of the game screen
game_screen = (468, 106, 1060, 720)

# Now you will click randomly and find the dynamic sections of the game
# These will be the score, player, enemies, and maybe other things maybe
# This will also decide if clicking/mouse or the keys will affect the game


# def discover_keys(game_screen):
time.sleep(3)
base_screen_list = list()
for frame_cap in range(100):
    printscreen = np.array(ImageGrab.grab(bbox=game_screen))
    base_screen_list.append(printscreen)
    # click = win32api.GetKeyState(0x01)
    # time.sleep(.1)


# img = Image.fromarray(screen_list[0])
# img.show()

