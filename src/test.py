import pynput
import time

keyboard = pynput.keyboard.Controller()
mouse = pynput.mouse.Controller()
x, y = mouse.position
count = 20
n = 0
while n<count:
    keyboard.press(pynput.keyboard.Key.down)
    x, y = mouse.position
    mouse.position = (x-2, y)
    n += 1
    time.sleep(0.05)