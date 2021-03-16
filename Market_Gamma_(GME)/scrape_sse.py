#import directly from Street smart edge
import numpy as np
import pandas as pd
# import beautifulsoup4
import lxml.html
import requests
# import requests_cache
import re

from datetime import datetime
import time
import random

from collections import namedtuple
import pickle
import os
import sys
github_dir = "c:\\Users\\student.DESKTOP-UT02KBN\\MSTG"
os.chdir(f"{github_dir}\\Market_Gamma_(GME)")


import pyautogui as pygu
import pydirectinput

from PIL import Image, ImageChops, ImageDraw

if __name__ == "__main__":
    #need to start w/ SSE 2nd from bottom row selected
    pygu.moveTo(x=1897,y=998, duration=0.359)
    pygu.doubleClick()
    cnt = 0
    while True:
        # slider_loc = list(pygu.locateAllOnScreen("slider_bottom.png"))[0]
        slider_loc = (1890, 925, 19,87)
        if len(list(pygu.locateAllOnScreen("slider_bottom.png",
                                           region=slider_loc))) > 0:
            print("seen all options")
            break
        #or check bottom
        pygu.screenshot(f"data_pics\img{cnt}.png")
        cnt += 1
        if cnt < 2:
            time.sleep(2 + 3*random.random())
        else:
            time.sleep(5 + 30*random.random())
        pygu.keyDown("pgdn"); time.sleep(0.1 + random.random()/10); pygu.keyUp("pgdn");
        # for _ in range(20):
        #     pygu.click()
        print(cnt, os.listdir("data_pics"))
        
#%%
#left top right bottom
cropper = lambda im: im.crop((10, 422, 1910,1010)) #for values
id_crop = lambda im: im.crop((158, 489, 360, 980)) #for just contract details

def remove_duplicates():
    """filter by eg. GME 03/19/2023 950 C
     NOTE: THis would remove values for the same contract collected at different times
    """
    cnt = int(max(os.listdir(f"{github_dir}\Market_Gamma_(GME)\data_pics"),
                  key = lambda i: i[3:-4]
                  )[3:-4])
    last = id_crop(
            Image.open(f"{github_dir}\Market_Gamma_(GME)\data_pics\img{cnt}.png"))
    cnt -= 1
    n_fails = 0
    while True:
        duplicate = id_crop(
                        Image.open(f"{github_dir}\Market_Gamma_(GME)\data_pics\img{cnt}.png"))
        # if not np.any(ImageChops.subtract(last, duplicate)):
        if ImageChops.difference(last, duplicate).getbbox() is None:
            print(f"Removing {cnt}")
            os.remove(f"data_pics\img{cnt}.png")
        else:
            print("the image ", cnt, "differs")
            if n_fails > 2:
                break
            n_fails += 1
        cnt -= 1
#%%
#17 indexes
def get_boundry(header, check=False):
    """get ix that seperate header columns
    check: visually plots to confirm where ix marked
    """
    # header = Image.open("table_header.png").convert('L')
    header_arr = np.array(header.crop((0,10, 1595,24)))

    #sep bar is 2 pixels wide and lighter then surrounding
    boundry_ix = []
    for c_ix in range(1, header_arr.shape[1] - 2):
        if np.all(np.logical_and(header_arr[:,c_ix - 1] > header_arr[:,c_ix],
                                 header_arr[:,c_ix + 1] < header_arr[:,c_ix],
                                 header_arr[:,c_ix + 2] > header_arr[:,c_ix],
                                 )):
            boundry_ix += [c_ix]
    #doesn't work, but would be ideal list(pygu.locateAll("table_header_sep.png","table_header.png" ))

    if check:
        im = Image.open("table_header.png").convert('L')
        draw = ImageDraw.Draw(im)
        w,h = im.size
        for ix in boundry_ix: 
            draw.line((ix,0, ix,h), fill=255, width=2)
        im.show()
        
    boundry_ix.insert(0,0)
    w,h = header.size
    boundry_ix += [w-1]
    bnd_box = [(ix1, 0, ix2,h) for ix1, ix2 in zip(boundry_ix[:-1], 
                                                   boundry_ix[1:])]
    col_names = []
    for bx in bnd_box:
        ocr = pytesseract.image_to_string(header.crop(bx))
        try:
            s = re.search("[a-zA-Z ]+", ocr).group()#filter cruft
        except Exception as e:
            if ocr == '\x0c':
                s = 'IV'
            else:
                raise e
        col_names += [s]
    return col_names, bnd_box

#horizontal ambigious, but vertical position certain
_, bot, *_ = pygu.locate("header_top_border.png","data_pics\img0.png" )
bot -= 9
header = Image.open("data_pics\img0.png").convert('L'
                                                  ).crop((15, bot-30, 1905, bot))
col_names, bnd_box = get_boundry(header)
if col_names[-1] == 'IV':
    #got a repeat for last value? Not sure why
    del col_names[-1] 
    del bnd_box[-1]
bad_ix = col_names.index("Last Trade")
bad_ix2 = col_names.index("Change")
del col_names[bad_ix],bnd_box[bad_ix],
del  col_names[bad_ix2-1], bnd_box[bad_ix2-1]

name2ix = {n:ix for ix,n in enumerate(col_names)}
#%%
header2screen = lambda b: (b[0]+15, bot, b[2]+15, 1080) #from header clipping to screen shot
screen_bnd = [header2screen(bx) for bx in bnd_box]
vals = []
for img in os.listdir("data_pics"):
    break
# split = list(pygu.locateAll(f"header_down_arrow.png","data_pics\{img}" ))
split = list(pygu.locateAll("header_down_arrow.png","del.png" ))
im = Image.open(f"data_pics\{img}")
for l,t, *_ in split: 
    top = im.crop(subheader)

for bx,n in zip(screen_bnd, col_names):
    ocr = pytesseract.image_to_string(im.crop(bx))
    if n == 'Symbol':
        s = re.findall("[a-zA-Z0-9 ]+", ocr) 
    else:
        s = list(map(float, re.findall("\d+", ocr)))
    print(n, len(s))
    vals += [s]
vals
#%%
from pytesseract import pytesseract
# pytesseract.pytesseract.tesseract_cmd = 'C:\Program Files\Tesseract-OCR\tesseract.exe'
pytesseract.tesseract_cmd = "c:\\Program Files\\Tesseract-OCR\\tesseract.exe"

pytesseract.image_to_string(cropper(Image.open(f"data_pics\{img}")))

#%%
im = Image.open("data_pics\img0.png").convert('LA')
draw = ImageDraw.Draw(im)
w,h = im.size
for ix in (top,top-10,top-20): 
    draw.line((0, ix, 1920,ix), fill=0, width=2)
im.show()


#%% 
def get_position():
    pos_l = []
    for _ in range(4):
        time.sleep(3)
        pos = pygu.position()
        print("start", pos)
        pos_l += [pos]
    x = [i.x for i in pos_l]
    y = [i.y for i in pos_l]
    #left top right bottom
    print((min(x), min(y), max(x), max(y)), "\n", pos_l)


#%%
#scrape

from scipy.signal import convolve2d
def _find_boundry_by_hfilter():
    sep = Image.open("table_header_sep.png").convert('L')
    w,h =  sep.size
    sep = np.array(sep.crop((0, h//5, w, 4*h//5)))#filter top/bottom boundaries
    boundary_sz = len(set([repr(sep[:,i]) for i in range(w)])) - 1
    kernel = [1] + [0]*boundary_sz + [-1]
    kernel = np.tile(kernel, (header.shape[0],1))
    out = convolve2d(header, kernel)
    Image.fromarray(out, 'L').show()

#%%    
# sep
h_off = (header.shape[0] - h)//2
header = header[h_off + h//5 : h_off + 4*h//5, :]
for c_ix in range(off, header.shape[1] - w):
    if np.all(header[:,c_ix:c_ix+w] == sep):
        print(c_ix)
# for 
# # pygu.position()
# # for _ in range(9):
# #     print(pygu.position)

# pydirectinput.doubleClick()
# for i in range(4):
#     print(i)
#     pydirectinput.keyDown('down')
#     time.sleep(2)
#     pydirectinput.keyUp('down')

# # pydirectinput.keyDown('down')
# #%%
# pygu.screenshot("data_pics\del.png")
# pygu.moveTo(x=1896,y=999, duration=0.259)

# #%%
# for i in range(4):
#     print(i)
#     time.sleep(random.random()/3)
#     pydirectinput.keyDown('down')


# #%%
# # pygu.press("pagedown")

# # pygu.click(clicks=23)
# # for _ in range(5):
# #     time.sleep(0.29321)
# #     pygu.mouseDown() 
# #     time.sleep(0.34)
# #     pygu.mouseUp()
# import win32con

# import win32api

# win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, int(1896), int(999), 0, 0)