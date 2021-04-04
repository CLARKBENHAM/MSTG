#import data from Street smart edge by processing screenshots
#upload to website
import numpy as np
import pandas as pd
# import beautifulsoup4
import lxml.html
import requests
# import requests_cache
import re
import math

from datetime import datetime
import time
import random

from collections import namedtuple, Counter
import pickle
import os
import sys
github_dir = "c:\\Users\\student.DESKTOP-UT02KBN\\MSTG"
os.chdir(f"{github_dir}\\Market_Gamma_(GME)")


import pyautogui as pygu
import pydirectinput

from PIL import Image, ImageChops, ImageDraw

from pytesseract import pytesseract #this didn't work even with single char segmentation
pytesseract.tesseract_cmd = "c:\\Program Files\\Tesseract-OCR\\tesseract.exe"

from skimage.filters import threshold_local
import cv2

import matplotlib.pyplot as plt

# to import calamari-OCR
#download https://github.com/Calamari-OCR/calamari_models/tree/master/uw3-modern-english
#with https://downgit.github.io/#/home
#calamari-predict --checkpoint C:\Users\student.DESKTOP-UT02KBN\Downloads\uw3-modern-english\uw3-modern-english\0.ckpt --files "MSTG\Market_Gamma_(GME)\del.png"
#see https://github.com/Calamari-OCR/calamari/blob/master/calamari_ocr/test/test_prediction.py
#for code
# sys.exit()

from functools import lru_cache#doesn't work for nonhashable fns
import collections
from itertools import groupby

#crop box order: (left top right bottom)
LR_OFFSET = 12#amount to cut from sides of screen
FSW, FSH = pygu.screenshot().size#full screen width

def memoize(func):
    """incase potentially have unhashable inputs and need to filter out
    """
    mx_size = 32
    cache = dict()
    lru_l = []
    def memoized_func(*args, **kwargs):
        vrs_tup = tuple(list(args) + list(kwargs.keys()) + list(kwargs.values()))
        if not all(isinstance(i, collections.Hashable) for i in vrs_tup):
            return func(*args, **kwargs)
        
        if vrs_tup in cache:
            return cache[vrs_tup]
        result = func(*args, **kwargs)
        cache[vrs_tup] = result
        nonlocal lru_l, mx_size
        lru_l += [vrs_tup]
        if len(lru_l) > mx_size:
            first = lru_l.pop(0)
            del cache[first]
        return result
    return memoized_func    

@memoize
def get_header_bnd_bx(im = "data_pics\img0.png", ret_header_top = False):
    """Finds where header bar[eg. "Strikes", ... "Gamma"] is
        im: either path or PIL.IMage
        ret_header_top: returns y-ix of top of header
    """
    if not isinstance(im, str) or os.path.exists(im):
        _, head_bot, *_ = pygu.locate("header_top_border.png",
                                      im)
        if isinstance(im, str):
            sw = Image.open(im).size[0]
        else:
            sw = im.size[0]
    else:
        print("Invalid Path: using screenshot")
        _, head_bot, *_ = pygu.locate("header_top_border.png",
                                      pygu.screenshot()) 
        sw = FSW
    #top of scrollbar up arrow touches bottom of column header
    head_bot -= 9 #bottom of header
    
    header_crop_only = (0, head_bot-30, sw, head_bot)
    if ret_header_top:
        return head_bot - 30
    else:
        return header_crop_only

@memoize
def get_taskbar_top(im):
    """Returns the top of the taskbar or bottom of image
        if there is no taskbar (im already cropped)
        im: path or PIL.Image
    """
    if isinstance(im, str):
        sw, sh = Image.open(im).size
    else:
        sw, sh = im.size
    #imprecise? Possiblly; grib
    has_taskbar = pygu.locate("windows_icon.png",
                              im,
                              confidence = 0.5,    
                              region=(0, sh-75, 75, sh)
                                )
    if has_taskbar is not None:
        _, t, *_ = has_taskbar
        return t - 8
    else:
        return sh
# print(get_taskbar_top(im)   ,get_taskbar_top(im2))   

def crop_fullscreen(im, reuse_im_path = ""):
    """removes non-option headers and sidebars from a full-screened image
    will adjust for layout settings
    reuse_im_path: assume im has the same layout as image at reuse_im_path image
            will reuse cached values from get_header_bnd_bx & get_taskbar_top
    """   
    #check if taskbar at bottom
    if os.path.exists(reuse_im_path):
        header_top = get_header_bnd_bx(im=reuse_im_path, ret_header_top = True)
        data_bottom = get_taskbar_top(im=reuse_im_path)
    else:
        header_top = get_header_bnd_bx(im=im, ret_header_top = True)
        data_bottom = get_taskbar_top(im)
        
    if len(reuse_im_path)>0 and not os.path.exists(reuse_im_path):
        #alright to run on first time
        print(f"Warning Invalid Path {reuse_im_path}: reprocessed Image")

    return im.crop((LR_OFFSET, header_top, FSW-LR_OFFSET, data_bottom))

def is_at_bottom(rows_open = False):
    """check if have scrolled to bottom of screen, 
    rows_open: With bottom rows expanded, but returns false if bottom row selected
                because it would be partially orange
    """
    #isue is width of scroll icon changes with num rows unfolded
    # slider_loc = list(pygu.locateAllOnScreen("slider_bottom.png"))[0]
    # slider_loc = (1890, 925, 19,87)
    #pygu.screenshot(f"bottom_footer.png")
    # ar = np.array(Image.open(f"bottom_footer.png"))
    # Image.fromarray(ar[-20:-3,5:-5]).save("bottom_footer_expanded_rows.png")
    # #use -20:-5 if want to include the bottom of last row, expanded 
    # # and NOT selected w/ orange highlight
    if rows_open:
        return len(list(pygu.locateAllOnScreen("bottom_footer_expanded_rows.png",
                                               confidence = 0.999,    
                                                # region=(1900, 0, 1080, 20)
                                               ))) > 0
    else:
        return len(list(pygu.locateAllOnScreen("bottom_footer.png",
                                               confidence = 0.999,    
                                                # region=(1900, 0, 1080, 20)
                                               ))) > 0   

def take_all_screenshots(is_cropped = False):
    """iterates through SSE once and screenshots non-headers
            saving to .\data_pics
        is_cropped will return only option data if True
        else crops a little on sides so vertical lines not present
    NOTE:
        need to start w/ SSE row at bottom selected
                (select top row and hit down bottom once)
        full screen so can't even see icon bar at bottom
                move taskbar to 2ndary display w/ https://www.tenforums.com/general-support/69164-taskbar-do-not-display-main-display.html (only on 1 monitor; drag to 2ndary)
        Make sure row ends align
    """
    #should be pre-selected? moves arrow down if click and already selected
    # pygu.moveTo(x=1897,y=998, duration=0.359)
    t = time.time()
    pygu.moveTo(x=100,y=0, duration=0.159)
    pygu.doubleClick()
    cnt = max([int(i[3:-4]) for i in os.listdir("data_pics")], 
              default = -1) + 1
    if cnt > 0:
        print(f"Screen shots start at {cnt}")
    reps = 0
    while not is_at_bottom():
        im = pygu.screenshot()
        if is_cropped:
            if reps == 0:
                im.save("data_pics\\template_del.png")
            im = crop_fullscreen(im, reuse_im_path = "data_pics\\template_del.png")
        else:
            im = im.crop((LR_OFFSET, 0, FSW-LR_OFFSET, FSH))
        im.save(f"data_pics\img{cnt}.png")

        cnt += 1
        reps += 1
        time.sleep(3 + 3*random.random())
        #don't think SSE checks for automated behavior; but just in case
        # if reps < 4:
        #     pass
        #     # time.sleep(2 + 3*random.random())
        # else:
        #     pass
            # break
            # time.sleep(5 + 30*random.random())
        pygu.keyDown("pgdn"); time.sleep(0.1 + random.random()/10); pygu.keyUp("pgdn");
    
    os.remove(f"data_pics\\template_del.png")
    print(f"Screen shots end at {cnt-1}")
    print(f"Total Time: {(time.time()-t)//60:.0f}' {(time.time()-t)%60:.0f} sec")    

# take_all_screenshots(is_cropped = True)
#%%
def _expand_strikes():
    """expands all hiden options; as bunched by expiry under single line
    runtime: ~2'. Faster to do by hand
    """
    pygu.moveTo(x=1897,y=998)
    pygu.click()
    while not is_at_bottom(rows_open=True):
        call_dropdown = list(pygu.locateAllOnScreen("calls_expiry_right_arrow.png",
                                                    confidence=0.990))
        put_dropdown = list(pygu.locateAllOnScreen("puts_expiry_right_arrow.png",
                                                   confidence=0.990))
        dropdowns = call_dropdown + put_dropdown
        if len(dropdowns) > 0:
            dropdown = min(dropdowns,
                           key = lambda i: i.top)
            print(dropdown, len(dropdowns))
            pygu.click(dropdown.left + 5,
                       dropdown.top + 5, 
                       duration=0.2 + random.random()/5#sometimes gets stuck/doubleclicks?
                       )
            #sse is slow, check actually expanded
            time.sleep(1) 
            clicked_region = (dropdown.left, dropdown.top, 75, 25)
            while True:
                expanded = list(pygu.locateAllOnScreen("calls_expiry_right_arrow.png",
                                                       confidence=0.990,
                                                       region = clicked_region))
                expanded += list(pygu.locateAllOnScreen("puts_expiry_right_arrow.png",
                                                       confidence=0.990,
                                                       region = clicked_region))
                if len(expanded) == 0:
                    break
                else:
                    time.sleep(1)
        pygu.keyDown("pgdn"); time.sleep(0.1 + random.random()/10); pygu.keyUp("pgdn");

def _remove_duplicates():
    """filter by eg. GME 03/19/2023 950 C
     NOTE: THis would remove values for the same contract collected at different times
    """
    cnt = int(max(os.listdir(f"{github_dir}\Market_Gamma_(GME)\data_pics"),
                  key = lambda i: i[3:-4]
                  )[3:-4])
    #for just contract details ('GME 03/19/2023 950 C') on full screen im
    id_crop = lambda im: im.crop((158, 489, 360, 980)) 
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
def get_col_boundry(header, plot_check=False, remove_variable_existance_cols = True):
    """get box that seperate header columns of a header only image
    header: clipped image of header from get_header_bnd_bx
    plot_check: visually plots to confirm where ix marked
    remove_variable_existance_cols: remove columns("Last Trade", "Change")
        whose values aren't in every row. Only set to false if are going to
        process on row by row basis and can deal w/ non-existance
    """
    header_arr = np.array(header.crop((0,10, FSW-2*LR_OFFSET-10,24)))#header.crop((0,10, 1595,24)))

    #sep bar is 2 pixels wide and lighter then surrounding
    boundry_ix = []
    for c_ix in range(1, header_arr.shape[1] - 2):
        if np.all(np.logical_and(header_arr[:,c_ix - 1] > header_arr[:,c_ix],
                                 header_arr[:,c_ix + 1] < header_arr[:,c_ix],
                                 header_arr[:,c_ix + 2] > header_arr[:,c_ix],
                                 )):
            boundry_ix += [c_ix]
    #doesn't work, but would be ideal list(pygu.locateAll("table_header_sep.png","table_header.png" ))

    if plot_check:
        im = header.convert('L')
        draw = ImageDraw.Draw(im)
        w,h = im.size
        for ix in boundry_ix: 
            draw.line((ix,0, ix,h), fill=255, width=2)
        im.show()
        
    boundry_ix.insert(0,0)
    w,h = header.size
    # boundry_ix += [w-1]
    header_bnd_box = [(ix1, 0, ix2,h) for ix1, ix2 in zip(boundry_ix[:-1], 
                                                   boundry_ix[1:])]
    #strikes box includes a space for the 'right arrow' next to the contract row
    header_bnd_box[0] = (25, 0, header_bnd_box[0][2], header_bnd_box[0][3])    
    
    #these values aren't in every row, can't tell which row apply too
    if remove_variable_existance_cols:    
        removed_names =  get_col_names(header, 
                                       header_bnd_box[2:4],
                                       remove_variable_existance_cols=False)
        assert ['Last Trade', 'Change'] == removed_names
        del header_bnd_box[3]
        del header_bnd_box[2]
    return header_bnd_box

def get_col_names(header, header_bnd_box, remove_variable_existance_cols = True):
    """
    header: clipped image of header from get_header_bnd_bx
    header_bnd_box: result of get_col_boundry

    """
    col_names = []
    for bx in header_bnd_box:
        ocr = pytesseract.image_to_string(header.crop(bx))
        try:
            s = re.search("[a-zA-Z ]+", ocr).group()#filter cruft
        except Exception as e:
            if ocr == '\x0c':
                s = 'IV'
            else:
                raise e
        col_names += [s]

    if remove_variable_existance_cols:    
        assert "Last Trade" not in col_names, "recheck get_col_boundry, should be excluded"
        assert "Change" not in col_names, "recheck get_col_boundry, should be excluded"        
    return col_names

def crop2row(im, bnd, shrink_w = 0):
    """returns a single row based on bounds; preserving im width* 
    shrink_w: extra amount taken off left & Right beyond limits
    bdn: (left, top, right, bottom)"""
    bnd = (shrink_w, 
           bnd[1],
           im.size[0] - shrink_w,
           bnd[3])
    return im.crop(bnd)

def crop2col(im, bnd, shrink_h = 0):
    """returns a single col based on bounds; preserving im height
    bdn: (left, top, right, bottom)"""    
    bnd = (bnd[0],
           shrink_h,
           bnd[2],
           im.size[1]-shrink_h)
    return im.crop(bnd)

def crop2cell(im, col_bnd, row_bnd):
    """
    Takes a column bound, a row bound and returns the intersection
    """
    col_w = col_bnd[2] - col_bnd[0]
    row_w = row_bnd[2] - row_bnd[0]
    assert col_w < row_w, "Think have called with col & row order flipped; should be col then row"
    bnd = (col_bnd[0],
           row_bnd[1],
           col_bnd[2],
           row_bnd[3])
    return im.crop(bnd)

def cut_subheaders(im):
    """only get data rows; cutout any subheaders in the middle of text 
     eg. "Puts Mar 19, 2021 (Fri: 03 days)" get removed
         the grey bars in middle/at top
    also cuts taskbar at bottom, if exists
     """
    sw, sh = im.size
    data_pieces = list(pygu.locateAll("header_down_arrow.png", im))
    #need to cut desktop icon bar at bottom; else will be counted as a row
    split_tops = [t for _,t,*_ in data_pieces] + [get_taskbar_top(im)]
    data_im = []
    for t1,t2 in zip(split_tops[:-1], split_tops[1:]): 
        data_im += [im.crop((0, t1+25, sw, t2-5))]
    new_h = sum([d.size[1] for d in data_im])
    new_w = sw
    new_im = Image.new('L', (new_w, new_h))
    y_offset = 0
    for d in data_im:
        new_im.paste(d, (0, y_offset))
        y_offset += d.size[1]
    #bottom 20 pixels are part of next row in this specific screenshot format
    return new_im

from scipy import signal

def get_row_boundries(new_im, header_bnd_box):
    """
    crop_im: pil image column data 
    header_bnd_box: output of get_header_bnd_box()
    returns list of row boundries for any image with the same height
        (i.e. #of subheaders cut out)
    Note: If look at images directly, windows photos adds an edge on 
            the right, bottom that doesn't exist in image
    """
    crop_im = crop2col(new_im, header_bnd_box[-7])#vega
    cv_im = np.array(crop_im)
    result = cv_im.copy()
    
    #using h-sobel gave too many false positives; instead blurring text horizontally
    
    _, th_l = cv2.threshold(cv_im, 120, 255, cv2.THRESH_BINARY)
    #erode, dilate have backwards effects, since will invert colors. erode makes more black->more white
    _, im_w = th_l.shape 
    kernel_hor = np.ones((7, im_w*3//4), dtype=np.uint8)#each row is ~26 pixels tall
    erode = cv2.erode(th_l, kernel_hor)#black squares where each number is
    
    kernel_ones = np.ones((3, 5), dtype=np.uint8)
    blocks = cv2.dilate(erode, kernel_ones)
    
    h_sum = np.sum(blocks, axis=1)
    empty_row_ix = np.where(h_sum != 0)[0]
    row_breakpoints = [0]
    segment = []
    for i,j in zip(empty_row_ix[:-1], empty_row_ix[1:]):
        segment += [i]
        if i+1 < j and len(segment) > 5:
            row_breakpoints += [int(np.median(segment))]
            segment = []
    
    if len(segment) > 4:
            row_breakpoints += [int(np.median(segment))]
    #little blank space at top
    if row_breakpoints[1] < 8:
       del row_breakpoints[0]         

    # pdb.set_trace()
    #if no white space at bottom then got a portion of a row, want to exclude anyway
    return [(0,t-1, new_im.size[0], b+1) for t,b in zip(row_breakpoints[:-1],
                                                     row_breakpoints[1:])]
    
    #looking for white holes in black background, so colors inverted
    contours, hierarchy  = cv2.findContours(~blocks, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Image.fromarray(cv2.drawContours(cv_im, contours, -1, (0,255,0), 3)).show()    

    #WARNING: cv2 y=0 is bottom, Image y=0 is top.
    contours = [c.reshape(-1,2) for c in contours]
    contour2box = lambda c: (0, #min(c[:,0]),
                             new_im.size[1] - max(c[:,1]) -3,
                             new_im.size[0], #max(c[:,0]), 
                             new_im.size[1] - min(c[:,1]) + 3)
                             
    return [contour2box(c) for c in contours]

def _check_boundries(im, bnds, cut_sep = 20):
    """
    for box in bns that segment of im will be placed in new image with
    cut_sep worth of pixel seperation
    """
    sort_l_bnds = sorted(bnds, key = lambda i: i[0])
    bnds_by_left =  [list(g) for _,g in 
                         groupby(sort_l_bnds , key = lambda i: i[0])]
    sort_t_bnds = sorted(bnds, key = lambda i: i[1])
    bnds_by_top = [list(g) for _,g in 
                   groupby(sort_t_bnds, key = lambda i: i[1])]
    h_sz = max(
            [sum(r[3] - r[1] for r in row)
             for row in bnds_by_left]
            ) + cut_sep*len(bnds_by_top)
    w_sz = max(
            [sum(r[2] - r[0] for r in row) 
             for row in bnds_by_top]
            ) + cut_sep*len(bnds_by_left)

    new_im = Image.new('L', (w_sz, h_sz))

    x_offset, y_offset = 0,0
    for ny, row_bnds in enumerate(bnds_by_top):
        row_bnds = sorted(row_bnds, key = lambda i: i[2])#left most first
        for nx, bnd in enumerate(row_bnds):
            d = im.crop(bnd)
            new_im.paste(d, (x_offset, y_offset))
            x_offset += d.size[0] + cut_sep
        y_offset = max(row_bnds, key = lambda i: i[3])[3] + cut_sep*(ny+1)
        x_offset = 0
    new_im.show()

def _check_preprocessing(im_num = (9, 37, 56, 89, 90, 91)):
    for ix, i in enumerate(im_num):
        im = Image.open(f"data_pics\img{i}.png")
        if ix == 0:#can resuse headers
            header_crop_only = get_header_bnd_bx(im=im)
            header = im.convert('L').crop(header_crop_only)
            header_bnd_box = get_col_boundry(header)
        col_names = get_col_names(header, header_bnd_box)
        new_im = cut_subheaders(im)
        
        full_row_bnds = get_row_boundries(new_im, header_bnd_box)
        cell_bnds = [(col_bnd[0],
                       row_bnd[1],
                       col_bnd[2],
                       row_bnd[3])
                     for row_bnd in full_row_bnds
                     for col_bnd in header_bnd_box]
        _check_boundries(new_im, cell_bnds)    
     
im = Image.open("data_pics\img108.png")
header_crop_only = get_header_bnd_bx(im=im)
header = im.convert('L').crop(header_crop_only)
header_bnd_box = get_col_boundry(header)
col_names = get_col_names(header, header_bnd_box)
new_im = cut_subheaders(im)
full_row_bnds = get_row_boundries(new_im, header_bnd_box)

_check_preprocessing(im_num = (2,3,4))
#%%
# #Works but not useful
# full_row.save("data_table.png")
# full_row.show()
# crop2col(new_im, header_bnd_box[1], shrink_h = 29).show()
# crop2col(new_im, header_bnd_box[1], shrink_h = 0).show()

# single_cell = crop2cell(new_im, header_bnd_box[1], full_row_bnds[1])
# single_cell.show()
# single_cell.save("data_table.png")
#calamari-predict --checkpoint C:\Users\student.DESKTOP-UT02KBN\Downloads\uw3-modern-english\uw3-modern-english\0.ckpt --files "MSTG\Market_Gamma_(GME)\data_table.png"

#pytesseract without config can read symbol single_cell better

#idea: increase region around char when segment from roi
#      increase text size on screen
#      roll own char recognition from k-means for digits
#%%
def _ocr2num(ocr, outtype):
    """returns numeric list from generated output and output type 
    outtype: useful for constraining # of periods
    """
    ocr = ocr.replace(",", "")
    if outtype is float:
        str2f = lambda i: float(i) \
                          if i.count(".") <= 1 \
                          else float(i[:i.index(".")] + i[i.index(".")+1:])
    elif outtype is int:
        str2f = lambda i: int(i) \
                          if i.count(".") == 0 \
                          else int(i.replace(".", ""))
                          
    return list(map(str2f, re.findall("\d+\.*\d*", ocr)))
    
def img2values(img_path, col_names=col_names, header_bnd_box=header_bnd_box):
    """returns values for a PIL Image screenshot
    col_names: names of each column (eg. ["Strikes", ..., "Gamma"])
    header_bnd_box: the boundries for the header image
             only the vertical, x=k boundaries are kept 
             (horizontal y=k are specific to header; 
              replaced with horizontal y=k that depend on final data img height)
    """
    im = Image.open(img_path)
    sw, sh = im.size
    
    #only get data rows; cutout any subheaders in the middle of text 
    # eg. "Puts Mar 19, 2021 (Fri: 03 days)" get removed
    data_pieces = list(pygu.locateAll("header_down_arrow.png", img_path))
    #need to cut desktop icon bar at bottom; else will be counted as a row
    split_tops = [t for _,t,*_ in data_pieces] + [sh-63]
    data_im = []
    for t1,t2 in zip(split_tops[:-1], split_tops[1:]): 
        data_im += [im.crop((0, t1+25, sw, t2-5))]
    new_h = sum([d.size[1] for d in data_im])
    new_w = sw
    new_im = Image.new('L', (new_w, new_h))
    y_offset = 0
    for d in data_im:
        new_im.paste(d, (0, y_offset))
        y_offset += d.size[1]
    
    vals = []
    for bx,n in zip(header_bnd_box, col_names):
        crop_im = new_im.crop(header2clipped(bx))
        outtype = int if  n in ("Volume", "Open Int")  \
                  else str if n == 'Symbol' \
                  else float
              
        if outtype is str:#Symbol column
            ocr = pytesseract.image_to_string(crop_im)    
            vals += [[i for i in re.findall("[a-zA-Z0-9 \/\.]+", ocr)
                      if len(i) > 14]]
            continue 
        
        #median numeric prediction of 3 different threshold preprocessers
        cv_im = np.array(crop_im)    
        if n == 'Symbol':
            my_config = '--psm 6'
        else:
            my_config = '--psm 6 digits tessedit_char_whitelist=-0123456789\\.,'
        
        ocr1 = pytesseract.image_to_string(cv_im, config= my_config)
        
        thresh_im = cv2.adaptiveThreshold(cv_im,
                                          255,
                                          cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY, 
                                          85, 
                                          11)
        ocr2 = pytesseract.image_to_string(thresh_im, config= my_config)
        
        blur = cv2.GaussianBlur(cv_im,(3,3),0)
        ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        ocr3 = pytesseract.image_to_string(th3, config= my_config)
         
        preds = list(map(lambda i: _ocr2num(i, outtype),
                    [ocr1, ocr2, ocr3]))
        ocr_l = list(map(len, preds))
        mnl, mxl = min(ocr_l), max(ocr_l)    
        if mnl == mxl: #preds equal len, 
            s = [sorted([i,j,k])[1] for i,j,k in zip(*preds)]
        else:
            #gave differgent answers in length; use modal length 
            common_len, nl = Counter(
                                list(map(len, preds))
                                     ).most_common(1)[0]
            ocr_names = ("No Preprocess", "Adative Gaussian", "Otsu")
            bad_n = [ocr_names[i] for i in range(3) 
                     if ocr_l[i] != common_len] #does better than common_len
            if nl > 1:
                print(f"warning ocr processes {bad_n}, failed for {n} on {img_path}")
            else:
                print(f"Warning ALL ocr processes Disagreed for {n} on {img_path}")
            s = preds[ocr_l.index(mxl)]
            
        # #decimal placement check; ERRORS on OPEN VOLUME
        # sum_seg = 0
        # out = []
        # for ix, (t1,t2) in enumerate(zip(split_tops[:-1], split_tops[1:])): 
        #     seg_sz = (len(s) * (t2-t1))//(split_tops[-1] - split_tops[0]) 
        #     if len(data_pieces) -2 == ix:
        #         segment = s[sum_seg:]
        #     else:
        #         segment = s[sum_seg:seg_sz]
        #     for ix in range(1, len(segment)-1):
        #         while segment[ix]*8 > segment[ix-1] and segment[ix]*8 > segment[ix+1]:
        #             segment[ix] /= 10
        #         while segment[ix]*8 < segment[ix-1] and segment[ix]*8 < segment[ix+1]:
        #             segment[ix] *= 10
        #     out += segment
        #     sum_seg += seg_sz
        vals += [s]
    return vals

img_path = 'del.png'
vals = img2values(img_path)
df = pd.DataFrame(list(zip(*vals)))
df.columns = col_names
df.head()
#%% extra info by cell; 
def proc_split_on_row_lines(im):
    """
    Split data image by col&row into each individal cell
    Returns 
    -------
    df from read image

    """
    header_crop_only = get_header_bnd_bx(im=im)
    header = im.convert('L').crop(header_crop_only)
    header_bnd_box = get_col_boundry(header)
    col_names = get_col_names(header, header_bnd_box)
    new_im = cut_subheaders(im)   
    full_row_bnds = get_row_boundries(new_im, header_bnd_box)
    cell_bnds = [(col_bnd[0],
                   row_bnd[1],
                   col_bnd[2],
                   row_bnd[3])
                 for row_bnd in full_row_bnds
                 for col_bnd in header_bnd_box]
    
    return cell_bnds

#WARNING: bottom, right sides of img in MSFT display have a bevel added; not actually on img. 
#       eg Image.fromarray(255*np.ones((500,500))).show()
crop_im = new_im.crop(header2clipped(header_bnd_box[9]))

cv_im = np.array(crop_im)
result = cv_im.copy()

_, th_l = cv2.threshold(cv_im, 120, 255, cv2.THRESH_BINARY)
#erode, dilate have backwards effects, since will invert colors. erode makes more black->more white
kernel_hor = np.ones((5, 50), dtype=np.uint8)#each row is ~26 pixels tall
erode = cv2.erode(th_l, kernel_hor)#black squares where each number is

kernel_ones = np.ones((3, 5), dtype=np.uint8)
blocks = cv2.dilate(erode, kernel_ones)

#looking for white holes in black background, so colors inverted
contours, hierarchy  = cv2.findContours(~blocks, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# Image.fromarray(cv2.drawContours(cv_im, contours, -1, (0,255,0), 3)).show()

#WARNING: cv2 y=0 is bottom, Image y=0 is top.
contours = [c.reshape(-1,2) for c in contours]
contour2box = lambda c: (0, #min(c[:,0]),
                         crop_im.size[1] - max(c[:,1]) -3,
                         crop_im.size[0], #max(c[:,0]), 
                         crop_im.size[1] - min(c[:,1]) + 3)#left top right bottom
#contour x,y but cv2 images are y,x
contour2cv = lambda c: (slice(min(c[:,1])-3, max(c[:,1])+3), #y
                        slice(min(c[:,0]+5), max(c[:,0]))#x, don't get a right side bar
                        )
# _draw_contours(contours, cv_im)
# _sh(cv_im[contour2cv(contours[8])])

im_data = []
_v = []
outtype = int
for c in contours:
    b = contour2box(c) 
    im_data += [crop_im.crop(b)]
    _im = cv_im[contour2cv(c)] #all digits 
    # _im = cv_im[cv2.boundingRect(c)]
    
    #need to improve pre-processing
    thresh = cv2.threshold(_im, 0, 255,
                           cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    #will be bad config for 'Symbol'
    my_config = '--psm 7 digits tessedit_char_whitelist=0123456789' #7 = single entry
    
    #?: 1 better on gray, 2 on white?
    ocr1 = pytesseract.image_to_string(_im, config= my_config)
    
    thresh_im = cv2.adaptiveThreshold(_im,
                                      255,
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 
                                      85, 
                                      11)
    ocr2 = pytesseract.image_to_string(thresh_im, config= my_config)
    
    blur = cv2.GaussianBlur(_im,(3,3),0)#sometimes helps, sometimes hurts
    ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    ocr3 = pytesseract.image_to_string(th3, config= my_config)

    ret3,th3 = cv2.threshold(_im,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    ocr4 = pytesseract.image_to_string(th3, config= my_config)    
    # thresh_im = cv2.adaptiveThreshold(_im,
    #                                   255,
    #                                   cv2.THRESH_OTSU,
    #                                   cv2.THRESH_BINARY, 
    #                                   85, 
    #                                   11)
    # ocr4 = pytesseract.image_to_string(thresh_im, config= my_config)
    
    # preds = list(map(lambda i: _ocr2num(i, outtype),
    #                 [ocr1, ocr2, ocr3]))
    preds = []
    for i in [ocr1, ocr2, ocr3, ocr4]:
        preds += _ocr2num(i, outtype)
    print(preds)
    s, n_cnt = Counter(preds).most_common(1)[0]
    # if n_cnt ==1:
    #     print("All disagree")
    
    
    _v += [s]
_concat_img(im_data, how='v').show()
_v
#grib: 2401.2855 gets split into 2401, 2855 by each

#%% need to improve pre-processing
def split_into_digit(new_im, header_bnd_box):
    """
    Split data image by col&row and into each individal digit
    ignores symbol column since "M" is 14 pix wide, same legnth as -1
    Returns 
    -------
    {col_name:
        list of row cell in that col
            list of image of digits in that row cell
            }
    """
    # pass
    digits = []
    bad = []
    bad_roi=[]
    ws = []
    vals = {}
    small_roi = []
    for ix, bx in enumerate(header_bnd_box):#column sep
        if ix == 1:#change config
            continue
        name = col_names[ix]
        crop_im = new_im.crop(header2clipped(bx))
        
        cv_im = np.array(crop_im)
        result = cv_im.copy()
        
        _, th_l = cv2.threshold(cv_im, 120, 255, cv2.THRESH_BINARY)
        #erode, dilate have backwards effects, since will invert colors. erode makes more black->more white
        kernel_hor = np.ones((5, 50), dtype=np.uint8)
        erode = cv2.erode(th_l, kernel_hor)#black squares where each number is
        
        kernel_ones = np.ones((3, 5), dtype=np.uint8)
        blocks = cv2.dilate(erode, kernel_ones)  
        
        #looking for white holes in black background, so colors inverted
        contours, hierarchy  = cv2.findContours(~blocks, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        #WARNING: cv2 y=0 is bottom, Image y=0 is top.
        contours = [c.reshape(-1,2) for c in contours]
        col_vals = []
        for c in contours:#row contounrs
            _im = cv_im[contour2cv(c)] #all digits   
    
            ref = cv2.threshold(_im, 200, 255, cv2.THRESH_BINARY_INV)[1]
            refCnts = cv2.findContours(ref.copy(),
                                       cv2.RETR_EXTERNAL,
             	                       cv2.CHAIN_APPROX_SIMPLE)
            refCnts = refCnts[0] if len(refCnts) == 2 else refCnts[1]
            
            ##sort contours L2R
            boundingBoxes = [cv2.boundingRect(cn) for cn in refCnts]
            cnts, boundingBoxes = zip(*sorted(zip(refCnts, boundingBoxes),
                                             key = lambda b:b[1][0],
                                             reverse=False))
            row_digits = []
            for (i, cn) in enumerate(cnts):#digit sep
                # compute the bounding box for the digit, extract it, and resize
                # it to a fixed size
                (x, y, w, h) = cv2.boundingRect(cn)
                #can remove comma, period either 2 or 4 based on col_name, - from call vs. put
                if w > 10 and h > 5:
                    #all >=17, but some have negative sign included
                    roi = ref[y:y + h, x:x + w]
                    v_sum = np.sum(roi, axis=0)
                    char_w = (8,9)#possible character widths
                    n_chars = w //min(char_w)
                    is_right_aligned = name != 'Strikes'
                    split_digits = []
                    if is_right_aligned:
                        #don't split whole img to exclude neg sign
                        r_border = w
                        while r_border >= min(char_w):
                            char_range = slice(max(r_border - char_w[1],0),
                                               r_border - char_w[0] + 1)
                            sep_ix = v_sum[char_range].argmin()
                            v_sep = max(r_border - char_w[1],0) + sep_ix
                            n_roi = roi[:,   v_sep: r_border]
                            n_roi = cv2.resize(n_roi, (57, 88))
                            r_border = v_sep
                            split_digits  += [n_roi]
                        split_digits = split_digits[::-1]#read in r2l
                    else:
                        char_w = (8,10)#strikes are bolded
                        r_border = 0
                        while r_border <= w - min(char_w):
                            char_range = slice(r_border + char_w[0],
                                               r_border + char_w[1]+1)
                            sep_ix = v_sum[char_range].argmin()
                            v_sep = r_border + char_w[0] + sep_ix
                            n_roi = roi[:,   r_border:v_sep]
                            n_roi = cv2.resize(n_roi, (57, 88))
                            r_border = v_sep  
                            split_digits  += [n_roi]
                            
                    digits += split_digits
                    row_digits += split_digits            
                    bad_roi += split_digits
                    
                    # #issue ploting troughts: 00 is thicker where touch than midline of 0
                    bad += [(bx, c, i)]
                    roi = ref[y:y + h, x:x + w]
                    roi = cv2.resize(roi, (57, 88))
                    # bad_roi += [roi]
                    print(w)
                elif h > 5 and w >=6:
                    #some invalid white sqs with w<6 
                    ws += [w]
                    roi = ref[y:y + h, x:x + w]
                    roi = cv2.resize(roi, (57, 88))
                    # update the digits dictionary, mapping the digit name to the ROI
                    digits += [roi]
                    row_digits += [roi]
                    
            col_vals += [row_digits]
        vals[name] = col_vals[::-1]
    return vals, bad_roi

vals, bad_roi = split_into_digit(new_im, header_bnd_box)

def _check_split_into_digits(new_im, vals):
    h = len(list(vals.values())[0])*88
    col_sep = Image.fromarray(np.ones((h, 50)))
    insert_col_sep = lambda m: _concat_img([m, col_sep], how='h')
    _concat_img([
        insert_col_sep(
            _concat_img([
                _concat_img(row_l, how='h')
                          for row_l in col_l], 
                        how='v'))
        for col_l in list(vals.values())],
                how='h').show()
    new_im.show()

_check_split_into_digits(new_im, vals)

# _make_sq_img(small_roi).show()
# _make_sq_img(bad_roi).show()
# Image.fromarray(ref).show()
# _make_sq_img(digits).show() #lots of doubled chars
# _draw_contours(cnts, _im)
# np.unique(np.array(digits), axis=0, return_counts=1)[1] #digits non-unique
# [pytesseract.image_to_string(i, config= my_config) for i in digits]

#%%
def proc_single_digits(vals):
    #pytesseract isn't accurrant enough for this
    """
    OCR's individual digits into the table they represent
    Parameters
    ----------
    vals : {col_name: [[digits in cell] cell in row]}

    Returns
    -------
    pd DataFrame
    """
    pass
n_decimal ={'Strikes': 2,#{n:2 if ix <5 else 0 if ix < 7 else 4 for ix,n in enumerate(col_names)}
         'Symbol': 2,
         'Bid': 2,
         'Midpoint': 2,
         'Ask': 2,
         'Volume': 0,
         'Open Int': 0,
         'Delta': 4,
         'Vega': 4,
         'IV Ask': 4,
         'IV Bid': 4,
         'Rho': 4,
         'Theta': 4,
         'IV': 4,
         'Gamma': 4}
my_config = '--psm 10 digits tessedit_char_whitelist=0123456789' #10 single char
def _proc_ocr(d, outtype):
    "np.array to single digit cast"
    # base = np.zeros((100,99), dtype=np.uint8) #outlining in black makes worse?
    # base[6:-6, 21:-21] = d
    ocr = pytesseract.image_to_string(Image.fromarray(d), 
                                      config= my_config)
    try:
        return str(int(_ocr2num(ocr, outtype)[0]))
    except:
        print("Failed output of: ", str(ocr))
        return ''
    
out = []
for name, col_l in vals.items():
    row_vals = []
    for row_l in col_l:
        outtype = int if n_decimal[name] == 0 else float
        cell_vals = [_proc_ocr(d, outtype) for d in row_l]
        row_val = outtype("".join(cell_vals))
        
        row_val /= 10**n_decimal[name]
        is_put = False#GRIB!!
        if name == 'Theta':
            row_val *= -1
        elif name in ('Delta', 'Rho') and is_put:
            row_val *= -1
        row_vals += [row_val]
    out += [row_vals]
    # return pd.DataFrame(out, columns = vals.keys()) 

# _df = proc_single_digits(vals)


#%% get bad image
#issue of multiple digits per box
bad_roi = []
neg_contours = []
nonneg_contours =[]
for ix, (bx, c, i) in enumerate(bad): 
    # if ix not in [28, 29, 30, 31, 32, 34, 35, 37, 38, 40]:
    crop_im = new_im.crop(header2clipped(bx))
    cv_im = np.array(crop_im)
    _im = cv_im[contour2cv(c)] #all digits   
    
    # _im = cv2.resize(_im, (500, 1000)) #doesn't really help
    # ref = cv2.dilate(ref, np.ones((10,10)))

    ref = cv2.threshold(_im, 200, 255, cv2.THRESH_BINARY_INV)[1]
    refCnts = cv2.findContours(ref.copy(),
                               cv2.RETR_EXTERNAL,
     	                       cv2.CHAIN_APPROX_SIMPLE)#only returns boxes
    
    refCnts = refCnts[0] if len(refCnts) == 2 else refCnts[1]
    
    if ix in [28, 29, 30, 31, 32, 34, 35, 37, 38, 40]:
        neg_contours += [refCnts]
    else:
        nonneg_contours += [refCnts]
        
    ##sort contours L2R
    boundingBoxes = [cv2.boundingRect(cn) for cn in refCnts]
    cnts, boundingBoxes = zip(*sorted(zip(refCnts, boundingBoxes),
                                     key = lambda b:b[1][0],
                                     reverse=False))

    # i = 0
    cn = cnts[i]

    (x, y, w, h) = cv2.boundingRect(cn)
    roi = ref[y:y + h, x:x + w]
    roi = cv2.resize(roi, (57, 88))
    # update the digits dictionary, mapping the digit name to the ROI
    bad_roi += [roi]
    # Image.fromarray(roi).show()
    
    # _draw_contours(cnts[i], _im)

# _sh(_im)
# _sh(bad_roi[-1])    
# _make_sq_img(bad_roi).show()
#%%
# #no different in contour length for engatives vs non-negatives
# print(list(map(lambda j: [i.shape[0] for i in j], neg_contours))) #n points per contour per image contours
# print("\n\n", list(map(lambda j: [i.shape[0] for i in j], nonneg_contours))) 

v_sum = np.sum(roi, axis=0)
fig,(ax1,ax2) = plt.subplots(2, sharex=True, constrained_layout=True)
ax1.plot(v_sum)
ax2.imshow(Image.fromarray(roi), aspect="auto")
fig.show()


# cv2.calcHist(_im, [0], None, [256], [0,256])

# print(ax1.get_xticks(), ax2.get_xticks())

#%% improve proc for digits of bad cell img 
crop_im = new_im.crop(header2clipped(header_bn_box[]))
cv_im = np.array(crop_im)
result = cv_im.copy()
    
_, th_l = cv2.threshold(cv_im, 120, 255, cv2.THRESH_BINARY)
#erode, dilate have backwards effects, since will invert colors. erode makes more black->more white
kernel_hor = np.ones((5, 50), dtype=np.uint8)
erode = cv2.erode(th_l, kernel_hor)#black squares where each number is

kernel_ones = np.ones((3, 5), dtype=np.uint8)
blocks = cv2.dilate(erode, kernel_ones)  

#looking for white holes in black background, so colors inverted
contours, hierarchy  = cv2.findContours(~blocks, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#WARNING: cv2 y=0 is bottom, Image y=0 is top.
contours = [c.reshape(-1,2) for c in contours]
    
# sh(_im)



_draw_contours(cnts, cv_im)
#%%
# Image.fromarray(cv_im[contour2cv(contours[4])]).show()

_im = cv_im[contour2cv(contours[-1])]
blur = cv2.GaussianBlur(_im,(3,3),0)
ret3,th3 = cv2.threshold(_im,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
pytesseract.image_to_string(th3, config= my_config)
#%% scrap
# crop_im = new_im.crop(header2clipped(header_bnd_box[5]))
crop_im = new_im.crop((30, 0, sw-100, 490))
cv_im = np.array(crop_im)
result = cv_im.copy()
thresh = cv2.threshold(cv_im, 20, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

# Detect horizontal lines
horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40,1))
detect_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
cnts = cv2.findContours(detect_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
for c in cnts:
    print("horizontal: ",c)
    cv2.drawContours(result, [c], -1, (36,255,12), 2)

# Detect vertical lines
vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,10))
detect_vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
cnts = cv2.findContours(detect_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
for c in cnts:
    print("vertical: ",c)
    cv2.drawContours(result, [c], -1, (36,255,12), 2)
    
Image.fromarray(result).show() 
Image.fromarray(thresh).show()
#%%
# import imutils.perspective

crop_im = new_im.crop(header2clipped(header_bnd_box[2]))

# thres_lvl = 90
# _, thresh_im = cv2.threshold(cv_im, thres_lvl, 255, cv2.THRESH_BINARY)
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
# close_im = cv2.morphologyEx(thresh_im, cv2.MORPH_CLOSE, kernel)
# result = 255 - close_im

# thresh = cv2.threshold(cv_im, 127, 255,  cv2.THRESH_OTSU)[1]
# thresh_im = cv2.bitwise_not(thresh)
# dsize = (thresh_im.shape[1]*16, thresh_im.shape[0]*16)
# thresh_im = cv2.resize(thresh_im, dsize)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 12))
    dilation = cv2.dilate(thresh_im, kernel, iterations=1)
    


cv_im = np.array(crop_im)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 12))
dilation = cv2.dilate(thresh_im, kernel, iterations=1)


ocr1 = pytesseract.image_to_string(cv_im, config= '--psm 6 digits tessedit_char_whitelist=0123456789')
thresh_im = cv2.adaptiveThreshold(cv_im,
                                  255,
                                  cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, 
                                  85, 
                                  11)
ocr2 = pytesseract.image_to_string(thresh_im, config= '--psm 6 digits tessedit_char_whitelist=0123456789')
blur = cv2.GaussianBlur(cv_im,(3,3),0)
ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
ocr3 = pytesseract.image_to_string(th3, config= '--psm 6 digits tessedit_char_whitelist=0123456789')

# Image.fromarray(thresh_im).show()
# Image.fromarray(dilation).show()
# Image.fromarray(th3).show()
# ocr = pytesseract.image_to_string(dilation, config= '--psm 6 digits tessedit_char_whitelist=0123456789')
# ocr = pytesseract.image_to_string(crop_im, lang='eng',
#         config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789')
s1 = list(map(float, re.findall("[\d\.]+", ocr1)))
s2 = list(map(float, re.findall("[\d\.]+", ocr2)))
s3 = list(map(float, re.findall("[\d\.]+", ocr3)))
s = [sorted([i,j,k])[1] for i,j,k in zip(s1,s2,s3)]
len(s),s
#%%
cntrs = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cntrs = cntrs[0] if len(cntrs) == 2 else cntrs[1]
result = cv_im.copy()
for c in cntrs:
    # # for each letter, create red rectangle
    # x, y, w, h = cv2.boundingRect(c)
    # cv2.rectangle(result, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # # prepare letter for OCR
    # box = thresh[y:y + h - 2, x:x + w]
    # box = cv2.bitwise_not(box)
    # box = cv2.GaussianBlur(box, (3, 3), 0)

    # # retreive the angle. For the meaning of angle, see below
    # # https://namkeenman.wordpress.com/2015/12/18/open-cv-determine-angle-of-rotatedrect-minarearect/
    # rect = cv2.minAreaRect(c)
    # angle = rect[2]

    # # put angle below letter
    # font = cv2.FONT_HERSHEY_SIMPLEX
    # bottomLeftCornerOfText = (x, y+h+20)
    # fontScale = 0.6
    # fontColor = (255, 0, 0)
    # lineType = 2
    # cv2.putText(result, str(angle), bottomLeftCornerOfText, font, fontScale, fontColor, lineType)

    # do the OCR
    custom_config = r'-l eng --oem 3 --psm 10'
    text = pytesseract.image_to_string(box, config=custom_config)
    print("Detected :" + text + ", angle: " + str(angle))


Image.fromarray(result).show()
pytesseract.image_to_string(result)
# blur = cv2.GaussianBlur(crop_im)
# edge = cv2.Canny(blur, 75, 200)

#%% run main
if __name__ == "__main__":
    pass
    # take_all_screenshots()

#%%  #helpful asides

_sh = lambda m: Image.fromarray(m).show()

def get_position():
    "print from pygu: curosr positions"
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

def _concat_img(data_im, how='h'):
    """conatenate a list of Images
    how: h for horizontal, v for vertical
    """
    if not isinstance(data_im[0], Image.Image):
        print("casting to Image")
        data_im = [Image.fromarray(i) for i in data_im]
        
    if how == 'v':
        new_h = sum([d.size[1] for d in data_im])
        new_w = max([d.size[0] for d in data_im])
    elif how == 'h':
        new_h = max([d.size[1] for d in data_im])
        new_w = sum([d.size[0] for d in data_im])
    new_im = Image.new('L', (new_w, new_h))
    y_offset = 0
    x_offset = 0
    for d in data_im:
        new_im.paste(d, (x_offset, y_offset))
        if how == 'v':
            y_offset += d.size[1]
        elif how == 'h':
            x_offset += d.size[0]            
    return new_im

def _make_sq_img(data_im):
    """a list of Images into a rectangle in row order
    data_im: list of Image of EQUAL SIZE
    """
    if not isinstance(data_im[0], Image.Image):
        print("casting to Image")
        data_im = [Image.fromarray(i) for i in data_im]

    iw, ih = data_im[0].size
    assert all((iw,ih) == i.size for i in data_im)
    n = len(data_im)
    xs = math.ceil(math.sqrt(n))
    ys = math.ceil(n/xs)

    new_im = Image.new('L', (xs*iw, ys*ih))
    y_offset = 0
    x_offset = 0
    for ix,d in enumerate(data_im):
        new_im.paste(d, (x_offset, y_offset))
        x_offset += iw
        if ix%xs == xs-1:
            y_offset += ih
            x_offset = 0
            
    if xs*ys - len(data_im) > 0:
        print(f"Last: {xs*ys-len(data_im)} sqs in Image are empty" )
    return new_im

def _draw_contours(cnts, _im):
    "draws contors on copy of _im, a np.array"
    result = _im.copy()
    for cn in cnts:
        # print("horizontal: ",c)
        cv2.drawContours(result, [cn], -1, (36,255,12), 2)
    Image.fromarray(result).show()   

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