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
import pathlib

#crop box order: (left top right bottom)
LR_OFFSET = 12#amount to cut from sides of screen
FSW, FSH = pygu.screenshot().size#full screen 
VALID_ROW_HTS = range(22,29)#pixel size of valid rows

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

def _press_page_down():
    """presses page down keys, needs to adjust since some keys presses too fast 
    for app to catch"""
    #so far no duplicates from app not reacting fast enough
    fixed_delay = 1#3
    mult_rand_delay = 3#3
    time.sleep(fixed_delay + mult_rand_delay*random.random())
    #don't think SSE checks for automated behavior; but just in case
    # if reps < 4:
    #     pass
    #     # time.sleep(2 + 3*random.random())
    # else:
    #     pass
        # break
        # time.sleep(5 + 30*random.random())
    fixed_hold = 0.1
    mult_rand_hold = 1/10
    # pygu.keyDown("pgdn"); time.sleep(fixed_hold + random.random()*mul_rand_hold); pygu.keyUp("pgdn");
    pygu.keyDown("pgdn")
    time.sleep(fixed_hold + random.random()*mul_rand_hold)
    pygu.keyUp("pgdn")

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
    while True:
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
        if is_at_bottom():
            break
        _press_page_down()
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
    while True:
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
        if is_at_bottom(rows_open=True):
            break
        _press_page_down()

#have dups 10 images apart? when got to img88 somehow slid back to img78
#may have been errant click? check in future, why have to use all2all
def _rename():
    "preserving img order; makes imgxx.png continous. [1,4,5] -> [0,1,2] same order"
    prev_names = sorted(os.listdir("data_pics"), key = lambda s: int(s[3:-4]))
    cnt = 0
    for p in prev_names:
        os.rename(f"data_pics\\{p}", f"data_pics\\img{cnt}.png")    
        cnt += 1

def _remove_duplicates_stack(rename = False):
    """filter by eg. GME 03/19/2023 950 C
    removes only in stack order, top to immeditatly below
    eg. 99 vs 98 and if 99 == 98 then 99 vs 97; 99!= 98 then 98 vs. 97  
     NOTE: THis would remove values for the same contract collected at different times
    rename: should rename values so img numbers consecutive
    """
    cnt = int(max(os.listdir(f"{github_dir}\Market_Gamma_(GME)\data_pics"),
                  key = lambda i: int(i[3:-4])
                  )[3:-4])
    #for just contract details ('GME 03/19/2023 950 C') on full screen im
    im = Image.open(f"data_pics\img{cnt}.png")
    is_cropped = im.size < (FSW, FSH)
    if is_cropped:
        header_crop_only = get_header_bnd_bx(im=im)
        header = im.convert('L').crop(header_crop_only)
        header_bnd_box = get_col_boundry(header)
        l, _, r, _ = header_bnd_box[1] #symbol
        h =  im.size[1]
        id_crop = lambda img: img.crop((l, 0, r, h)) 
    else:
        id_crop = lambda img: img.crop((158, 489, 360, 980)) 
    last = id_crop(im)
    cnt -= 1
    n_removed = 0
    while cnt >= 0:
        duplicate = id_crop(
                        Image.open(f"data_pics\\img{cnt}.png"))
        print(ImageChops.difference(last, duplicate).getbbox(), cnt)
        if ImageChops.difference(last, duplicate).getbbox() is None:
            _concat_img([last, duplicate], how='h').show()
            print(f"Removing {cnt}")
            os.remove(f"data_pics\\img{cnt}.png")
            n_removed += 1
        else:
            last = duplicate
        cnt -= 1
        
    if rename and n_removed > 0:
       _rename()
            
def _remove_dups_all2all():
    "compares ALL images to all images, returns duplicates"
    dup_files = set()
    dup_ims = []
    for f1 in os.listdir("data_pics"):
        for f2 in os.listdir("data_pics"):
            if f1 <= f2:#only remove larger
                continue
            im1 = Image.open(f"data_pics\\{f1}")
            im2 = Image.open(f"data_pics\\{f2}")
            if im1 == im2:
                print(f1, f2)
                dup_files.add((f1,f2))
                dup_ims += [(im1,im2)]
                
    remove_f = set([i for i,j in dup_files])
    for f1 in remove_f:
        os.remove(f"data_pics\\{f1}")
    _rename()
    return dup_files, dup_ims

# _remove_duplicates_stack(rename = False)
# _remove_dups_all2all()
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

def cut_subheaders(im, confidence=0.95):
    """only get data rows; cutout any subheaders in the middle of text 
     eg. "Puts Mar 19, 2021 (Fri: 03 days)" get removed
         the grey bars in middle/at top
    also cuts taskbar at bottom, if exists
    confidence: < 0.98
     """
    sw, sh = im.size
    data_pieces = list(pygu.locateAll("header_down_arrow.png",
                                      im,
                                      confidence=confidence))
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
    kernel_hor = np.ones((5, im_w//4), dtype=np.uint8)#each row is ~26 pixels tall
    erode = cv2.erode(th_l, kernel_hor)#black squares where each number is    
    
    # #remove excess curve in front, (from negative sign?)
    # kernel_ones = np.ones((3, min(VALID_ROW_HTS)//2), dtype=np.uint8)
    # blocks = cv2.dilate(erode, kernel_ones)
    blocks = erode
    
    h_sum = np.sum(blocks[:, -im_w//4:], axis=1)
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

    #if no white space at bottom then got a portion of a row, want to exclude anyway
    out = [(0,t, new_im.size[0], b) for t,b in zip(row_breakpoints[:-1],
                                                     row_breakpoints[1:])]
    bad_rows = [i for i in out if i[3]-i[1] not in VALID_ROW_HTS]
    if len(bad_rows) > 0:
        print(f"WARNING!! removing {bad_rows} boundries")

    return [i for i in out if i[3]-i[1] in VALID_ROW_HTS]
    
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

# im = Image.open("data_pics\img108.png")
# header_crop_only = get_header_bnd_bx(im=im)
# header = im.convert('L').crop(header_crop_only)
# header_bnd_box = get_col_boundry(header)
# col_names = get_col_names(header, header_bnd_box)
# new_im = cut_subheaders(im)
# full_row_bnds = get_row_boundries(new_im, header_bnd_box)
#%%
def ocr_all_files():        
    t = time.time()
    dfs = []
    l_times = []
    for ix, pth in enumerate(os.listdir("data_pics")):
        loop_t = time.time()
        fname = pathlib.Path(f"data_pics\\{pth}")
        im = Image.open(fname)
        if ix == 0:
            header_crop_only = get_header_bnd_bx(im=im)
            header = im.convert('L').crop(header_crop_only)
            header_bnd_box = get_col_boundry(header)
            col_names = get_col_names(header, header_bnd_box)
            #try psm 7(1 line) or 8 (1 word)? #no sig improvement where psm 6 fail
            #char_whitelist doesn't work on Tesseract v4.0
            symbol_config =  '--psm 6'
            numeric_config = '--psm 6 digits tessedit_char_whitelist=-0123456789.,'
            #if is data in a 'Symbol' colum
            get_config = lambda b: symbol_config \
                                if b[0] == header_bnd_box[1][0] \
                                else numeric_config                            
    
        new_im = cut_subheaders(im)   
        full_row_bnds = get_row_boundries(new_im, header_bnd_box)
        cell_bnds = {col_name: [(col_bnd[0],
                                row_bnd[1],
                                col_bnd[2],
                                row_bnd[3])
                                for row_bnd in full_row_bnds]
                     for col_bnd, col_name in zip(header_bnd_box,
                                                  col_names)}
        
        #pytesseract casts to RGB anyway, and thresholding worsens results
        df = pd.DataFrame({col_name:[pytesseract.image_to_string(new_im.crop(b),
                                                                 config = get_config(b))
                                for b in col_crop]
                          for col_name, col_crop in cell_bnds.items()
                          })
        
        #Note: bias in using time saved file, not time displayed file
        df['Filename'] = fname #used4 debugging 
        df['Observed Time'] = datetime.fromtimestamp(fname.stat().st_ctime)
        dfs += [df]
        l_times += [time.time() - loop_t]
        print(f"Loop Time: {(time.time() - loop_t)//60:.0f}' {(time.time() - loop_t)%60:.0f} sec")    
        # if ix > 4:
        #     break                                               
    duration = time.time()-t
    print(f"Total Time:{duration//3600:.0f}h  {(duration%3600)//60:.0f}' {(duration)%60:.0f}\"")
    print(f"{np.mean(l_times):.0f}\" per ocr im, SD {np.std(l_times):.2f}\" vs. <4\" per screenshot")
    # Total Time:2h  14' 9"
    # 71" per ocr im, SD 3.75" vs. <4" per screenshot
    
    with open("ocrd_dfs", 'wb') as f:
        pickle.dump(dfs, f)
    # with open("ocrd_dfs", 'rb') as f:
    #     dfs = pickle.load(f)
    
    ocr_df = pd.concat(dfs)
    return ocr_df

ocr_df = ocr_all_files()
#%%
col2re = {'Strikes':'\d+\.\d{2}',
           #.50 C and 7.50 both valid entries
          'Symbol': '[A-Z]+ \d{2}/\d{2}/\d{4} \d*\.\d{2} *[CPcp¢]',
          'Bid': '\d+\.\d{2}',
          'Midpoint': '\d+\.\d{2}',
          'Ask': '\d+\.\d{2}',
          'Volume': '\d+',
          'Open Int':'\d+',
          'Delta': '-{0,1}[01]\.\d{4}',
          'Vega': '\d\.\d{4}',
          'IV Ask': '\d+\.\d{4}',
          'IV Bid': '\d+\.\d{4}',
          'Rho': '\d\.\d{4}',
          'Theta': '-{0,1}\d\.\d{4}',
          'IV': '\d+\.\d{4}',
          'Gamma': '0.\d{4}',
          #know below are right, non-ocr
          'Observed Time': '.+', 
          'Filename': '.+',
          }

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
            [sum(r[3] - r[1] for r in col)
             for col in bnds_by_left]
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

def _check_preprocessing(im_num = (9, 37, 51, 57, 89, 90, 91, 111), bad_only=False):
    """for images with file numbered in iterable in_num, will plot the cell croppings
    for visual inspection
    bad_only: those bounds which have non-standard height, outside VALID_ROW_HTS
    """
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
        if bad_only:
            cell_bnds = [i for i in cell_bnds if i[3]-i[1] not in VALID_ROW_HTS]
            if len(cell_bnds) == 0:#all good
                print(f"No errors for {i}")
                continue
        _check_boundries(new_im, cell_bnds)      

def _num_invalid_ocr(df, check_ix = range(99)):
    "total number of entries across all cells that don't match regex"
    # check_ix = range(len(df))
    return sum([sum(df.iloc[[ix],:].apply(lambda i: len(re.findall(col2re[i.name],
                                                            str(i.values[0]))
                                                  ) == 0))
                for ix in check_ix])

def _invalid_cols(df, check_ix = range(99)):
    """name of columns with entries that don't match regex
        from rows with iloc in check_ix
    """
    invalid_col = lambda i: i.name if \
                            len(re.findall(col2re[i.name],
                                           str(i.values[0]))
                                ) == 0 \
                            else ''
    return set([s for ix in check_ix 
                 for s in df.iloc[[ix],:].apply(invalid_col)
                 if s != ''])

def _invalid_iloc(df,  check_ix = range(99)):
    """iloc ix of entries that don't match regex, given row iloc in check_ix
        returns from OCR columns
    """
    invalid_col = lambda i: i.name if \
                            len(re.findall(col2re[i.name],
                                           str(i.values[0]))
                                ) == 0 \
                            else ''
    out = [(ix, df.columns.get_loc(s)) 
            for ix in check_ix 
                 for s in df.iloc[[ix],:].apply(invalid_col)
                 if s != '']
    assert max(out, key = lambda i: i[1])[1] < 15, "Invalid Entry in non-ocr column"
    return out

def _plot_imgs_concat(bad_cells, mx_h = 20, cut_sep = 20, ret_offset = False):
    """given a list of images, plot them going down in column order
        bad_cells: [<PIL.Image.Image>, ...]
        mx_h: number of images to display in 1 column
        cut_sep: number of pixels to put between images on all sides
        ret_offset: include the top left pixel of where put cells 
    """
    get_w = lambda i:  i.size[0]# - i.size[0]
    get_h = lambda i:  i.size[1]# - i.size[1]
    bad_cells = [bad_cells[ix*mx_h:(ix+1)*mx_h]
                for ix in range(len(bad_cells)//mx_h 
                                + (len(bad_cells) % mx_h > 0))]
    
    #max height in each column, since that used for offset when writing to im
    h_sz = max(
            sum(get_h(r) for r in col)
            for col in bad_cells
            ) + cut_sep*len(bad_cells[0]) #max num rows 
    #sum of max width in each col
    w_sz = sum(
            [get_w(max(col, key = lambda r: get_w(r)))
              for col in bad_cells]
            ) + cut_sep*len(bad_cells) #num cols

    canvas = Image.new('L', (w_sz, h_sz))    
    x_offset, y_offset = 0,0
    offsets = []
    for ix, col in enumerate(bad_cells):
        for r in col:
            canvas.paste(r, (x_offset, y_offset))
            offsets += [(x_offset, y_offset)]
            y_offset += get_h(r) + cut_sep
        x_offset += get_w(max(col, key = lambda r: get_w(r))) +  cut_sep
        y_offset = 0
    if ret_offset:
        return canvas, offsets
    else:
        return canvas
    
#grib writes to wrong spot, tesseract isn't matched to cell. Can tell since "GME" isn't on a strike cell
def _plot_invalid_cells(df, check_ix = range(99)):
    "creates image of all invalid cells, with pytesseracts guess next to it"
    inv_ix = _invalid_iloc(df, check_ix = check_ix)
    bad_cells = []
    prev_fname = ''
    for rix, cix in inv_ix:
        fname = df.iloc[rix]['Filename']
        if fname != prev_fname:
            im = Image.open(fname)
            new_im = cut_subheaders(im)   
            full_row_bnds = get_row_boundries(new_im, header_bnd_box)
            prev_fname = fname
        col_bnd = header_bnd_box[cix]
        row_bnd = full_row_bnds[df.index[rix]]

        cell_bnds = (col_bnd[0],
                     row_bnd[1],
                     col_bnd[2],
                     row_bnd[3])
        bad_cells += [new_im.crop(cell_bnds)]
    
    canvas, offsets = _plot_imgs_concat(bad_cells, ret_offset = True)
    d = ImageDraw.Draw(canvas)
    for (rix, cix), (x_offset, y_offset) in zip(inv_ix, offsets):
        d.text((x_offset + 20, y_offset + 10),
               repr(df.iloc[rix, cix]),
               fill=0,#black
               )
    canvas.show()
    return bad_cells, inv_ix, canvas

def _check_ix_align(n_cells = 100):
    "Check _plot_imgs_concat mapping imgs to offsets"
    blank_cells = [Image.fromarray(np.ones((25,100))*255) 
                   for _ in range(n_cells)]
    for ix,b in enumerate(blank_cells):
        ImageDraw.Draw(b).text((10,10), str(ix), fill=0)
    canvas, offsets = _plot_imgs_concat(blank_cells, ret_offset = True)
    d = ImageDraw.Draw(canvas)
    i = 0
    for (x_offset, y_offset) in offsets:
        d.text((x_offset + 59, y_offset + 10),
               repr(i),
               fill=0,#black
               )
        i +=1
    canvas.show()
    return offsets
    
def _check_row_cropping(bad_cells, inv_ix, check_cut_subheaders=False):
    """result of _plot_invalid_cells
    checks confidence to cut_subheaders and 
    get_row_boundries
    """
    #prev crop
    bad_crop = [b for ix,b in enumerate(bad_cells) 
                if b.size[1] not in VALID_ROW_HTS]
    _plot_imgs_concat(bad_crop).show()
    
    #bad row croppping
    bad_crop_ix = [ix for ix,b in enumerate(bad_cells) 
                   if b.size[1] not in VALID_ROW_HTS]
    bad_files = list(set([ocr_df.iloc[inv_ix[ix][0], 
                                      ocr_df.columns.get_loc("Filename")]
                          for ix in bad_crop_ix]))
    bad_im_num = [int(re.findall("(\d+)", str(i))[0]) for i in bad_files]
    _check_preprocessing(im_num = bad_im_num, bad_only=True)
    
    #bad cut_subheader, check new confidence
    if not check_cut_subheaders:
        return
    crop_inv_ix = [inv_ix[ix] for ix in bad_crop_ix]
    for confidence in (0.97, 0.95, 0.93, 0.9):
        nbad_cells = []
        prev_fname = ''
        ims = []
        for rix, cix in crop_inv_ix:
            fname = df.iloc[rix]['Filename']
            if fname != prev_fname:
                im = Image.open(fname)
                new_im = cut_subheaders(im, confidence = confidence)   
                ims += [new_im]
                full_row_bnds = get_row_boundries(new_im, header_bnd_box)
                prev_fname = fname
            col_bnd = header_bnd_box[cix]
            row_bnd = full_row_bnds[df.index[rix]]
            cell_bnds = (col_bnd[0],
                                row_bnd[1],
                                col_bnd[2],
                                row_bnd[3])
            # if  row_bnd[3] - row_bnd[1] > 16:
            nbad_cells += [new_im.crop(cell_bnds)]
            print(row_bnd, row_bnd[3] - row_bnd[1])
            #title doesn't work on windows?!?
        _plot_imgs_concat(nbad_cells).show(title=f"Bad Crops with cut_subheaders(confidence={confidence})")
        break
    
bad_cells, inv_ix, canvas = _plot_invalid_cells(ocr_df, 
                                                check_ix = range(len(ocr_df)))    
canvas.save("pytesseract_cell_errors.png")
# _check_row_cropping(bad_cells, inv_ix)#likely fixed
# #%%
# #have issue of empty cells, because aren't written if no existing bid-ask prx
# blank_cell = [b for ix, b in enumerate(bad_cells) if ix%20 == 17 and ix > 20][-5]
# blank_ix = [b for ix, b in enumerate(inv_ix) if ix%20 == 17 and ix > 20][-5]

# fname = ocr_df.iloc[blank_ix[0], ocr_df.columns.get_loc("Filename")]
# im = Image.open(fname)
# im.show()
# #%%
# _plot_imgs_concat([b for ix, b in enumerate(bad_cells) if ix%20 == 3 and ix > 20]).show()
# #%%
# blank_cell = [b for ix, b in enumerate(bad_cells) if ix%20 == 17 and ix > 20][-5]
# blank_ix = [b for ix, b in enumerate(inv_ix) if ix%20 == 17 and ix > 20][-5]

# fname = ocr_df.iloc[blank_ix[0], ocr_df.columns.get_loc("Filename")]
# im = Image.open(fname)
# im.show()

# #%%
# #deal with some cells being blanks
# blank_cells, blank_ixs = zip(*[(b,ix) for b,ix in zip(bad_cells, inv_ix)
#                           if np.array(b).min() > 170]#and (np.array(b)==0).sum() ==0]
#                              )#includes orange selected cell, if blank
# # _plot_imgs_concat(blank_cells).show()
# blank_cols = [ocr_df.columns[ix[1]] for ix in blank_ixs]

# # Image.open(ocr_df.iloc[blank_ixs[0][0], ocr_df.columns.get_loc("Filename")]).show()
# rix, cix = blank_ixs[12]
# im = Image.open(ocr_df.iloc[rix, ocr_df.columns.get_loc("Filename")])
# new_im = cut_subheaders(im)
# full_row_bnds = get_row_boundries(new_im, header_bnd_box)
# col_bnd = header_bnd_box[cix]
# row_bnd = full_row_bnds[df.index[rix]]
# cell_bnds = (col_bnd[0],
#                                 row_bnd[1],
#                                 col_bnd[2],
#                                 row_bnd[3])
# new_im.crop(cell_bnds).show()
# #%%
# cell_bnds = [(col_bnd[0],
#                                 row_bnd[1],
#                                 col_bnd[2],
#                                 row_bnd[3])
#              for col_bnd in header_bnd_box
#                  for row_bnd in full_row_bnds]
# [b for b in cell_bnds 
#  if  np.array(new_im.crop(b)).min() > 170]
# #%%
# bad_symbol = ocr_df[ocr_df['Symbol'].apply(lambda i: len(re.findall(col2re['Symbol'],i)) ==0)]
# bad_symbol_cells = []
# for fname,ix in zip(bad_symbol['Filename'], bad_symbol.index):
#         im = Image.open(fname)
#         new_im = cut_subheaders(im)   
#         full_row_bnds = get_row_boundries(new_im, header_bnd_box)
#         # col_bnd = header_bnd_box[cix]
#         row_bnd = full_row_bnds[df.index[ix]]
#         # cell_bnds = (col_bnd[0],
#         #              row_bnd[1],
#         #              col_bnd[2],
#         #              row_bnd[3])
#         bad_symbol_cells += [new_im.crop(row_bnd)]
# _plot_imgs_concat(bad_symbol_cells).show()
# #%%
#%%
col2n_decimal ={'Strikes': 2,#{n:2 if ix <5 else 0 if ix < 7 else 4 for ix,n in enumerate(col_names)}
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

def cast_ocr_col(col):
    "takes series of output of pytesseract and processes"
    if col.name in ('Observed Time', 'Filename'):
        return col
    tp = str if col.name == 'Symbol' else \
         int if col.name in ('Volume', 'Open Int') else \
         float
    guesses = []
    def _cast_val(s):
        nonlocal guesses
        #No always true, multiple non-zero img give this output
        if s == '\x0c':
            guesses += [{repr(s)}]
            return 0
        else:
            s1 = s
            s = s.replace("\n\x0c", "")
            try:
                return tp(re.findall(col2re[col.name], s)[0])
            except:
                #make int regex
                col_re = col2re[col.name].replace(".", "")
                if len(re.findall(col_re, s)) > 0 and col.name != 'Symbol':
                    return tp(re.findall(col_re, s)[0]/10**col2n_decimal[col.name])
                if col.name == 'Bid':
                    return 0
                if col.name == 'Ask':
                    return np.Inf
                guesses += [{repr(s1)}]
                if col.name == 'Symbol':
                    return np.nan
                else:
                    return tp(0)
    out = col.apply(_cast_val)
    print(f"In {col.name}, Guessed on {guesses}")
    #why volume and oi worst by far??
    return out

def _plot_rows_where_not(cond_rows, df):
    "takes df of T/F and plots rows where True"
    if not isinstance(cond_rows, pd.Series):
        cond_rows = cond_rows.apply(any, axis=1)
    cond_rows = cond_rows.values
    files = df['Filename'][cond_rows]
    row_ix = df.index[cond_rows]
    bad_cells = []
    prev_fname = ''
    for f,rix in zip(files, row_ix):
        if f != prev_fname:
            im = Image.open(f)
            new_im = cut_subheaders(im)   
            full_row_bnds = get_row_boundries(new_im, header_bnd_box)
            prev_fname = f
        row_bnd = full_row_bnds[rix]
        bad_cells += [new_im.crop(row_bnd)]
    _plot_imgs_concat(bad_cells, mx_h = len(bad_cells)).show()
    
def check_fix_ocr(df):
    """"checks option conditions/ definitions
    a sufficent condition for ocr errors, but not nessisary.
        (won't detect volume/OI issues)
    Don't seem to be misreading chars, if number exists is likely valid
    """
    #assume if wrong these are going to be larger than should be?

    #if all 3 valid floats, then can only detect, can't fix a misinterpretation
    # chg_mid = 1
    # #many valid bids of 0
    # chg_bid = df['Bid'] == 0 |  df['Bid'] >= df['Midpoint']
    
    
    
    # badbidmid = df['Bid'] > df['Midpoint']
    # badmidask = df['Midpoint'] > df['Ask']
    
    # badbid = badbidmid & df['Midpoint'] >= pred_mid
    # badmid = 1
    # badask = badmidask % df['Midpoint'] <= pred_mid
    
    # chg_bid = df['Bid'] == 0 |  df['Bid'] >= df['Midpoint']
    # chg_mid = 1
    # chg_ask = df['Ask'] == np.Inf | df['Midpoint'] >= df['Ask']
    # if not all(bidlmid) and all(midlask):
    #     print(f"{sum(bidlmid)} locs failed for bid >= mid, {sum(midlask)} for ask <= mid")
    #     df['Bid'][chg_bid] = pred_bid[chg_bid]
    #     df['Midpoint'][chg_mid] = pred_mid[chg_mid]
    #     df['Ask'][chg_ask] = pred_ask[chg_ask]
            
    
    assert all(df[['Vega', 'Volume', 'Open Int', 'Bid', 'Midpoint', 'Ask']] >= 0)
    strike2str = lambda i: str(i) if str(i) != "0.5" else ".50"
    assert all(df.apply(lambda r: strike2str(r['Strikes']) in r['Symbol'], axis=1))
    assert all(df.apply(lambda r: (r['Is_Call'] & (r['Delta']>=0))\
                             or (not r['Is_Call'] & (r['Delta'] <=0)),
                        axis=1))
    #even ~$4 stock has options priced in whole dollar or 0.5$ increments
    assert all(df['Strikes'].apply(lambda i: i%1 in (0.5, 0.0))), "invalid strike ending"
    
    #check monotonic
    g_is_mono = lambda g: all(g[c].is_monotonic or g[c].is_monotonic_decreasing
                              for c in ['Bid', 'Midpoint', 'Ask', 'Delta', 'Vega',
                                        'IV Ask', 'IV Bid', 'Rho', 'Theta', 'IV'])
    g_by_strike = df.groupby(['Is_Call', 'Strikes'])
    g_by_exp = df.groupby(['Is_Call', 'Expiry'])
    assert all(g_is_mono(g) for _,g in g_by_strike)    
    assert all(g_is_mono(g) for _,g in g_by_exp)    
    
    #timespreads all positive
    g_by_strike = df.groupby(['Is_Call', 'Strike'])
    assert all([(np.argsort(g['Expiry']) == np.argsort(g['Ask'])) \
                & (np.argsort(g['Expiry']) == np.argsort(g['Bid']))
                for g in g_by_exp]), "timespread isn't positive"

    #prices monotonic in strike
    g_by_exp = df.groupby(['Is_Call', 'Expiry'])
    assert all([np.argsort(g['Strike']) == np.argsort(g['Ask'])
                if g['Is Call'][0] else
                np.argsort(g['Strike'], reverse=true) == np.argsort(g['Ask'])   #put         
                for g in g_by_exp]), "prices not monotonic"
    
def _check_option_arb(df):#grib, write in other file?
    """"checks option arbitrage conditions
    """
    #butterflys negative
    def _make_butterflys(g):
        "takes groupby object by Is Call and expiry date"
        return [(g[ix-1], g[ix], g[ix], g[ix+1]) for ix in range(1, len(g)-1)]
    
    
    #iron butterflys negative
    
    #no iron butterfly, regular butterly arb
    
    #boxes positive
    
def proc_ocr_df(df):
    "converts OCR'd results from screenshot into other columns"
    df = df.apply(cast_ocr_col).dropna()
    
    pred_mid = np.around((df['Ask'] - df['Bid'])/2, 2)
    pred_ask = np.around(df['Midpoint'] + (df['Midpoint'] - df['Bid']),2)
    midbid = df['Midpoint'] - df['Bid']
    askmid = df['Ask'] - df['Midpoint']
    #assumes min increment is 0.01; 0.0101 for floating point
    good_ix = np.abs(askmid - midbid) <=0.0101
    print(f"{len(df) - sum(good_ix)} locs failed for either bid,mid or ask OCR")
    #known to be wrong
    bad_ask = df['Ask'] == np.Inf
    bad_mid = midbid == 0
    if sum(bad_ask & bad_mid) > 0:
        print(f"had to build {sum(bad_ask & bad_mid)} off bid alone")
        ix = bad_ask & bad_mid
        df['Ask'][ix] = np.around(df['Bid']*1.3 + 0.3,2)
        df['Midpoint'][ix] = np.around(df['Bid']*1.2 + 0.2,2)
    else:
        df['Ask'][bad_ask] = pred_ask[bad_ask]
        df['Midpoint'][bad_mid] = pred_mid[bad_mid]
    #bid is 0 when maybe shouldn't be?
    pred_bid = np.around(df['Ask'] - 2*(df['Ask'] - df['Midpoint']),2)
    ix = (pred_bid > 0.05) & (df['Bid'] == 0)
    print(f"Replaced {sum(ix)} vals in Bid for being 0")
    df['Bid'][ix] = pred_bid[ix]
    
    df['Is_Call'] = df['Symbol'].apply(lambda i: i[-1])
    assert all(df['Is_Call'].isin(['C', 'c', 'P', 'p'])), "invalid reading of Symbol column"
    df['Is_Call'] = df['Is_Call'].isin(['C', 'c', '¢'])
    df['Expiry'] = df['Symbol'].apply(lambda i: datetime.strptime(i.split(' ')[1],
                                                                  '%m/%d/%Y'))
    return df
        
# proc_df = proc_ocr_df(ocr_df)
check_fix_ocr(proc_df)

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
    pass

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
        outtype = int if col2n_decimal[name] == 0 else float
        cell_vals = [_proc_ocr(d, outtype) for d in row_l]
        row_val = outtype("".join(cell_vals))
        
        row_val /= 10**col2n_decimal[name]
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
crop_im = new_im.crop(header2clipped(header_bound_box[0]))
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