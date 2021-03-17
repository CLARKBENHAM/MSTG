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

from collections import namedtuple, Counter
import pickle
import os
import sys
github_dir = "c:\\Users\\student.DESKTOP-UT02KBN\\MSTG"
os.chdir(f"{github_dir}\\Market_Gamma_(GME)")


import pyautogui as pygu
import pydirectinput

from PIL import Image, ImageChops, ImageDraw

from pytesseract import pytesseract

from skimage.filters import threshold_local
import cv2

pytesseract.tesseract_cmd = "c:\\Program Files\\Tesseract-OCR\\tesseract.exe"
sys.exit()
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
_, head_bot, *_ = pygu.locate("header_top_border.png","data_pics\img0.png" ) 
#top of scrollbar up arrow touches bottom of column header
head_bot -= 9 #bottom of header
sw, sh = Image.open("data_pics\img0.png").size
boundry_sz = 15
header = Image.open("data_pics\img0.png").convert('L'
                                                  ).crop((boundry_sz, head_bot-30, sw-boundry_sz, head_bot))
col_names, bnd_box = get_boundry(header)
#strikes box includes a space for the 'right arrow' next to the contract row
bnd_box[0] = (15, 0, bnd_box[0][2], bnd_box[0][3])

if col_names[-1] == 'IV':
    #got a repeat for last value? Not sure why
    del col_names[-1] 
    del bnd_box[-1]
#since these values aren't in every row, can't tell which row apply too
bad_ix = col_names.index("Last Trade")
bad_ix2 = col_names.index("Change")
del col_names[bad_ix],bnd_box[bad_ix],
del  col_names[bad_ix2-1], bnd_box[bad_ix2-1]

name2ix = {n:ix for ix,n in enumerate(col_names)}
#%% temp
header2screen = lambda b: (b[0]+boundry_sz, head_bot, b[2]+boundry_sz, sh) #from header clipping to screen shot
screen_bnd = [header2screen(bx) for bx in bnd_box]
vals = []
for img in os.listdir("data_pics"):
    break
#crop non-data: contract headers + stuff at top

# split = list(pygu.locateAll(f"header_down_arrow.png","data_pics\{img}" ))
# im = Image.open(f"data_pics\{img}")
img = "del.png"
split = list(pygu.locateAll("header_down_arrow.png",img))
split_tops = [t for _,t,*_ in split] + [sh-63]#ignore desktop icon bar at bottom

im = Image.open(img)

data_im = []
for t1,t2 in zip(split_tops[:-1], split_tops[1:]): 
    data_im += [im.crop((0, t1+25, sw, t2-5))]#only get data r0ws

new_h = sum([d.size[1] for d in data_im])
new_w = sw
new_im = Image.new('L', (new_w, new_h))
y_offset = 0
for d in data_im:
    new_im.paste(d, (0, y_offset))
    y_offset += d.size[1]

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
    
def img2values(img_path, col_names=col_names, bnd_box=bnd_box):
    """returns values for a PIL Image that is cropped to show only 1 column of data
    if outtype in (int, float) returns median numeric prediction 
        of 3 different threshold preprocessing
    outtype: ID's column and useful for constraining # of periods
    """
    im = Image.open(img_path)
    sw, sh = im.size
    split = list(pygu.locateAll("header_down_arrow.png", img_path))
    #need to cut desktop icon bar at bottom; else will be counted as a row
    split_tops = [t for _,t,*_ in split] + [sh-63]
    
    #only get data rows; cutout top headers, any subheaders in the middle of text
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
    header2clipped = lambda bx:  (boundry_sz + bx[0], 0, boundry_sz + bx[2], new_h)
    for bx,n in zip(bnd_box, col_names):
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
        my_config = '--psm 6 digits tessedit_char_whitelist=0123456789\\.\\,'
        
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
                     if ocr_l[i] != mxl] #does better than common_len
            if nl > 1:
                print(f"warning ocr processes {bad_n}, failed for {n} on {img_path}")
            else:
                print(f"Warning ALL ocr processes Disagreed for {n} on {img_path}")
            s = preds[ocr_l.index(mxl)]
            
        #decimal placement check; ERRORS on OPEN VOLUME
        sum_seg = 0
        out = []
        for ix, (t1,t2) in enumerate(zip(split_tops[:-1], split_tops[1:])): 
            seg_sz = (len(s) * (t2-t1))//(split_tops[-1] - split_tops[0]) 
            if len(split) -2 == ix:
                segment = s[sum_seg:]
            else:
                segment = s[sum_seg:seg_sz]
            for ix in range(1, len(segment)-1):
                while segment[ix]*8 > segment[ix-1] and segment[ix]*8 > segment[ix+1]:
                    segment[ix] /= 10
                while segment[ix]*8 < segment[ix-1] and segment[ix]*8 < segment[ix+1]:
                    segment[ix] *= 10
            out += segment
            sum_seg += seg_sz
        vals += [out]
    return vals

vals = img2values(img)
df = pd.DataFrame(list(zip(*vals)))
df.head()
#%% extra info by cell; split on row lines
#each row is ~26 pixels
import math
#WARNING: bottom, right sides of img in MSFT display have a bevel added; not actually on img. 
#       eg Image.fromarray(255*np.ones((500,500))).show()
crop_im = new_im.crop(header2clipped(bnd_box[5]))

cv_im = np.array(crop_im)
result = cv_im.copy()

_, th_l = cv2.threshold(cv_im, 120, 255, cv2.THRESH_BINARY)
#erode, dilate have backwards effects, since will invert colors. erode makes more black->more white
kernel_hor = np.ones((5, 50), dtype=np.uint8)
erode = cv2.erode(th_l, kernel_hor)#black squares where each number is

kernel_ones = np.ones((3, 5), dtype=np.uint8)
blocks = cv2.dilate(erode, kernel_ones)
# Image.fromarray(blocks).show() 

# blocks[:,-5:] = 255#some border boundries get written to white size 

# kernel_hor = np.ones((5, 20))
# kernel_hor[0,:] *= -1#np.uint8(0)
# kernel_hor[-1,:] *= -1#np.uint8(0)
# kernel_v = np.array([[-1,-1,-1]*5, 
#                      [0,0,0]*5,
#                      [1,1,1]*5])
# erode = cv2.filter2D(blocks, -1, kernel_v)
# [2:-2, 2:-20] #some boundry siz

# edges = cv2.Canny(~blocks, 80, 120)
# lines = cv2.HoughLinesP(edges, 1, math.pi/2, 2, None, 30, 1);
# for line in lines[0]:
#     pt1 = (line[0],line[1])
#     pt2 = (line[2],line[3])
#     cv2.line(cv_im, pt1, pt2, (0,0,255), 3)
# Image.fromarray(cv_im).show() 

# # Detect horizontal lines
# horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40,1))
# detect_horizontal = cv2.morphologyEx(~blocks, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
# cnts = cv2.findContours(detect_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cnts = cnts[0] if len(cnts) == 2 else cnts[1]
# for c in cnts:
#     print("horizontal: ",c)
#     cv2.drawContours(result, [c], -1, (36,255,12), 2)
# Image.fromarray(result).show()     

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
                        slice(min(c[:,0]), max(c[:,0])))

im_data = []
_v = []
outtype = int
for c in contours:
    b = contour2box(c) 
    im_data += [crop_im.crop(b)]
    _im = cv_im[contour2cv(c)] 
    #need to improve pre-processing
    
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
#%%
# Image.fromarray(cv_im[contour2cv(contours[4])]).show()
sh = lambda m: Image.fromarray(m).show()
_im = cv_im[contour2cv(contours[-1])]
blur = cv2.GaussianBlur(_im,(3,3),0)
ret3,th3 = cv2.threshold(_im,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
pytesseract.image_to_string(th3, config= my_config)
#%% scrap
# crop_im = new_im.crop(header2clipped(bnd_box[5]))
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

crop_im = new_im.crop(header2clipped(bnd_box[2]))

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
#%%
# pytesseract.image_to_string(thres_im)
Image.fromarray(thresh_im).show()
# Image.fromarray(close_im).show()
# Image.fromarray(result).show()

#%%
im = Image.open("data_pics\img0.png").convert('LA')
draw = ImageDraw.Draw(im)
w,h = im.size
for ix in (top,top-10,top-20): 
    draw.line((0, ix, 1920,ix), fill=0, width=2)
im.show()


#%%  #helpful asides
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

def _concat_img(data_im, how='h'):
    """conatenate a list of Images
    how: h for horizontal, v for vertical
    """
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