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

# if __name__=='__main__':
    # help_fn_funcs = [fn_name for fn_name in dir(help_fn) if "__" not in fn_name]
    # __all__ = help_fn_funcs
    
#     import sys
#     from inspect import getmembers, isfunction
#     fcn_list = [o[0] for o in getmembers(sys.modules[__name__], isfunction)]
#     print(f"All Function Names: ", locals().keys())
    