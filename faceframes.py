# -*- coding: utf-8 -*-
"""
Draw Person of Interest Squares

By Jo-dan
"""

import cv2
import numpy as np


def drawline(img, pt1, pt2, color, thickness=1, style='dotted', gap=20):
    dist = ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** .5
    pts = []
    for i in np.arange(0, dist, gap):
        r = i/dist
        x = int((pt1[0] * (1 - r) + pt2[0] * r) + .5)
        y = int((pt1[1] * (1 - r) + pt2[1] * r) + .5)
        p = (x, y)
        pts.append(p)

    if style == 'dotted':
        for p in pts:
            cv2.circle(img, p, thickness, color, -1)
    elif style == 'dashed vertical':
        for p in pts:
            cv2.line(img, (p[0], p[1]-3), (p[0], p[1]+3), color, thickness)
    elif style == 'dashed horizontal':
        for p in pts:
            cv2.line(img, (p[0]-3, p[1]), (p[0]+3, p[1]), color, thickness)
    else:
        s = pts[0]
        e = pts[0]
        i = 0
        for p in pts:
            s = e
            e = p
            if i % 2 == 1:
                cv2.line(img, s, e, color, thickness)
            i += 1

def drawpoly(img, pts, color, thickness=1, style='dotted'):
    s = pts[0]
    e = pts[0]
    pts.append(pts.pop(0))
    for p in pts:
        s = e
        e = p
        drawline(img, s, e, color, thickness, style)

def drawrect(img, pt1, pt2, color, thickness=1, style='dotted'):
    pts = [pt1, (pt2[0], pt1[1]), pt2, (pt1[0], pt2[1])]
    drawpoly(img, pts, color, thickness, style)



def poi_box(win, x, y, w, h, sub_type):
    if sub_type == 'ADMIN':
        box_col = (58, 238, 247)
        accent_col = (58, 238, 247)
    elif sub_type == 'ANALOG':
        box_col = (000, 000, 000)
        accent_col = (58, 238, 247)
    elif sub_type == 'USER':
        box_col = (243, 124, 13)
        accent_col = (243, 124, 13)
    elif sub_type == 'THREAT':
        box_col = (000, 000, 255)
        accent_col = (000, 000, 255)
    elif sub_type == 'UNKNOWN':
        box_col = (255, 255, 255)
        accent_col = (255, 255, 255)

#    if w >= 300:
#        corner_len = 0.05
#        corner_thick = 8
#        point_len = .04
#        point_thick = 2
#        rect_thick = 1
#        gap = 30
#    elif w >= 100:
#        corner_len = 0.05
#        corner_thick = 3
#        point_len = .04
#        point_thick = 1
#        rect_thick = 1
#        gap = 20
#    else:
#        corner_len = 0.05
#        corner_thick = 1
#        point_len = .04
#        point_thick = 1
#        rect_thick = 1
#        gap = 10
    corner_len = 0.1
    corner_thick = int(round(w/38))
    point_len = .06
    # point_thick = int(round(w/140))
    point_thick = 2
    rect_thick = 2
    gap = int(round(w/10))

    # cv2.rectangle(win, (x, y), (x+w, y+h), col, 1)
    # drawrect(win, (x, y), (x+w, y+h), col, 1)
    drawline(win, (x, y), (x+w, y), box_col,
             rect_thick, 'dashed horizontal', gap)
    drawline(win, (x, y+h), (x+w, y+h), box_col,
             rect_thick, 'dashed horizontal', gap)
    drawline(win, (x, y), (x, y+h), box_col,
             rect_thick, 'dashed vertical', gap)
    drawline(win, (x+w, y), (x+w, y+h), box_col,
             rect_thick, 'dashed vertical', gap)
    # top left
    cv2.line(win, (x, y), (x+int(round(corner_len*w)), y),
             accent_col, corner_thick)
    cv2.line(win, (x, y), (x, y+int(round(corner_len*h))),
             accent_col, corner_thick)
    # top right
    cv2.line(win, (x+w, y), (x+w-int(round(corner_len*w)), y),
             accent_col, corner_thick)
    cv2.line(win, (x+w, y), (x+w, y+int(round(corner_len*h))),
             accent_col, corner_thick)
    # bottom left
    cv2.line(win, (x, y+h), (x+int(round(corner_len*w)), y+h),
             accent_col, corner_thick)
    cv2.line(win, (x, y+h), (x, y+h-int(round(corner_len*h))),
             accent_col, corner_thick)
    # bottom right
    cv2.line(win, (x+w, y+h), (x+w-int(round(corner_len*w)), y+h),
             accent_col, corner_thick)
    cv2.line(win, (x+w, y+h), (x+w, y+h-int(round(corner_len*h))),
             accent_col, corner_thick)
    # target points
    cv2.line(win, (x+int(round(.5*w)), y),
             (x+int(round(.5*w)), y+int(round(point_len*h))),
             accent_col, point_thick)
    cv2.line(win, (x+int(round(.5*w)), y+h),
             (x+int(round(.5*w)), y+h-int(round(point_len*h))),
             accent_col, point_thick)
    cv2.line(win, (x, y+int(round(.5*h))),
             (x+int(round(point_len*w)), y+int(round(.5*h))),
             accent_col, point_thick)
    cv2.line(win, (x+w, y+int(round(.5*h))),
             (x+w-int(round(point_len*w)), y+int(round(.5*h))),
             accent_col, point_thick)


def sam_circle(win, x, y, w, h, sub_type):
    circle_col = (214, 214, 214)
    outercircle_col = (10, 10, 10)
    accentcircle_col = (200, 200, 200)
    triangle_colour = (000, 000, 255)
    if w > h:
        r = w
    else:
        r = h
    tran1 = win.copy()
    tran2 = win.copy()
    trant = win.copy()
    cv2.circle(tran1, (x+int(round(.5*w)), y+int(round(.5*h))),
               int(round(0.9*r)), circle_col, 40)
    cv2.circle(tran1, (x+int(round(.5*w)), y+int(round(.5*h))),
               int(round(1*r))+20, outercircle_col, 30)
    cv2.circle(tran2, (x+int(round(.5*w)), y+int(round(.5*h))),
               int(round(1*r))+20, accentcircle_col, 2)
    trant = cv2.addWeighted(tran1, .3, tran2, .7, 0, trant)
    win = cv2.addWeighted(trant, .7, win, .3, 0, win)

#    cv2.line(win, (x-int(round(.5*w)), y-int(round(.5*h))), (x+w+int(round(.5*w)), y-int(round(.5*h))), triangle_colour, 3)
#    cv2.line(win, (x-int(round(.5*w)), y-int(round(.5*h))), (x+int(round(.5*w)), y+h+int(round(.5*h))), triangle_colour, 3)
#    cv2.line(win, (x+w+int(round(.5*w)), y-int(round(.5*h))), (x+int(round(.5*w)), y+h+int(round(.5*h))), triangle_colour, 3)

    cv2.line(win, (x-int(round(.25*r)), y), (x+w+int(round(.25*r)), y),
             triangle_colour, 3)
    cv2.line(win, (x-int(round(.25*r)), y), (x+int(round(.5*r)),
                y+h+int(round(.4*r))), triangle_colour, 3)
    cv2.line(win, (x+w+int(round(.25*r)), y), (x+int(round(.5*r)),
                   y+h+int(round(.4*r))), triangle_colour, 3)

def overlayimg(back, fore, x, y, w, h):
    # Load two images
    img1 = np.array(back)
    img2 = np.array(fore)

    # create new dimensions
    r = float((h)) / img2.shape[1]
    dim = ((w), int(img2.shape[1] * r))

    # Now create a mask of box and create its inverse mask also
    img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    # resize box and masks
    resized_img2 = cv2.resize(img2, dim, interpolation=cv2.INTER_AREA)
    resized_mask = cv2.resize(mask, dim, interpolation=cv2.INTER_AREA)
    resized_mask_inv = cv2.resize(mask_inv, dim, interpolation=cv2.INTER_AREA)

    # I want to put box in co-ordinates, So I create a ROI
    rows, cols, channels = resized_img2.shape
    roi = img1[y:y+rows, x:x+cols]

    # Now black-out the area of box in ROI
    img1_bg = cv2.bitwise_and(roi, roi, mask=resized_mask_inv)

    # Take only region of box from box image.
    img2_fg = cv2.bitwise_and(resized_img2, resized_img2, mask=resized_mask)

    # Put box in ROI and modify the main image
    dst = cv2.add(img1_bg, img2_fg)
    img1[y:y+rows, x:x+cols] = dst
    return img1


def poi_image(frame, x, y, w, h, sub_type):
    assets_path = "gui\\machine\\"
    box_path = assets_path + sub_type.lower() + '_focus.tif'
    box = cv2.imread(box_path)
    # cv2.imshow('wins', box)
    x -= 30
    y -= 20
    w += 60
    h += 60
    return overlayimg(frame, box, x, y, w, h)
    
def poi_infobox(frame, x, y, subject_number, subject_name, subject_type):
    if subject_type == 'ADMIN' or subject_type == 'ANALOG':
        id_colour = (58, 238, 247)
    elif subject_type == 'USER':
        id_colour = (243, 124, 13)
    elif subject_type == 'THREAT':
        id_colour = (000, 000, 255)
    else:
        id_colour = (000, 000, 000)
    multiple = 0.50
    infobox_path = "gui\machine\infobox_slim_short_out.tif"
    infobox = cv2.imread(infobox_path)    
    grey_path = "gui\machine\infobox_slim_in_short.tif"
    grey = cv2.imread(grey_path)
    
    w = int(400 * multiple)
    h = int(219 * multiple)
    
    grey_frame = overlayimg(frame, grey, x, y, w, h)
    frame = cv2.addWeighted(grey_frame, 0.75, frame, 0.25, 0)
    
    id_type = "{} IDENTIFIED".format(subject_type)
    id_alias = "ALIAS: {}".format(subject_name)
    id_num = "***-***-{}".format(str(subject_number).zfill(3))
    cv2.putText(infobox, id_type, (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, id_colour, 2)
    cv2.putText(infobox, id_alias, (15, 92), cv2.FONT_HERSHEY_SIMPLEX, 1, (12, 12, 12), 2)
    cv2.putText(infobox, "SSN:", (15, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(infobox, id_num, (125, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return overlayimg(frame, infobox, x, y, w, h)
    
def poi_statusbox_old(frame, x, y, uptime, subjectno):
    multiple = 0.50
    statusbox_path = "gui\machine\statusbox.tif"
    statusbox = cv2.imread(statusbox_path)

    w = int(576 * multiple)
    h = int(164 * multiple)
    
    cv2.putText(statusbox, "STATUS:", (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(statusbox, "ACTIVE", (150, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(statusbox, "UPTIME: {}".format(uptime), (15, 92), cv2.FONT_HERSHEY_SIMPLEX, 1, (12, 12, 12), 2)
    cv2.putText(statusbox, "SUBJECTS DETECTED: {}".format(subjectno), (15, 135), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return overlayimg(frame, statusbox, x, y, w, h)

def poi_statusbox(frame, x, y, uptime, subjectno):
    multiple = .5
    statusbox_path = "gui\machine\statusbox_new.tif"
    statusbox = cv2.imread(statusbox_path)
   
    w = int(576 * multiple)
    h = int(164 * multiple)


    cv2.putText(statusbox, "o", (30, 33), cv2.FONT_HERSHEY_SIMPLEX, .25, (000, 255, 000), 15)
    cv2.putText(statusbox, "PROGRAM: THE MACHINE", (55, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (220, 220, 220), 3)
    cv2.putText(statusbox, "STATUS:", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 1, (013, 013, 013), 2)
    cv2.putText(statusbox, "ACTIVE", (150, 85), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
    cv2.putText(statusbox, "UPTIME: {}".format(uptime), (15, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(statusbox, "SUBJECTS DETECTED: {}".format(subjectno), (15, 168), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return overlayimg(frame, statusbox, x, y, w, h)

def samaritan_image(frame, x, y, w, h, sub_type):
    assets_path = "gui\\samaritan\\"
    if sub_type == 'ADMIN' or sub_type == 'ANALOG':
        stype = 'irrelevant'
    elif sub_type == 'THREAT':
        stype = 'threat'
    elif sub_type == 'UNKNOWN':
        stype = 'deviant'
    elif sub_type == 'USER':
        stype = 'irrelevant'
    focus_path = assets_path + stype + "_focus.tif"
    # print focus_path
    focus = cv2.imread(focus_path)
    #cv2.imshow('focus', focus)
    x -= 150
    y -= 150
    w += 300
    h += 300
    return overlayimg(frame, focus, x, y, w, h)
