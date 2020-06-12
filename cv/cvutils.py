import numpy as np
import dlib

"""
convert between several formats:
(x, y, x+w, y+h) is coord
(y, x+w, y+h, x) is mcoord (mirror coords)
(x, y, w, h) is bb
[(x, y) (x+w, y+h)] is rect (dlib rectangle)
array([x, y, x+w, ,y+h]) is npcoord
"""

def coord_to_rect(t):
    # take a tuple bounding (startX, startY, endX, endY) and convert it
    # to the format [(x, y), (x+w, y+h)]
    (startX, startY, endX, endY) = t
    rtn = dlib.rectangle(left=startX, top=startY, right=endX, bottom=endY)
    return rtn


def mcoord_to_rect(t):
    # take a tuple bounding (startX, startY, endX, endY) and convert it
    # to the format [(x, y), (x+w, y+h)]
    (top, right, bottom, left) = t
    rtn = dlib.rectangle(left=left, top=top, right=right, bottom=bottom)
    return rtn


def mcoord_to_coord(fr):
    # take a face_recognition package tuple bounding
    # (top, right, bottom, left) and convert it
    # to the format (startX, startY, endX, endY)
    (top, right, bottom, left) = fr
    rtn = (left, top, right, bottom)
    return rtn


def npcoord_to_coord(n):
    # convert np.array() to tuple
    # print "npcoord_to_coord: ", n
    return n[0], n[1], n[2], n[3]

