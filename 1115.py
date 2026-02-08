# 1115.py
#ref1:https://rdmilligan.wordpress.com/2015/07/31/3d-augmented-reality-using-opencv-and-python/
# Objfile loader    :  objloader.py
# Wavefront OBJ file:         ./data/cube.obj
# material template library:  ./data/cube.mtl

# pip install pygame


import cv2
import numpy as np
np.set_printoptions(precision=2, suppress=True)

#1
from OpenGL.GL import *
from OpenGL.GLU import *

import pygame
from pygame.locals import * 
from objloader import OBJ # objloader.py

# video
cap = cv2.VideoCapture('./data/chess1.wmv')
#cap = cv2.VideoCapture(0)
if (not cap.isOpened()): 
     print('Error opening video')
     import sys
     sys.exit()     
height, width = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
              int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
imageSize = width, height

#2
patternSize = (6, 3)
def FindCornerPoints(src_img, patternSize):
    found, corners = cv2.findChessboardCorners(src_img, patternSize)
    if not found:
        return found, corners

    term_crit=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 10, 0.01)    
    gray = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
    corners = cv2.cornerSubPix(gray, corners, (5,5), (-1,-1), term_crit)
    corners = corners[::-1] # reverse order, in this example, to set origin to (left-upper)
    return found, corners

#3: set world(object) coordinates to Z = 0
xN, yN = patternSize # (6, 3)
mW = np.zeros((xN*yN, 3), np.float32) # (18, 3)
mW[:, :2] = np.mgrid[0:xN, 0:yN].T.reshape(-1, 2) # mW points on Z = 0
mW[:, :2]+= 1 # (1, 1, 0): coord of the start corner point in the pattern


#4: load camera matrix K
with np.load('./data/calib_1104.npz') as X:
    K = X['K']
    dists = X['dists']  
#dists = np.zeros(5)
print("K=\n", K)
print("dists=", dists)

#5: OpenGL camera parameters, opengl_proj
#5-1:
cx = K[0, 2] 
cy = K[1, 2] 
fx = K[0, 0]
fy = K[1, 1]
w, h = imageSize # (width=640, height=480)
near = 0.1 # near plane
far  = 100 #  far plane

#5-2:
opengl_proj = np.array([[2*fx/w, 0.0,  (w - 2*cx)/w, 0.0], 
                        [0.0, 2*fy/h, (-h + 2*cy)/h, 0.0], # y-down,origin: left-upper
                        [0.0,   0.0,   (-far - near)/(far - near), -2.0*far*near/(far-near)],
                        [0.0,   0.0,   -1.0,          0.0] ] )

#opengl_proj[1]*= -1 # y-up,  origin: left-bottom
print("opengl_proj=", opengl_proj)

#6: calculate (left, right, top, bottom) 
F = 10.0   # set arbitrary focal length, use in #8-2
mx = fx/F  # x-pixels per unit
my = fy/F  # y-pixels per unit

#convert image coords to unit of woord coords, use in #6
left  = cx/mx
right = (w-cx)/mx
top   = cy/my
bottom= (h-cy)/my
print("left=",  left)
print("right=", right)
print("top=",   top)
print("bottom=",bottom)

#7
def initGL():
    glClearColor(0.0, 0.0, 0.0, 0.0)

    glMatrixMode(GL_PROJECTION)
    glLoadTransposeMatrixd(opengl_proj) # row-major matrix    
    texture_id = glGenTextures(1)
    return texture_id

#8: texture mapping   
def drawBackgroundTexture():
    glBegin(GL_QUADS)
    glTexCoord2f(0.0, 1.0); glVertex3f(-left,  -bottom, 0.0)
    glTexCoord2f(1.0, 1.0); glVertex3f( right, -bottom, 0.0)
    glTexCoord2f(1.0, 0.0); glVertex3f( right, top, 0.0)
    glTexCoord2f(0.0, 0.0); glVertex3f(-left,  top, 0.0)
    glEnd( )

#9
def displayImage(image):
    
    image_size = image.size

    # create texture   	
    image = cv2.undistort(image, K, dists)
    glBindTexture(GL_TEXTURE_2D, background_texture)    
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    
    # using Pillow
##    from PIL import Image    
##    image = Image.fromarray(image)     
##    ix, iy = image.size[:2]
##    image = image.tobytes('raw', 'BGRX', 0, 1)
##    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, ix, iy,
##                                0, GL_RGBA, GL_UNSIGNED_BYTE,image)    
    # using numpy
    iy, ix = image.shape[:2]
    image = image[...,::-1].copy() # BGR-> RGB
    image = np.frombuffer(image.tobytes(), dtype='uint8', count=image_size)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, ix, iy,
                                0, GL_RGB, GL_UNSIGNED_BYTE,image)    
    # draw background image
    glEnable(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D, background_texture)
    glPushMatrix()
    glTranslatef(0.0,0.0,-F)
    
    drawBackgroundTexture()
    glPopMatrix()
    glDisable(GL_TEXTURE_2D)

#10
def drawCube(bottomFill=True):
    index = [0, 5, 17, 12] # 4-corner index    
    pts0 = mW[index]       # Z = 0

    if bottomFill:
        glBegin(GL_QUADS)
    else:
        glBegin(GL_LINE_LOOP)
    glColor3f(1.0,1.0,0.0)
    for i in range(4):
        glVertex3fv(pts0[i])
    glEnd()

    glColor3f(1.0,0.0,0.0)
    pts2 = pts0.copy()
    pts2[:, 2]= -2        # Z = -2
    glBegin(GL_LINE_LOOP) 
    for i in range(4):
        glVertex3fv(pts2[i])
    glEnd()

    glColor3f(0.0,0.0,1.0)
    glBegin(GL_LINES) 
    for i in range(4):
        glVertex3fv(pts0[i])
        glVertex3fv(pts2[i])
    glEnd()
#11    
def displayAxesCube(view_matrix):
    axis3d = np.float32([[0,0,0], [3,0,0],
                         [0,3,0], [0,0,-3]]).reshape(-1,3)

    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadTransposeMatrixd(view_matrix)
    
    # draw axes
    glLineWidth(5)
    glBegin(GL_LINES)
    glColor3f(0.0, 0.0, 1.0) #blue
    glVertex3fv(axis3d[0])   # X
    glVertex3fv(axis3d[1])

    glColor3f(0.0, 1.0, 0.0) #green 
    glVertex3fv(axis3d[0])   # Y
    glVertex3fv(axis3d[2])

    glColor3f(1.0, 0.0, 0.0) #red
    glVertex3fv(axis3d[0])   # -Z
    glVertex3fv(axis3d[3])
    glEnd()
    glPopMatrix()
    
    # draw cube
    glMatrixMode(GL_MODELVIEW)
    glPushMatrix() 
    glLoadTransposeMatrixd(view_matrix)
    drawCube()
    glPopMatrix()         
    
#12
def getViewMatrix(rvec, tvec):
    tvec = tvec.flatten()
    R = cv2.Rodrigues(rvec)[0]
    # Axes Y, Z of OpenGL are the same as -Y, -Z of OpenCV respectively
    view_matrix = np.array([[ R[0,0], R[0,1], R[0,2], tvec[0]],
                            [-R[1,0],-R[1,1],-R[1,2],-tvec[1]],
                            [-R[2,0],-R[2,1],-R[2,2],-tvec[2]],
                            [ 0.0   , 0.0   , 0.0   , 1.0    ]])    
    return view_matrix

#13:  init pygame and PyOpenGL
pygame.init()
pygame.display.set_caption("Augmented Reality: Pygame, OpenGL and OpenCV")
screen = pygame.display.set_mode(imageSize, DOUBLEBUF | OPENGL|RESIZABLE)

background_texture = initGL()
model = OBJ('./data/cube.obj') # model size: (2 x 2 x 2)
           
#14: main loop   
t = 0 # frame counter
tx = 0.5
ty = 0.5
stop = False
while True:
#14-1: handle keyboard events 
    for e in pygame.event.get():
        if e.type == QUIT:  #  pygame.QUIT      
            stop=True
        elif e.type == KEYDOWN:
            if e.key == K_ESCAPE:
                stop=True
            elif e.key == K_LEFT:
                tx -= 1.0
            elif e.key == K_RIGHT:
                tx += 1.0
            elif e.key == K_UP:
                ty -= 1.0
            elif e.key == K_DOWN:
                ty += 1.0                
#14-2
    ret, frame = cap.read()
    if not ret or stop:
        break
    
    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
    displayImage(frame)

    found, corners = FindCornerPoints(frame, patternSize)
    if not found:
        pygame.display.flip()
        continue
    
#14-3    
    ret, rvec, tvec = cv2.solvePnP(mW, corners, K, dists)
    view_matrix = getViewMatrix(rvec, tvec) 
    displayAxesCube(view_matrix)

#14-4: display obj 
    glPushMatrix()
    glLoadTransposeMatrixd(view_matrix)
    
    glTranslate(tx, ty, -1)  # move on Z = 0
    glScale(0.5, 0.5, 1.0)   # model size: (1 x 1 x 2)
 
    model.render() # glCallList(model.gl_list)    
    glPopMatrix()
   
#14-5
    t += 1
    glColor3f(1.0, 1.0, 1.0) # white
    pygame.display.flip()
    #pygame.time.wait(6000)
             
#15 
pygame.quit()
if cap.isOpened(): cap.release()
 


