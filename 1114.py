# 1114.py
#https://github.com/francoisruty/fruty_opencv-opengl-projection-matrix/blob/master/test.py
#https://strawlab.org/2011/11/05/augmented-reality-with-OpenGL/

# pip install PyOpenGL PyOpenGL_accelerate

import cv2
import numpy as np
from OpenGL.GL import *
from OpenGL.GLUT import *

#1:
img1 = cv2.imread('./data/image1.jpg')
img2 = cv2.imread('./data/image2.jpg')
imageSize= (img1.shape[1], img1.shape[0]) # (width, height)
image_select= 1 # img1 
patternSize = (6, 3)       
 
#2: set world(object) coordinates to Z = 0
xN, yN = patternSize # (6, 3)
mW = np.zeros((xN*yN, 3), np.float32) # (18, 3)
mW[:, :2] = np.mgrid[0:xN, 0:yN].T.reshape(-1, 2) # mW points on Z = 0
mW[:, :2] += 1

#3: load calibration parameters
with np.load('./data/calib_1104.npz') as X:
    K, dists, rvecs, tvecs = [X[i] for i in ('K','dists','rvecs','tvecs')]
##dists = np.zeros(5)
print("K=\n", K)
print("dists=", dists)

#4: OpenGL camera parameters, opengl_proj
#4-1:
cx = K[0, 2] 
cy = K[1, 2] 
fx = K[0, 0]
fy = K[1, 1]
w, h = imageSize # (width=640, height=480)
near = 0.1    # near plane
far  = 100.0  #  far plane
#4-2:
opengl_proj = np.array([[2*fx/w, 0.0,  (w - 2*cx)/w, 0.0], 
                        [0.0, 2*fy/h, (-h + 2*cy)/h, 0.0], # Y-down
                        [0.0,   0.0,   (-far - near)/(far - near), -2.0*far*near/(far-near)],
                        [0.0,   0.0,   -1.0,          0.0] ] )

#opengl_proj[1]*= -1 # Y-up 
print("opengl_proj=", opengl_proj)

#5: calculate (left, right, top, bottom) 
F = 10.0   # set arbitrary focal length, use in #8
mx = fx/F  # x-pixels per unit
my = fy/F  # y-pixels per unit

#convert image coords to unit of woord coords, use in #7
left  = cx/mx
right = (w-cx)/mx
top   = cy/my
bottom= (h-cy)/my

print("left=",  left)
print("right=", right)
print("top=",   top)
print("bottom=",bottom)

#6: OpenGL: setup GL_PROJECTION with opengl_proj
def initGL():
    glClearColor(0.0, 0.0, 0.0, 0.0)
    glMatrixMode(GL_PROJECTION)
    glLoadTransposeMatrixd(opengl_proj) # row-major matrix
##    opengl_proj = np.transpose(opengl_proj)
##    glLoadMatrixd(opengl_proj) # column-major matrix
    
##    proj = glGetFloatv(GL_PROJECTION_MATRIX)
##    proj = np.transpose(proj) 
##    print("proj=", proj)
    
    texture_id = glGenTextures(1)
    return texture_id
    
#7: texture mapping   
def drawBackgroundTexture():
    glBegin(GL_QUADS)
    glTexCoord2f(0.0, 1.0); glVertex3f(-left,  -bottom, 0.0)
    glTexCoord2f(1.0, 1.0); glVertex3f( right, -bottom, 0.0)
    glTexCoord2f(1.0, 0.0); glVertex3f( right, top, 0.0)
    glTexCoord2f(0.0, 0.0); glVertex3f(-left,  top, 0.0)
    glEnd( )
    
#8
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

#9
def drawCube(bottomFill=True):
    index = [0, 5, 17, 12] # 4-corner index    
    pts0 = mW[index]       # Z = 0
    print('KDK pts0=',pts0)

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
    
#10    
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
    
#11
def getViewMatrix(rvec, tvec):
    tvec = tvec.flatten()
    R = cv2.Rodrigues(rvec)[0]
    # Axes Y, Z of OpenGL are the same as -Y, -Z of OpenCV respectively
    view_matrix = np.array([[ R[0,0], R[0,1], R[0,2], tvec[0]],
                            [-R[1,0],-R[1,1],-R[1,2],-tvec[1]],
                            [-R[2,0],-R[2,1],-R[2,2],-tvec[2]],
                            [ 0.0   , 0.0   , 0.0   , 1.0    ]])    
    return view_matrix

#12
tx = 0.5
ty = 0.5
def displayFun():
    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)

    #12-1: create texture
    if image_select == 1:
        image = img1
    else:
        image = img2    	
    displayImage(image) 

    #12-2:create view matrix using tvec and rvec
    i = image_select-1
    tvec = tvecs[i].flatten()
    rvec = rvecs[i]
    view_matrix = getViewMatrix(rvec, tvec)
    #print("view_matrix=", view_matrix)

    #12-3: display X, Y, -Z  
    displayAxesCube(view_matrix) 

    #12-4: draw glut cube
    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()    
    glLoadTransposeMatrixd(view_matrix)
    glTranslatef(tx, ty,-1) # move on Z = 0
    glScale(1., 1., 2.0)    # size: (1 x 1 x 2)
   
##    glColor3f(1.0, 1.0, 0.0)
##    glutSolidCube(1) 
    
    glColor3f(0.0, 1.0, 1.0)
    glutWireCube(1) 
    glPopMatrix()

    glColor3f(1.0, 1.0, 1.0) # white
    glutSwapBuffers()

#13: handle keyboard events
def keyFun(key,x,y):
    global image_select

    if key == b'\x1b':
         glutDestroyWindow(win_id)
         #exit(0)
    elif key == b'1':
        image_select = 1
    elif key == b'2':
        image_select = 2        
    glutPostRedisplay() #displayFun()
    
#14: handle arrow keys
def specialKeyFun(key,x,y):
    global tx, ty
    if key == GLUT_KEY_LEFT:
        tx -= 1.0
    elif key == GLUT_KEY_RIGHT:
        tx += 1.0
    elif key == GLUT_KEY_UP:
        ty -= 1.0
    elif key == GLUT_KEY_DOWN:
        ty += 1.0   
    glutPostRedisplay() #displayFun()

#15         
def main():
    global win_id, background_texture
    glutInit()
    glutInitDisplayMode(GLUT_RGBA|GLUT_DOUBLE|GLUT_DEPTH)
    glutInitWindowSize(640, 480)
    # win_id=glutCreateWindow("Augmented Reality: GLUT, OpenGL and OpenCV") 
    # win_id = glutCreateWindow(b"Augmented Reality: GLUT, OpenGL and OpenCV") # for cv2==4.13
    win_id = glutCreateWindow("Augmented Reality: GLUT, OpenGL and OpenCV".encode("ascii")) # for cv2==4.13


    background_texture = initGL()
    glutDisplayFunc(displayFun)
    glutKeyboardFunc(keyFun)
    glutSpecialFunc(specialKeyFun) # arrow key
    glutMainLoop()
if __name__ == "__main__":
    main()
