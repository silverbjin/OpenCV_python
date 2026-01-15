#0200 binary_image
import cv2 as cv

img_color = cv.imread('./data/box_small.png', cv.IMREAD_COLOR)
img_gray = cv.cvtColor(img_color, cv.COLOR_BGR2GRAY)
ret, img_binary = cv.threshold(img_gray, 150, 255, cv.THRESH_BINARY)

print("img value: ", img_color)
print("type(img): ", type(img_color))
print("img.shape: ", img_color.shape)
cv.waitKey(0)

height,width = img_binary.shape[:2]

f = open('./data/0200_binary.txt', 'w')

for y in range(height):
    for x in range(width):
        print("%3d"%img_binary[y,x], end=" ")
        f.write("%3d " % img_binary[y,x])
    print("")
    f.write("\n")

f.close()
cv.imshow("Binary", img_binary)
cv.waitKey(0)