#1207.py
from PIL import Image, ImageWin

#1
# img = Image.open("./data/lena.png")
img = Image.open("./data/lena.jpg")

#2
dib = ImageWin.Dib(img)
wnd1 = ImageWin.ImageWindow(dib)

#3
dib2 = ImageWin.Dib(image="RGB", size=(512, 480))
wnd2 = ImageWin.ImageWindow(dib2)

#4
wnd1.mainloop()
wnd2.mainloop()
