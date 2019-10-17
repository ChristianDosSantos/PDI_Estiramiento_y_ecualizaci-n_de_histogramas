import cv2
import matplotlib.pyplot as plt
import numpy as np 

# Constants section
cb = 72
db = 212
cg = 83
dg = 226
cr = 10
dr = 92
bi = 255 
ai = 0
colors = ('b' , 'g' ,'r')
ch = 0
dh = 175
cs = 0
ds = 255
cv = 10
dv = 255

# Inicio del Programa
src = cv2.imread('images/imagenpdi1.jpg')
cv2.imshow("prueba",src)
# cv2.waitKey(0)
for i,col in enumerate(['b','g','r']):
    hist = cv2.calcHist([src],[i],None,[256],[0,256])
    plt.figure()
    plt.plot(hist, color = col)
    plt.xlim([0, 256])
plt.show()

b, g, r = cv2.split(src)
b = np.array(b,dtype = int)
g = np.array(g,dtype = int)
r = np.array(r,dtype = int)
width , height, depth = src.shape

for i in range (0, width, 1):
    for j in range (0, height, 1):
        b[i,j] = ((b[i,j] - cb)*(bi-ai)//(db-cb)) + ai
        if b[i,j] > 255:
            b[i,j] = 255
        elif b[i,j] < 0:
            b[i,j] = 0
        g[i,j] = ((g[i,j] - cg)*(bi-ai)//(dg-cg)) + ai
        if g[i,j] > 255:
            g[i,j] = 255
        elif g[i,j] < 0:
            g[i,j] = 0
        r[i,j] = ((r[i,j] - cr)*(bi-ai)//(dr-cr)) + ai
        if r[i,j] > 255:
            r[i,j] = 255
        elif r[i,j] < 0:
            r[i,j] = 0

b = np.array(b,dtype = np.uint8)
g = np.array(g,dtype = np.uint8)
r = np.array(r,dtype = np.uint8)

histb = cv2.calcHist([b], [0], None, [256], [0, 256])
histg = cv2.calcHist([g], [0], None, [256], [0, 256])
histr = cv2.calcHist([r], [0], None, [256], [0, 256])

plt.figure()
plt.plot(histb, color = colors[0])
plt.figure()
plt.plot(histg, color = colors[1])
plt.figure()
plt.plot(histr, color = colors[2])
plt.show()

image_proc = cv2.merge((b,g,r))
cv2.imshow('Expansion del histograma BGR',image_proc)
# cv2.waitKey(0)
image_hsv = cv2.cvtColor(image_proc, cv2.COLOR_BGR2HSV)
hc, sc, vc = cv2.split(image_hsv)

for i,col in enumerate(['b','g','r']):
    hist = cv2.calcHist([image_hsv],[i],None,[256],[0,256])
    plt.figure()
    plt.plot(hist, color = col)
    plt.xlim([0, 256])
plt.show()

hc = np.array(hc,dtype = int)
sc = np.array(sc,dtype = int)
vc = np.array(vc,dtype = int)

for i in range (0, width, 1):
    for j in range (0, height, 1):
        hc[i,j] = ((hc[i,j] - ch)*(bi-ai)/(dh-ch)) + ai
        if hc[i,j] > 255:
            hc[i,j] = 255
        elif hc[i,j] < 0:
            hc[i,j] = 0
        sc[i,j] = ((sc[i,j] - cs)*(bi-ai)/(ds-cs)) + ai
        if sc[i,j] > 255:
            sc[i,j] = 255
        elif sc[i,j] < 0:
            sc[i,j] = 0
        vc[i,j] = ((vc[i,j] - cv)*(bi-ai)/(dv-cv)) + ai
        if vc[i,j] > 255:
            vc[i,j] = 255
        elif vc[i,j] < 0:
            vc[i,j] = 0

hc = np.array(hc,dtype = np.uint8)
sc = np.array(sc,dtype = np.uint8)
vc = np.array(vc,dtype = np.uint8)

image_proc2 = cv2.merge((hc, sc, vc))

for i,col in enumerate(['b','g','r']):
    hist = cv2.calcHist([image_proc2],[i],None,[256],[0,256])
    plt.figure()
    plt.plot(hist, color = col)
    plt.xlim([0, 256])
plt.show()

cv2.imshow('Expansion del histograma HSV',image_proc2)
# cv2.waitKey(0)

image_final = cv2.cvtColor(image_proc2, cv2.COLOR_HSV2BGR)
cv2.imshow('Imagen Final',image_final)
cv2.waitKey(0)