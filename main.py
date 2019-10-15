import cv2
import matplotlib.pyplot as plt
import numpy as np 

# Constants section
cb = 20
db = 255
cg = 25
dg = 225
cr = 0
dr = 100
bi = 255 
ai = 0
colors = ('b' , 'g' ,'r')
ch = 50
dh = 150
cs = 150
ds = 255
cv = 5
dv = 220

# Inicio del Programa
src = cv2.imread('images/imagen2.jpg')
cv2.imshow("prueba",src)
cv2.waitKey(0)
for i,col in enumerate(['b','g','r']):
    hist = cv2.calcHist([src],[i],None,[256],[0,256])
    plt.figure()
    plt.plot(hist, color = col)
    plt.xlim([0, 256])
plt.show()

b, g, r = cv2.split(src)
width , height, depth = src.shape

for i in range (0, width, 1):
    for j in range (0, height, 1):
        b[i,j] = ((b[i,j] - cb)*(db-cb)/(bi-ai)) + ai
        g[i,j] = ((g[i,j] - cg)*(dg-cg)/(bi-ai)) + ai
        r[i,j] = ((r[i,j] - cr)*(dr-cr)/(bi-ai)) + ai

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
cv2.imshow('Expansion del histograma',image_proc)
# cv2.waitKey(0)
image_hsv = cv2.cvtColor(image_proc, cv2.COLOR_BGR2HSV)
hc, sc, vc = cv2.split(image_hsv)

for i,col in enumerate(['b','g','r']):
    hist = cv2.calcHist([image_hsv],[i],None,[256],[0,256])
    plt.figure()
    plt.plot(hist, color = col)
    plt.xlim([0, 256])
plt.show()

for i in range (0, width, 1):
    for j in range (0, height, 1):
        hc[i,j] = ((hc[i,j] - ch)*(dh-ch)/(bi-ai)) + ai
        sc[i,j] = ((sc[i,j] - cs)*(ds-cs)/(bi-ai)) + ai
        vc[i,j] = ((vc[i,j] - cv)*(dv-cv)/(bi-ai)) + ai

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