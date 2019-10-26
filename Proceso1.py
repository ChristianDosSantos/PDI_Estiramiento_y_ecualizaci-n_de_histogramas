import cv2
import matplotlib.pyplot as plt
import numpy as np 
import matplotlib.image as mpimg

# Constants section
bi = 255 
ai = 0
colors = ('b' , 'g' ,'r')
pixPercent = 0.005

# Inicio del Programa
src = cv2.imread('images/imagenpdi1.jpg')

hist1 = cv2.calcHist([src],[0],None,[256],[0,256])
hist2 = cv2.calcHist([src],[1],None,[256],[0,256])
hist3 = cv2.calcHist([src],[2],None,[256],[0,256])

fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 12
fig_size[1] = 9

srcRgb = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)

plt.subplot(221)
plt.imshow(srcRgb)
plt.title('Imagen Original')
plt.ylabel('Vertical pixels')
plt.xlabel('Horizontal pixels')
plt.subplot(222)
plt.plot(hist1, color = colors[0])
plt.title('Histograma Azul')
plt.ylabel('Number of pixels')
plt.xlabel('Intensity')
plt.subplot(223)
plt.plot(hist2, color = colors[1])
plt.title('Histograma Verde')
plt.ylabel('Number of pixels')
plt.xlabel('Intensity')
plt.subplot(224)
plt.plot(hist3, color = colors[2])
plt.title('Histograma Rojo')
plt.ylabel('Number of pixels')
plt.xlabel('Intensity')
plt.suptitle('Imagen inicial y sus histogramas', fontsize=16)

b, g, r = cv2.split(src)
b = np.array(b,dtype = int)
g = np.array(g,dtype = int)
r = np.array(r,dtype = int)
width , height, depth = src.shape

pixTotalStr = int(width*height*pixPercent)
print(pixTotalStr)

pix_accb = 0
pix_accg = 0
pix_accr = 0
for i in range(0,256,1):
    if (pix_accb < pixTotalStr):
        pix_accb += int(hist1[i])
        if pix_accb >= pixTotalStr:
            cb = i
    if (pix_accg < pixTotalStr):
        pix_accg += int(hist2[i])
        if pix_accg >= pixTotalStr:
            cg = i
    if (pix_accr < pixTotalStr):
        pix_accr += int(hist3[i])
        if pix_accr >= pixTotalStr:
            cr = i
    
print(cb)
print(cg)
print(cr)

pix_accb = 0
pix_accg = 0
pix_accr = 0
for i in range(255,-1,-1):
    if (pix_accb < pixTotalStr):
        pix_accb += int(hist1[i])
        if pix_accb >= pixTotalStr:
            db = i
    if (pix_accg < pixTotalStr):
        pix_accg += int(hist2[i])
        if pix_accg >= pixTotalStr:
            dg = i
    if (pix_accr < pixTotalStr):
        pix_accr += int(hist3[i])
        if pix_accr >= pixTotalStr:
            dr = i

print(db)
print(dg)
print(dr)

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

image_proc = cv2.merge((b,g,r))

image_procRgb = cv2.cvtColor(image_proc, cv2.COLOR_BGR2RGB)

plt.figure()
plt.subplot(221)
plt.imshow(image_procRgb)
plt.title('Imagen Procesada')
plt.ylabel('Vertical pixels')
plt.xlabel('Horizontal pixels')
plt.subplot(222)
plt.plot(histb, color = colors[0])
plt.title('Histograma Azul')
plt.ylabel('Number of pixels')
plt.xlabel('Intensity')
plt.subplot(223)
plt.plot(histg, color = colors[1])
plt.title('Histograma Verde')
plt.ylabel('Number of pixels')
plt.xlabel('Intensity')
plt.subplot(224)
plt.plot(histr, color = colors[2])
plt.title('Histograma Rojo')
plt.ylabel('Number of pixels')
plt.xlabel('Intensity')
plt.suptitle('Estiramiento de histogramas en el espacio RGB', fontsize=16)

image_hsv = cv2.cvtColor(image_proc, cv2.COLOR_BGR2HSV)
hc, sc, vc = cv2.split(image_hsv)

hist1 = cv2.calcHist([image_hsv],[0],None,[256],[0,256])
hist2 = cv2.calcHist([image_hsv],[1],None,[256],[0,256])
hist3 = cv2.calcHist([image_hsv],[2],None,[256],[0,256])

plt.figure()
plt.subplot(221)
plt.imshow(image_hsv)
plt.title('Imagen Inicial en HSV')
plt.ylabel('Vertical pixels')
plt.xlabel('Horizontal pixels')
plt.subplot(222)
plt.plot(hist1, color = colors[0])
plt.title('Histograma Hue')
plt.ylabel('Number of pixels')
plt.xlabel('Intensity')
plt.subplot(223)
plt.plot(hist2, color = colors[1])
plt.title('Histograma Saturation')
plt.ylabel('Number of pixels')
plt.xlabel('Intensity')
plt.subplot(224)
plt.plot(hist3, color = colors[2])
plt.title('Histograma Value')
plt.ylabel('Number of pixels')
plt.xlabel('Intensity')
plt.suptitle('Imagen en el espacio HSV', fontsize=16)

pix_acch = 0
pix_accs = 0
pix_accv = 0
for i in range(0,256,1):
    if (pix_acch < pixTotalStr):
        pix_acch += int(hist1[i])
        if pix_acch >= pixTotalStr:
            ch = i
    if (pix_accs < pixTotalStr):
        pix_accs += int(hist2[i])
        if pix_accs >= pixTotalStr:
            cs = i
    if (pix_accv < pixTotalStr):
        pix_accv += int(hist3[i])
        if pix_accv >= pixTotalStr:
            cv = i

print(ch)
print(cs)
print(cv)

pix_acch = 0
pix_accs = 0
pix_accv = 0
for i in range(255,-1,-1):
    if (pix_acch < pixTotalStr):
        pix_acch += int(hist1[i])
        if pix_acch >= pixTotalStr:
            dh = i
    if (pix_accs < pixTotalStr):
        pix_accs += int(hist2[i])
        if pix_accs >= pixTotalStr:
            ds = i
    if (pix_accv < pixTotalStr):
        pix_accv += int(hist3[i])
        if pix_accv >= pixTotalStr:
            dv = i

print(db)
print(dg)
print(dr)

sc = np.array(sc,dtype = int)
vc = np.array(vc,dtype = int)


for i in range (0, width, 1):
    for j in range (0, height, 1):
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

sc = np.array(sc,dtype = np.uint8)
vc = np.array(vc,dtype = np.uint8)

image_proc2 = cv2.merge((hc, sc, vc))

hist1 = cv2.calcHist([image_proc2],[0],None,[256],[0,256])
hist2 = cv2.calcHist([image_proc2],[1],None,[256],[0,256])
hist3 = cv2.calcHist([image_proc2],[2],None,[256],[0,256])

image_proc2Rgb = cv2.cvtColor(image_proc2, cv2.COLOR_BGR2RGB)

plt.figure()
plt.subplot(221)
plt.imshow(image_proc2Rgb)
plt.title('Imagen Procesada en HSV')
plt.ylabel('Vertical pixels')
plt.xlabel('Horizontal pixels')
plt.subplot(222)
plt.plot(hist1, color = colors[0])
plt.title('Histograma Hue')
plt.ylabel('Number of pixels')
plt.xlabel('Intensity')
plt.subplot(223)
plt.plot(hist2, color = colors[1])
plt.title('Histograma Saturation')
plt.ylabel('Number of pixels')
plt.xlabel('Intensity')
plt.subplot(224)
plt.plot(hist3, color = colors[2])
plt.title('Histograma Value')
plt.ylabel('Number of pixels')
plt.xlabel('Intensity')
plt.suptitle('Imagen procesada en el espacio HSV', fontsize=16)

image_final = cv2.cvtColor(image_proc2, cv2.COLOR_HSV2BGR)
cv2.imwrite('images/proceso1.jpg', image_final)

hist1 = cv2.calcHist([image_final],[0],None,[256],[0,256])
hist2 = cv2.calcHist([image_final],[1],None,[256],[0,256])
hist3 = cv2.calcHist([image_final],[2],None,[256],[0,256])

image_finalRgb = cv2.cvtColor(image_final, cv2.COLOR_BGR2RGB)

plt.figure()
plt.subplot(221)
plt.imshow(image_finalRgb)
plt.title('Imagen Final en RGB')
plt.ylabel('Vertical pixels')
plt.xlabel('Horizontal pixels')
plt.subplot(222)
plt.plot(hist1, color = colors[0])
plt.title('Histograma Azul')
plt.ylabel('Number of pixels')
plt.xlabel('Intensity')
plt.subplot(223)
plt.plot(hist2, color = colors[1])
plt.title('Histograma Verde')
plt.ylabel('Number of pixels')
plt.xlabel('Intensity')
plt.subplot(224)
plt.plot(hist3, color = colors[2])
plt.title('Histograma Rojo')
plt.ylabel('Number of pixels')
plt.xlabel('Intensity')
plt.suptitle('Imagen Final del Proceso 1', fontsize=16)

plt.show()
