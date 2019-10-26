import matplotlib.pyplot as plt 
import numpy as np 
import cv2
import math as mt

#Constants
colors = ('b' , 'g' ,'r')

#Implementation
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

Cdf_acc = 0
for i in range(0,256,1):
    if (i == 0):
        Cdf_acc += int(hist1[i])
        Cdf1 = [Cdf_acc]
    else:
        Cdf_acc += int(hist1[i])
        Cdf1.append(Cdf_acc)
maxcdf = max(Cdf1)
Cdf1[:] = [x/maxcdf for x in Cdf1]


Cdf_acc = 0
for i in range(0,256,1):
    if (i == 0):
        Cdf_acc += int(hist2[i])
        Cdf2 = [Cdf_acc]
    else:
        Cdf_acc += int(hist2[i])
        Cdf2.append(Cdf_acc)
maxcdf = max(Cdf2)
Cdf2[:] = [x/maxcdf for x in Cdf2]


Cdf_acc = 0
for i in range(0,256,1):
    if (i == 0):
        Cdf_acc += int(hist3[i])
        Cdf3 = [Cdf_acc]
    else:
        Cdf_acc += int(hist3[i])
        Cdf3.append(Cdf_acc)
maxcdf = max(Cdf3)
Cdf3[:] = [x/maxcdf for x in Cdf3]

plt.figure()
plt.subplot(221)
plt.plot(Cdf1, color = colors[0])
plt.title('CDF normalizada del canal B')
plt.ylabel('Valor de la CDF')
plt.xlabel('Intensidad del pixel')
plt.subplot(222)
plt.plot(Cdf2, color = colors[1])
plt.title('CDF normalizada del canal G')
plt.ylabel('Valor de la CDF')
plt.xlabel('Intensidad del pixel')
plt.subplot(223)
plt.plot(Cdf3, color = colors[2])
plt.title('CDF normalizada del canal R')
plt.ylabel('Valor de la CDF')
plt.xlabel('Intensidad del pixel')
plt.suptitle('CDF de cada canal', fontsize=16)    

width,height,depth = src.shape

for i in range(0,width,1):
    for j in range(0,height,1):
        if Cdf1[b[i,j]]==1:
            b[i,j] = 255
        else:
            b[i,j] = mt.sqrt(2*(70**2)*mt.log(1/(1-Cdf1[b[i,j]])))
            if b[i,j] < 0:
                b[i,j] = 0
            elif b[i,j] > 255:
                b[i,j] = 255
        if Cdf2[g[i,j]]==1:
            g[i,j] = 255
        else:
            g[i,j] = mt.sqrt(2*(70**2)*mt.log(1/(1-Cdf2[g[i,j]])))
            if g[i,j] < 0:
                g[i,j] = 0
            elif g[i,j] > 255:
                g[i,j] = 255
        if Cdf3[r[i,j]]==1:
            r[i,j] = 255
        else:
            r[i,j] = mt.sqrt(2*(70**2)*mt.log(1/(1-Cdf3[r[i,j]])))
            if r[i,j] < 0:
                r[i,j] = 0
            elif r[i,j] > 255:
                r[i,j] = 255


b = np.array(b,dtype = np.uint8)
g = np.array(g,dtype = np.uint8)
r = np.array(r,dtype = np.uint8)

image_final = cv2.merge((b,g,r))
cv2.imwrite('images/proceso2.jpg', image_final)

hist1 = cv2.calcHist([image_final],[0],None,[256],[0,256])
hist2 = cv2.calcHist([image_final],[1],None,[256],[0,256])
hist3 = cv2.calcHist([image_final],[2],None,[256],[0,256])

image_finalRgb = cv2.cvtColor(image_final, cv2.COLOR_BGR2RGB)

plt.figure()
plt.subplot(221)
plt.imshow(image_finalRgb)
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
plt.suptitle('Imagen final y sus histogramas', fontsize=16)

plt.show()


    