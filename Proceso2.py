import matplotlib.pyplot as plt 
import numpy as np 
import cv2
import math as mt

#constants
colors = ('b' , 'g' ,'r')

#Implementation
src = cv2.imread('images/imagen2.jpg')
cv2.imshow('Imagen Inicial',src)
cv2.waitKey(0)

# hist = []
# for i in range(0,2,1):
#     hist = [hist, cv2.calcHist([src],[i],None,[256],[0,256])]
#     plt.figure()
#     plt.plot(hist[i+1], color = colors[i])
#     plt.xlim([0, 256])
# plt.show()

hist1 = cv2.calcHist([src],[0],None,[256],[0,256])
plt.figure()
plt.plot(hist1, color = colors[0])
plt.xlim([0, 256])
hist2 = cv2.calcHist([src],[1],None,[256],[0,256])
plt.figure()
plt.plot(hist2, color = colors[1])
plt.xlim([0, 256])
hist3 = cv2.calcHist([src],[2],None,[256],[0,256])
plt.figure()
plt.plot(hist3, color = colors[2])
plt.xlim([0, 256])
plt.show()

b, g, r = cv2.split(src)

# for j in range(0,3,1):
#     Cdf_acc = 0
#     for i in range(0,256,1):
#         Cdf[j,i] = Cdf_acc + hist[j,i]
#     Cdf[j,:] = Cdf[j,:]/max(Cdf[j,:])
#     plt.figure()
#     plt.plot(Cdf[j,:])
# plt.show()

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
plt.figure()
plt.plot(Cdf1)

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
plt.figure()
plt.plot(Cdf2)

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
plt.plot(Cdf3)

plt.show()    

width,height,depth = src.shape

for i in range(0,width,1):
    for j in range(0,height,1):
        if Cdf1[b[i,j]]==1:
            b[i,j] = 255
        else:
            b[i,j] = mt.sqrt(2*(50**2)*mt.log(1/(1-Cdf1[b[i,j]])))
        if Cdf2[g[i,j]]==1:
            g[i,j] = 255
        else:
            g[i,j] = mt.sqrt(2*(50**2)*mt.log(1/(1-Cdf2[g[i,j]])))
        if Cdf3[r[i,j]]==1:
            r[i,j] = 255
        else:
            r[i,j] = mt.sqrt(2*(50**2)*mt.log(1/(1-Cdf3[r[i,j]])))

image_final = cv2.merge((b,g,r))
for i,col in enumerate(['b','g','r']):
    hist = cv2.calcHist([image_final],[i],None,[256],[0,256])
    plt.figure()
    plt.plot(hist, color = col)
    plt.xlim([0, 256])
plt.show()

cv2.imshow('Imagen Final',image_final)
cv2.waitKey(0)

    