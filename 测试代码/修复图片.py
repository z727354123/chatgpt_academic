import cv2
import numpy as np

def fill3box(qr):
    qr[0:7,0:7] = 1
    qr[14:21,14:21] = 1
    qr[14:21,0:7] = 1
    qr[0,0:6]=0
    qr[0:6,0]=0
    qr[0:6,6]=0
    qr[6,0:7]=0
    qr[2:5,2:5]=0
    qr[14:21,14:21] = qr[0:7,0:7]
    qr[14:21,0:7] = qr[0:7,0:7]
    return qr

im = cv2.imread('/Users/judy/Downloads/123_png.png')
im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
im = cv2.resize(im,(210,210))

im = 1-((im - im.min())/(im.max()-im.min())) #normalize and adjust contrast
avg=np.average(im)
qr = np.ones((21,21))
w,h = im.shape[:2]
im_orig = im.copy()

im[im<avg]=0#binarize
im[im>avg]=1
for y in range(21):
    for x in range(21):
        x1,y1 = (round(x*w/21),round(y*h/21))
        x2,y2 = (round(x1+10),round(y1+10))

        im_box = im[y1:y2,x1:x2]
        if np.average(im_box)<0.6 and qr[y,x]!=0:#0.6 need tweaking
            qr[y,x]=0

qr = fill3box(qr) #clean up 3 box areas as they need to be fixed
# debug visualization
for x in range(21):
    p1 = (round(x*w/21),0)
    p2 = (round(x*w/21),h)
    cv2.line(im_orig,p1,p2,(255),1)

for y in range(21):
    p1 = (0,round(y*h/21))
    p2 = (w,round(y*h/21))
    cv2.line(im_orig,p1,p2,(255),1)

qr = cv2.resize(qr,(210,210),interpolation=cv2.INTER_NEAREST)

im = (im*255).astype(np.uint8)
qr= (qr*255).astype(np.uint8)
im_orig= (im_orig*255).astype(np.uint8)

cv2.imwrite('im.png',im)
cv2.imwrite('qr.png',qr)
cv2.imwrite('im_orig.png',im_orig)