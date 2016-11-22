import cv2
import numpy as np
#Harris

filename = 'ts4.jpg'
img1 = cv2.imread(filename) #img1 is input iage after reading


gray = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)#gray is rgb2gray conversion of img1

gray = np.float32(gray)
print(gray.shape)


dst = cv2.cornerHarris(gray, 2, 3, 0.04)#2 is Neighborhood size 3 is 3x3 sobel operator 0.04 threshold for Harris detector free parameter

#result is dilated for marking the corners, not important
dst = cv2.dilate(dst,None)
a=np.array(dst)
x,y=a.shape
print(x,y)

c=dst>0.00001*dst.max()  #threshold to consider harris points further used in gray thresholding

# Threshold for an optimal value, it may vary depending on the image.
img1[dst>0.00001*dst.max()]=[0,0,255]     #provides red color to harris corner

cv2.imshow('dst',img1)

cv2.imwrite('harris1.jpg',img1)     #saving image


cv2.waitKey(0)

q=np.unravel_index(dst.argmax(),dst.shape)
print(q)


#Harris completes here




img = cv2.imread('blur4(k=19).jpg')             #bilateral image from bilateral filter part

gray2 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)    #gray2 is rgb2gray conversion of img

ret, thresh = cv2.threshold(gray2,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)   #threshold considering factors threshold binary inversion + otsu binarization

# noise removal
kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

# sure background area
sure_bg = cv2.dilate(opening,kernel,iterations=3)

# Finding sure foreground area
dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
ret, sure_fg = cv2.threshold(dist_transform,0.9*dist_transform.max(),255,0)

# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)
# Marker labelling
ret, markers = cv2.connectedComponents(sure_fg)

# Add one to all labels so that sure background is not 0, but 1
markers = markers+1

# Now, mark the region of unknown with zero
markers[unknown==255] = 0
markers = cv2.watershed(img,markers)



print "Number of segments:", np.max(markers)   #printing number of markers

a=np.array(markers)
print(a.shape)


img[markers == -1] = [0,0,255]      #giving background boundary red color
cv2.imshow('output',img)
cv2.imwrite('wh1(k=19).jpg',img)
print(markers)
cv2.waitKey(0)
# watershed on whole image completes here


#NOW firstly extracting foreground area
m = cv2.convertScaleAbs(markers)           #On each element of the input array, the function convertScaleAbs performs three
print(m)                                           # operations sequentially: scaling, taking an absolute value, conversion to an unsigned 8-bit type:
print(m.shape)
print(m)
  #using otsu binarization + threshold binary inverse
ret,thresh = cv2.threshold(m,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
res = cv2.bitwise_and(img,img,mask = thresh)     #this is for retrieval of foreground area from not sure background area

cv2.imshow('bitwise',res)
cv2.imwrite('bitwise(k=19).jpg',res)                 #extracted foreground image ie road part

cv2.waitKey(0)
print(res)

x,y,z=res.shape
print(x)
print(y)
print(z)


gray1 = cv2.cvtColor(res,cv2.COLOR_BGR2GRAY)           #converting it into gray

gray1 = np.float32(gray1)

m,n=gray1.shape
print(m)
print(n)

for i in range(1,m):
    for j in range(1,n):
        if gray1[i,j]!=0:
            if(j>=582 and j<=654 and i>=171 and i<=255):
                gray1[i,j]=gray1[i,j]*c[i,j]                     #getting harris (c variable see above) getting harris corner in that bitwise segmented road part
            else:
                gray1[i,j]=0


filename1 = 'ts4.jpg'                                            #original image
img13 = cv2.imread(filename1)                                    # img13 variable

img13[gray1>0]=[0,255,255]                                       #condition to show harris on conditioned part
cv2.imshow('dst',img13)
print(gray1)


cv2.waitKey(0)
b=0
abc=cv2.imread(filename1)
harris_points=[]
for i in range(1,m):
    for j in range(1,n):                                         #getting harris corner and this if condition is we have consider 4 points near
        if gray1[i,j]!=0:                                        # white markers these are those co ordinates
            if(j>=582 and j<=654 and i>=171 and i<=255):
                z=(i,j)
                harris_points.append(z)
                b=b+1

harris_pointS=np.array(harris_points)
print(b)
print(harris_points)
abcd=[]
for i in range(b):                                               #now again playing watershed along each harris point by considering 32x32
    s,d=harris_points[i]                                         #window along each point , applying same watershed that is applied above. Here watershed is applied on extracted part above.
    xc=abc[s-16:s+16,d-16:d+16]
    grayxc = cv2.cvtColor(xc,cv2.COLOR_BGR2GRAY)
    ret, threshxc = cv2.threshold(grayxc,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    # noise removal
    kernelxc = np.ones((3,3),np.uint8)
    openingxc = cv2.morphologyEx(threshxc,cv2.MORPH_OPEN,kernel, iterations = 2)

    # sure background area
    sure_bgxc = cv2.dilate(openingxc,kernelxc,iterations=3)

    # Finding sure foreground area
    dist_transformxc = cv2.distanceTransform(openingxc,cv2.DIST_L2,5)
    ret, sure_fgxc = cv2.threshold(dist_transformxc,0.9*dist_transformxc.max(),255,0)

    # Finding unknown region
    sure_fgxc = np.uint8(sure_fgxc)
    unknown = cv2.subtract(sure_bgxc,sure_fgxc)
    # Marker labelling
    ret, markersxc = cv2.connectedComponents(sure_fgxc)

    # Add one to all labels so that sure background is not 0, but 1
    markersxc = markersxc+1

    # Now, mark the region of unknown with zero
    markersxc[unknown==255] = 0
    markersxc = cv2.watershed(xc,markersxc)




    abc[s-16:s+16,d-16:d+16]=xc

    

    nSegs = np.max(markersxc)
    print(nSegs)


    g=(0,0)
                         #this type for mulitplication
    h=(1,1)

    if(nSegs!=0 and nSegs<=3):                          #considering condition that no of segments must not be zero and one and must be less than 3
        if(nSegs%3==0):                                 # for no. of segments to be 3

                axc=xc[markersxc==1]
                bxc=xc[markersxc==2]
                cxc=xc[markersxc==3]

                if((axc>=91).all() and (axc<=230).all()):               #condition for pixel for all in that segment
                    if((bxc>=90).all() and (bxc<=158).all()):
                        if((cxc>=90).all() and (bxc<=158).all()):
                            abcd.append(h)

                        else:
                            abcd.append(g)
                    else:
                        abcd.append(g)
                elif((axc>=90).all() and (axc<=158).all()):
                    if((bxc>=90).all() and (bxc<=158).all()):
                        if((cxc>=91).all() and (cxc<=255).all()):
                            abcd.append(h)
                        else:
                            abcd.append(g)
                    elif((bxc>=91).all() and (bxc<=230).all()):
                        if((cxc>=90).all() and (cxc<=158).all()):
                            abcd.append(h)
                        else:
                            abcd.append(g)
                    else:
                        abcd.append(g)
                else:
                    abcd.append(g)




        elif(nSegs%3==2):                                                   #condition for no. of segments to be 2

                axc=xc[markersxc==1]
                bxc=xc[markersxc==2]
                print(bxc)
                if((axc>=91).all() and (axc<=255).all()):
                    if((bxc>=90).all() and (bxc<=158).all()):
                        abcd.append(h)
                    else:
                        abcd.append(g)

                elif((axc>=90).all() and (axc<=158).all()):
                    if((bxc>=91).all() and (bxc<=255).all()):
                        abcd.append(h)
                    else:
                        abcd.append(g)

                else:
                    abcd.append(g)

        else:
            abcd.append(g)
    else:
        abcd.append(g)

cvb=np.array(abcd)                     #numpy array
cv2.waitKey(0)



print(abcd)


vbn=harris_pointS*cvb             #multiplying both array as cvb will give result in 0,1 whether it is part of marker or not

print(vbn)


m,n=vbn.shape             #shape of vbn

print(m)
print(n)
for e in range (0,m-1):                           # here m is size(img,1) as we have our numpy array of size mx2
    if(vbn[e,0]==0 and vbn[e,1]==0):v             #checking condition here and getting co-ordinates of harris gray1 image
        c=harris_pointS[e,0];
        d=harris_pointS[e,1];
        gray1[c,d]=0;
        print("true")

    else:
        c=harris_pointS[e,0];
        d=harris_pointS[e,1];
        gray1[c,d]=gray1[c,d]
        print("false")


print(gray1)


img13a=cv2.imread(filename1)                       #again reading filename so not have confusion again by searching


img13a[gray1>0]=[0,255,255]                       #condition for showing harris on real image yellow color as opencv works on bgr vale
cv2.imshow('fnally',img13a)
cv2.imwrite('wedfeedfsddffrgrgdcdce1.jpg',img13a)  #saving our final image
cv2.waitKey(0)
