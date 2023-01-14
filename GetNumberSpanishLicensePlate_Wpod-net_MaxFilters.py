# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 18:01:52 2023

@author: Alfonso Blanco Garcia taking wpod-net from
    https://github.com/quangnhat185/Plate_detect_and_recognize
    https://medium.com/@quangnhatnguyenle/detect-and-recognize-vehicles-license-plate-with-machine-learning-and-python-part-1-detection-795fda47e922
    
"""
import cv2
import numpy as np

from local_utils import detect_lp
from os.path import splitext, basename
from keras.models import model_from_json

import os
import re
import imutils
import pytesseract

# Test images
dirname= "test6Training\\images"

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

#X_resize=220
#Y_resize=70

###################################################

from skimage.transform import radon

import numpy
from numpy import  mean, array, blackman, sqrt, square
from numpy.fft import rfft

try:
    # More accurate peak finding from
    # https://gist.github.com/endolith/255291#file-parabolic-py
    from parabolic import parabolic

    def argmax(x):
        return parabolic(x, numpy.argmax(x))[0]
except ImportError:
    from numpy import argmax


def GetRotationImage(image):

   
    I=image
    I = I - mean(I)  # Demean; make the brightness extend above and below zero
    
    
    # Do the radon transform and display the result
    sinogram = radon(I)
   
    
    # Find the RMS value of each row and find "busiest" rotation,
    # where the transform is lined up perfectly with the alternating dark
    # text and white lines
      
    # rms_flat does no exist in recent versions
    #r = array([mlab.rms_flat(line) for line in sinogram.transpose()])
    r = array([sqrt(mean(square(line))) for line in sinogram.transpose()])
    rotation = argmax(r)
    #print('Rotation: {:.2f} degrees'.format(90 - rotation))
    #plt.axhline(rotation, color='r')
    
    # Plot the busy row
    row = sinogram[:, rotation]
    N = len(row)
    
    # Take spectrum of busy row and find line spacing
    window = blackman(N)
    spectrum = rfft(row * window)
    
    frequency = argmax(abs(spectrum))
   
    return rotation, spectrum, frequency

#####################################################################


def Detect_Spanish_LicensePlate(Text):
    
    if len(Text) != 7: return -1
    if (Text[0] < "0" or Text[0] > "9" ) : return -1 
    if (Text[1] < "0" or Text[2] > "9" ) : return -1   
    if (Text[2] < "0" or Text[2] > "9" ) : return -1   
    if (Text[3] < "0" or Text[3] > "9" ) : return -1     
    if (Text[4] < "A" or Text[4] > "Z" ) : return -1 
    if (Text[5] < "A" or Text[5] > "Z" ) : return -1 
    if (Text[6] < "A" or Text[6] > "Z" ) : return -1 
    return 1

def ProcessText(text):
    
    if len(text)  > 7:
       text=text[len(text)-7:] 
    if Detect_Spanish_LicensePlate(text)== -1: 
       return ""
    else:
       return text
   
def ApendTabLicensesFounded (TabLicensesFounded, ContLicensesFounded, text):
    
    SwFounded=0
    for i in range( len(TabLicensesFounded)):
        if text==TabLicensesFounded[i]:
            ContLicensesFounded[i]=ContLicensesFounded[i]+1
            SwFounded=1
            break
    if SwFounded==0:
       TabLicensesFounded.append(text) 
       ContLicensesFounded.append(1)
    return TabLicensesFounded, ContLicensesFounded   

########################################################################
def loadimagesRoboflow (dirname):
 #########################################################################
 # adapted from:
 #  https://www.aprendemachinelearning.com/clasificacion-de-imagenes-en-python/
 # by Alfonso Blanco García
 ########################################################################  
     imgpath = dirname + "\\"
     
     images = []
     Licenses=[]
     
     
     print("Reading imagenes from ",imgpath)
     NumImage=-2
     
     Cont=0
     for root, dirnames, filenames in os.walk(imgpath):
         
         
         NumImage=NumImage+1
         
         for filename in filenames:
             
             if re.search("\.(jpg|jpeg|png|bmp|tiff)$", filename):
                 
                 
                 filepath = os.path.join(root, filename)
                 License=filename[:len(filename)-4]
                 
                 # Spanish license plate is NNNNAAA
                 if Detect_Spanish_LicensePlate(License)== -1: continue
                 
                 image = cv2.imread(filepath)
                 
                 #Color Balance
                #https://blog.katastros.com/a?ID=01800-4bf623a1-3917-4d54-9b6a-775331ebaf05
                
                 img = image
                    
                 r, g, b = cv2.split(img)
                
                 r_avg = cv2.mean(r)[0]
                
                 g_avg = cv2.mean(g)[0]
                
                 b_avg = cv2.mean(b)[0]
                
                 
                 # Find the gain occupied by each channel
                
                 k = (r_avg + g_avg + b_avg)/3
                
                 kr = k/r_avg
                
                 kg = k/g_avg
                
                 kb = k/b_avg
                
                 
                 r = cv2.addWeighted(src1=r, alpha=kr, src2=0, beta=0, gamma=0)
                
                 g = cv2.addWeighted(src1=g, alpha=kg, src2=0, beta=0, gamma=0)
                
                 b = cv2.addWeighted(src1=b, alpha=kb, src2=0, beta=0, gamma=0)
                
                 
                 balance_img = cv2.merge([b, g, r])
                 
                 image=balance_img
                              
                   
                 images.append(image)
                 Licenses.append(License)
                
                 Cont+=1
     
     return images, Licenses
def load_model(path):
    try:
        path = splitext(path)[0]
        with open('%s.json' % path, 'r') as json_file:
            model_json = json_file.read()
        model = model_from_json(model_json, custom_objects={})
        model.load_weights('%s.h5' % path)
        print("Loading model successfully...")
        return model
    except Exception as e:
        print(e)
        
wpod_net_path = "wpod-net.json"
wpod_net = load_model(wpod_net_path)

def preprocess_image(image_path,resize=False):
    #img = cv2.imread(image_path)
    img=image_path
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255
    if resize:
        img = cv2.resize(img, (224,224))
    return img


images, Licenses=loadimagesRoboflow (dirname)
print("Found %i images..."%(len(images)))

# If there is no plate founded, the program would warn you with an error 
#“No License plate is founded!”. In this case, try to increase Dmin value
# to adjust the boundary dimension.
#def get_plate(image_path, Dmax=608, Dmin=256):
def get_plate(image_path, Dmax, Dmin):
    vehicle = preprocess_image(image_path)
    ratio = float(max(vehicle.shape[:2])) / min(vehicle.shape[:2])
    side = int(ratio * Dmin)
    bound_dim = min(side, Dmax)
    _ , LpImg, _, cor = detect_lp(wpod_net, vehicle, bound_dim, lp_threshold=0.5)
    return LpImg, cor

def get_license(img):
    Dmax=608
    Dmin=256
    while Dmin < 600:
        if Dmin > 500:
            return []
           
        LpImg,cor = get_plate(img, Dmax, Dmin)
        #print("Detect %i plate(s) in"%len(LpImg),splitext(basename(test_image))[0])
        #print("Coordinate of plate(s) in image: \n", cor)
        if len(LpImg)> 0: 
         #cv2.imshow(Licenses[i], LpImg[0])
         #cv2.waitKey(0)
         return LpImg[0]
         break
         #cv2.imshow(Licenses[i], LpImg[0])
         #cv2.waitKey(0)
         
        else:
           Dmin=Dmin+50
           

def ThresholdStable(image):
    # -*- coding: utf-8 -*-
    """
    Created on Fri Aug 12 21:04:48 2022
    Author: Alfonso Blanco García
    
    Looks for the threshold whose variations keep the image STABLE
    (there are only small variations with the image of the previous 
     threshold).
    Similar to the method followed in cv2.MSER
    https://datasmarts.net/es/como-usar-el-detector-de-puntos-clave-mser-en-opencv/https://felipemeganha.medium.com/detecting-handwriting-regions-with-opencv-and-python-ff0b1050aa4e
    """
  
    thresholds=[]
    Repes=[]
    Difes=[]
    
    gray=image 
    grayAnt=gray

    ContRepe=0
    threshold=0
    for i in range (255):
        
        ret, gray1=cv2.threshold(gray,i,255,  cv2.THRESH_BINARY)
        Dife1 = grayAnt - gray1
        Dife2=np.sum(Dife1)
        if Dife2 < 0: Dife2=Dife2*-1
        Difes.append(Dife2)
        if Dife2<22000: # Case only image of license plate
        #if Dife2<60000:    
            ContRepe=ContRepe+1
            
            threshold=i
            grayAnt=gray1
            continue
        if ContRepe > 0:
            
            thresholds.append(threshold) 
            Repes.append(ContRepe)  
        ContRepe=0
        grayAnt=gray1
    thresholdMax=0
    RepesMax=0    
    for i in range(len(thresholds)):
        #print ("Threshold = " + str(thresholds[i])+ " Repeticiones = " +str(Repes[i]))
        if Repes[i] > RepesMax:
            RepesMax=Repes[i]
            thresholdMax=thresholds[i]
            
    #print(min(Difes))
    #print ("Threshold Resultado= " + str(thresholdMax)+ " Repeticiones = " +str(RepesMax))
    return thresholdMax

 
 
# Copied from https://learnopencv.com/otsu-thresholding-with-opencv/ 
def OTSU_Threshold(image):
# Set total number of bins in the histogram

    bins_num = 256
    
    # Get the image histogram
    
    hist, bin_edges = np.histogram(image, bins=bins_num)
   
    # Get normalized histogram if it is required
    
    #if is_normalized:
    
    hist = np.divide(hist.ravel(), hist.max())
    
     
    
    # Calculate centers of bins
    
    bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2.
    
    
    # Iterate over all thresholds (indices) and get the probabilities w1(t), w2(t)
    
    weight1 = np.cumsum(hist)
    
    weight2 = np.cumsum(hist[::-1])[::-1]
   
    # Get the class means mu0(t)
    
    mean1 = np.cumsum(hist * bin_mids) / weight1
    
    # Get the class means mu1(t)
    
    mean2 = (np.cumsum((hist * bin_mids)[::-1]) / weight2[::-1])[::-1]
    
    inter_class_variance = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
    
    # Maximize the inter_class_variance function val
    
    index_of_max_val = np.argmax(inter_class_variance)
    
    threshold = bin_mids[:-1][index_of_max_val]
    
    #print("Otsu's algorithm implementation thresholding result: ", threshold)
    return threshold

#########################################################################
def ApplyCLAHE(gray):
#https://towardsdatascience.com/image-enhancement-techniques-using-opencv-and-python-9191d5c30d45
    
    gray_img_eqhist=cv2.equalizeHist(gray)
    hist=cv2.calcHist(gray_img_eqhist,[0],None,[256],[0,256])
    clahe=cv2.createCLAHE(clipLimit=200,tileGridSize=(3,3))
    gray_img_clahe=clahe.apply(gray_img_eqhist)
    return gray_img_clahe



#########################################################################
def FindLicenseNumber (gray,  License, x_resize, y_resize, \
                       Resize_xfactor, Resize_yfactor):
#########################################################################

   
    
    #https://stackoverflow.com/questions/65718126/adaptive-threshold-error-215assertion-failed-src-type-cv-8uc1-in-funct
    from keras.preprocessing import image
    gray=gray*255
   
    gray = image.img_to_array(gray, dtype='uint8')
    gray = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)
    
    #print(gray)
    
    X_resize=x_resize
    Y_resize=y_resize
    print(gray.shape) 
    Resize_xfactor=1.5
    Resize_yfactor=1.5
    
    gray=cv2.resize(gray,None,fx=Resize_xfactor,fy=Resize_yfactor,interpolation=cv2.INTER_CUBIC)
    
    #gray = cv2.resize(gray, (X_resize,Y_resize), interpolation = cv2.INTER_AREA)
    #cv2.imshow("Gray", gray)
    #cv2.waitKey(0)
    """
    rotation, spectrum, frquency =GetRotationImage(gray)
    rotation=90 - rotation
    #print("Car" + str(NumberImageOrder) + " Brillo : " +str(SumBrightnessLic) +   
    #      " Desviacion : " + str(DesvLic))
    if (rotation > 0 and rotation < 30)  or (rotation < 0 and rotation > -30):
      
        gray=imutils.rotate(gray,angle=rotation)
    """
    TabLicensesFounded=[]
    ContLicensesFounded=[]
    
    TotHits=0
    ##############################################################################
    # https://perez-aids.medium.com/introduction-to-image-processing-part-2-image-enhancement-1135a2198793
    # 
    from skimage import img_as_ubyte
   
    # Contrast Stretching
    from skimage.exposure import rescale_intensity
    dark_image_intensity = img_as_ubyte(gray)
    dark_image_contrast = rescale_intensity(dark_image_intensity ,
                      in_range=tuple(np.percentile(dark_image_intensity , (2, 90))))  
   
    text = pytesseract.image_to_string(dark_image_contrast, lang='eng',  \
      config='--psm 6 --oem 3')
    text = ''.join(char for char in text if char.isalnum())
    text=ProcessText(text)
    if ProcessText(text) != "":
    #if Detect_Spanish_LicensePlate(text)== 1:
           TabLicensesFounded, ContLicensesFounded =ApendTabLicensesFounded (TabLicensesFounded, ContLicensesFounded, text)   
           if text==Licenses[i]:
              print(text + "  Hit with Filter Contrast Stretching" )
              TotHits=TotHits+1
           else:
               print(Licenses[i] + " detected with Filter Contrast Stretching "+ text) 
   ################################################
    #img=gray * 255
    #img = gray.astype("uint8")
    img = cv2.GaussianBlur(gray,(3,3),0)
    thresh = cv2.adaptiveThreshold(img, 255, 1, 1, 11, 2) 
    text = pytesseract.image_to_string(thresh, lang='eng',  \
      config='--psm 6 --oem 3')
    text = ''.join(char for char in text if char.isalnum()) 
   
    text=ProcessText(text)
    if ProcessText(text) != "":
    #if Detect_Spanish_LicensePlate(text)== 1:
           TabLicensesFounded, ContLicensesFounded =ApendTabLicensesFounded (TabLicensesFounded, ContLicensesFounded, text)   
           if text==Licenses[i]:
              print(text + "  Hit with FILTRO KAGGLE" )
              TotHits=TotHits+1
           else:
               print(Licenses[i] + " detected with FILTRO KAGGLE "+ text) 
   
    #################################################################
    blur = cv2.GaussianBlur(gray, (7,7), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    
    # Morph open to remove noise and invert image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    invert = 255 - opening

    # Perform text extraction
    text = pytesseract.image_to_string(invert, lang='eng',  \
      config='--psm 6 --oem 3')
    text = ''.join(char for char in text if char.isalnum()) 
    
    text=ProcessText(text)
    if ProcessText(text) != "":
    #if Detect_Spanish_LicensePlate(text)== 1:
           TabLicensesFounded, ContLicensesFounded =ApendTabLicensesFounded (TabLicensesFounded, ContLicensesFounded, text)   
           if text==Licenses[i]:
              print(text + "  Hit with MORPH" )
              TotHits=TotHits+1
           else:
               print(Licenses[i] + " detected with MORPH "+ text) 
    
    ###################################################
    #gray1 = cv2.convertScaleAbs(gray, alpha=(255.0))
    
    # convert to grayscale and blur the image
    #gray1 = cv2.cvtColor(gray1, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(7,7),0)
    
    # Applied inversed thresh_binary 
    binary = cv2.threshold(blur, 180, 255,
                         cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    ## Applied dilation 
    kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thre_mor = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel3)
    #   Otsu's thresholding
   
  
    text = pytesseract.image_to_string(binary, lang='eng',  \
      config='--psm 6 --oem 3')
    text = ''.join(char for char in text if char.isalnum()) 
    text=ProcessText(text)
    if ProcessText(text) != "":
    
        ApendTabLicensesFounded (TabLicensesFounded, ContLicensesFounded, text)
        if text==Licenses[i]:
            print(text + "  Hit with QuangNeguyen binary" )
            TotHits=TotHits+1
        else:
            print(Licenses[i] + " detected with QuangNeguyen binary "+ text)
    
    text = pytesseract.image_to_string(thre_mor, lang='eng',  \
      config='--psm 6 --oem 3')
    text = ''.join(char for char in text if char.isalnum()) 
    text=ProcessText(text)
    if ProcessText(text) != "":
    
        ApendTabLicensesFounded (TabLicensesFounded, ContLicensesFounded, text)
        if text==Licenses[i]:
            print(text + "  Hit with QuangNeguyen thremor" )
            TotHits=TotHits+1
        else:
            print(Licenses[i] + " detected with QuangNeguyenas tremor "+ text)
     
    gray_img_clahe=ApplyCLAHE(gray)
    
    th=OTSU_Threshold(gray_img_clahe)
    max_val=255
    
    ret, o3 = cv2.threshold(gray_img_clahe, th, max_val, cv2.THRESH_TOZERO)
    
    text = pytesseract.image_to_string(o3, lang='eng',  \
      config='--psm 6 --oem 3')
    text = ''.join(char for char in text if char.isalnum()) 
    text=ProcessText(text)
    if ProcessText(text) != "":
    #if Detect_Spanish_LicensePlate(text)== 1:
            TabLicensesFounded, ContLicensesFounded =ApendTabLicensesFounded (TabLicensesFounded, ContLicensesFounded, text)   
            if text==Licenses[i]:
               print(text + "  Hit with CLAHE and THRESH_TOZERO" )
               #TotHits=TotHits+1
            else:
                print(Licenses[i] + " detected as "+ text) 
    
    #   Otsu's thresholding
    ret2,gray1 = cv2.threshold(gray,0,255,cv2.THRESH_TRUNC+cv2.THRESH_OTSU)
  
    text = pytesseract.image_to_string(gray1, lang='eng',  \
      config='--psm 6 --oem 3')
    text = ''.join(char for char in text if char.isalnum()) 
    text=ProcessText(text)
    if ProcessText(text) != "":
    
        ApendTabLicensesFounded (TabLicensesFounded, ContLicensesFounded, text)
        if text==Licenses[i]:
            print(text + "  Hit with Otsu's thresholding of cv2 and THRESH_TRUNC" )
            TotHits=TotHits+1
        else:
            print(Licenses[i] + " detected as "+ text)
    
    threshold=ThresholdStable(gray)
    ret, gray1=cv2.threshold(gray,threshold,255,  cv2.THRESH_TRUNC) 
   
    text = pytesseract.image_to_string(gray1, lang='eng',  \
      config='--psm 6 --oem 3')
    text = ''.join(char for char in text if char.isalnum()) 
    text=ProcessText(text)
    if ProcessText(text) != "":    
    
        ApendTabLicensesFounded (TabLicensesFounded, ContLicensesFounded, text)    
        if text==Licenses[i]:
            print(text + "  Hit with Stable and THRESH_TRUNC" )
            TotHits=TotHits+1
        else:
            print(Licenses[i] + " detected as "+ text)         
        
    ret, gray1=cv2.threshold(gray,threshold,255,  cv2.THRESH_TOZERO) 
    
    text = pytesseract.image_to_string(gray1, lang='eng',  \
      config='--psm 6 --oem 3')
    text = ''.join(char for char in text if char.isalnum()) 
    text=ProcessText(text)
    if ProcessText(text) != "":
  
        ApendTabLicensesFounded (TabLicensesFounded, ContLicensesFounded, text)   
        if text==Licenses[i]:
           print(text + "  Hit with Stable and THRESH_TOZERO" )
           TotHits=TotHits+1
        else:
            print(Licenses[i] + " detected as "+ text) 
        
     
    ####################################################
    # experimental formula based on the brightness
    # of the whole image 
    ####################################################
    
    SumBrightness=np.sum(gray)  
    threshold=(SumBrightness/177600.00) 
    
    #####################################################
  
    ret, gray1=cv2.threshold(gray,threshold,255,  cv2.THRESH_TOZERO)
   
    text = pytesseract.image_to_string(gray1, lang='eng',  \
      config='--psm 6 --oem 3')
    text = ''.join(char for char in text if char.isalnum()) 
    text=ProcessText(text)
    if ProcessText(text) != "":
   
        ApendTabLicensesFounded (TabLicensesFounded, ContLicensesFounded, text)   
        if text==Licenses[i]:
           print(text + "  Hit with Brightness and THRESH_TOZERO" )
           TotHits=TotHits+1
        else:
            print(Licenses[i] + " detected as "+ text)
     
    ret, gray1=cv2.threshold(gray,threshold,255,  cv2.THRESH_OTSU)
    
    text = pytesseract.image_to_string(gray1, lang='eng',  \
      config='--psm 6 --oem 3')
    text = ''.join(char for char in text if char.isalnum()) 
    text=ProcessText(text)
    if ProcessText(text) != "":
   
        ApendTabLicensesFounded (TabLicensesFounded, ContLicensesFounded, text)   
        if text==Licenses[i]:
           print(text + "  Hit with Brightness and THRESH_OTSU" )
           TotHits=TotHits+1
        else:
            print(Licenses[i] + " detected as "+ text) 
     
   
    for z in range(4,8):
    
       kernel = np.array([[0,-1,0], [-1,z,-1], [0,-1,0]])
       gray1 = cv2.filter2D(gray, -1, kernel)
              
      
       text = pytesseract.image_to_string(gray1, lang='eng',  \
          config='--psm 6 --oem 3')
       text = ''.join(char for char in text if char.isalnum()) 
       text=ProcessText(text)
       if ProcessText(text) != "":
      
           ApendTabLicensesFounded (TabLicensesFounded, ContLicensesFounded, text)   
           if text==Licenses[i]:
              print(text +  "  Hit with Sharpen filter"  )
              TotHits=TotHits+1
           else:
               print(Licenses[i] + " detected as "+ text) 
    
    gray2= cv2.bilateralFilter(gray,3, 75, 75)
    for z in range(5,11):
       kernel = np.array([[-1,-1,-1], [-1,z,-1], [-1,-1,-1]])
       gray1 = cv2.filter2D(gray2, -1, kernel)
      
       text = pytesseract.image_to_string(gray1, lang='eng',  \
         config='--psm 6 --oem 3')
       text = ''.join(char for char in text if char.isalnum()) 
       text=ProcessText(text)
       if ProcessText(text) != "":
       
           ApendTabLicensesFounded (TabLicensesFounded, ContLicensesFounded, text)   
           if text==Licenses[i]:
              print(text +  "  Hit with Sharpen filter modified"  )
              TotHits=TotHits+1
           else:
               print(Licenses[i] + " detected as "+ text) 
              
        
    return TabLicensesFounded, ContLicensesFounded 

# https://medium.com/@sachitlele311/basics-of-image-preprocessing-using-opencv-eebfd32f68b7        
def rotate(img2, angle, Rotpoint=None): 
  (height, width)  = img2.shape[:2]   
  if Rotpoint==None :
      Rotpoint=(width//2, height//2)
      Rotmat=cv2.getRotationMatrix2D(Rotpoint, angle, 1.0)
      Rotpoint=(width//2, height//2)
      dimensions=(width, height)
      return cv2.warpAffine(img2, Rotmat, dimensions)
# MAIN  
        
ContDetected=0
ContNoDetected=0 
TotHits=0
TotFailures=0   
with open( "LicenseResults.txt" ,"w") as  w:

    for i in range(len(images)):
        #img = preprocess_image(images[i],True)
        img = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY)
        """
        rotation, spectrum, frquency =GetRotationImage(img)
        rotation=90 - rotation
        #print("Car" + str(NumberImageOrder) + " Brillo : " +str(SumBrightnessLic) +   
        #      " Desviacion : " + str(DesvLic))
        if (rotation > 0 and rotation < 30)  or (rotation < 0 and rotation > -30):
          
           #images[i]=imutils.rotate(images[i],angle=rotation)     
           #img = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY)
           image=rotate(images[i], rotation)
           img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        """
        LpImg=get_license(img)
        if len(LpImg)== 0:
         ContNoDetected=ContNoDetected+1      
         print(Licenses[i] + " NON DETECTED ")
         continue
        
        ContDetected=ContDetected+1
        print(Licenses[i] + " DETECTED ")
        
        gray=LpImg
        License=Licenses[i]
        
        x_resize=220
        y_resize=70
        
        Resize_xfactor=1.78
        Resize_yfactor=1.78
        
        ContLoop=0
        
        SwFounded=0
        
        
        
        
        TabLicensesFounded, ContLicensesFounded= FindLicenseNumber (gray,  License,  x_resize,  y_resize,  \
                               Resize_xfactor, Resize_yfactor)
          
        
        print(TabLicensesFounded)
        print(ContLicensesFounded)
        
        ymax=-1
        contmax=0
        licensemax=""
      
        for y in range(len(TabLicensesFounded)):
            if ContLicensesFounded[y] > contmax:
                contmax=ContLicensesFounded[y]
                licensemax=TabLicensesFounded[y]
        
        if licensemax == License:
           print(License + " correctly recognized") 
           TotHits+=1
        else:
            print(License + " Detected but not correctly recognized")
            TotFailures +=1
        print ("")  
        lineaw=[]
        lineaw.append(License) 
        lineaw.append(licensemax)
        lineaWrite =','.join(lineaw)
        lineaWrite=lineaWrite + "\n"
        w.write(lineaWrite)
              
print("")           
print("Total Hits = " + str(TotHits ) + " from " + str(len(images)) + " images readed")
     
print("")          
print("plates detected = "+str(ContDetected))
print("plates no detected = "+str(ContNoDetected))    

def draw_box(image_path, cor, thickness=3): 
    pts=[]  
    x_coordinates=cor[0][0]
    y_coordinates=cor[0][1]
    # store the top-left, top-right, bottom-left, bottom-right 
    # of the plate license respectively
    for i in range(4):
        pts.append([int(x_coordinates[i]),int(y_coordinates[i])])
    
    pts = np.array(pts, np.int32)
    pts = pts.reshape((-1,1,2))
    vehicle_image = preprocess_image(image_path)
    
    cv2.polylines(vehicle_image,[pts],True,(0,255,0),thickness)
    return vehicle_image

