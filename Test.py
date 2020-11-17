# Script di Test per provare più algoritmi (findContours, matchTemplate ecc.)

# Elenco librerie utili

import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
from os import listdir
from os.path import isfile, join
import imutils
from imutils import perspective
from imutils import contours
from scipy.spatial import distance as dist
import periphery


# Lettura di tutti i file di test

# Leggere immagini catene dalla folder test_catene e conta il numero di file presenti
mypath_catene='C:/Users/Andrea Fabrizi/Visual Studio/test_catene'
onlyfiles_catene = [ f for f in listdir(mypath_catene) if isfile(join(mypath_catene,f)) ]

# Leggere immagini maglie di riferimento dalla folder test_catene/maglie e conta il numero di file presenti
mypath_maglie='C:/Users/Andrea Fabrizi/Visual Studio/test_catene/maglie'
onlyfiles_maglie = [ f for f in listdir(mypath_maglie) if isfile(join(mypath_maglie,f)) ]

# Definire i vettori utilizzati per la lettura dei singoli file
images = np.empty(len(onlyfiles_catene), dtype=object)
template = np.empty(len(onlyfiles_catene), dtype=object)

images_gray = np.empty(len(onlyfiles_catene), dtype=object)
images_thresh = np.empty(len(onlyfiles_catene), dtype=object)
images_final = np.empty(len(onlyfiles_catene), dtype=object)

dst = np.empty(len(onlyfiles_catene), dtype=object)
orig = np.empty(len(onlyfiles_catene), dtype=object)

# Lettura

for n in range(0, len(onlyfiles_catene)):
    images[n] = cv2.imread(join(mypath_catene,onlyfiles_catene[n]))

for n in range(0,len(onlyfiles_maglie)):
    template[n] = cv2.imread(join(mypath_maglie,onlyfiles_maglie[n]))

# Estrazione maglie (linee, centri, p.ti estremi) data un'immagine
#
#def estra_maglia(images, num_files)
#   
#   for n in range(0, num_files):
#
#
#

# Calcola Punto Medio
def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)



# Metodo 1: matchTemplate
# Cerca le ricorrenze di un'immagine di riferimento (una o più maglie, fino a 5) all'interno delle immagini delle catene

def occurrences(images, template, onlyfiles_catene):
    
    # Riferire la larghezza della catena (in cm)
    width_ref = 1
    # Scegliere quante maglie di riferimento avere (1, 2, 3, 4 o 5 maglie)
    num_maglie = int(input("Quante maglie vuoi considerare (1, 2, 3, 4 o 5)? "))
    
    # Conviene considerare immagini in scala di grigio
    # In particolare dopo una binarizzazione di Otsu con soglia minima abbastanza alta - 180
    # E dopo un'erosione per "tappare" i buchi
    template = cv2.cvtColor(template[int(num_maglie)-1], cv2.COLOR_BGR2GRAY)
    ret,template = cv2.threshold(template,180,255,cv2.THRESH_OTSU)
    template = cv2.erode(template, None, iterations=2)

    # Scegliere la soglia (più bassa è, più occorrenze dovrebbe trovare)
    threshold = float(input("Che soglia per il matchTemplate desideri? Si consoglia da 0.35 a 0.7. "))
    
    # Precisione occorrenze (con il 100% si evita che si sovrappongano le occorrenze trovate)
    prec = 75/100
    # Si inizializzano i colori del punto del riquadro di delimitazione
    pixelsPerMetric = None

    for n in range(0, len(onlyfiles_catene)):

        # Conviene considerare immagini in scala di grigio
        # In particolare dopo una binarizzazione di Otsu con soglia minima abbastanza alta - 180
        # E dopo un'erosione per "tappare" i buchi
        images_gray[n] = cv2.cvtColor(images[n], cv2.COLOR_BGR2GRAY)
        ret,images_thresh[n] = cv2.threshold(images_gray[n],180,255,cv2.THRESH_OTSU)
        images_final[n] = cv2.erode(images_thresh[n], None, iterations=2)
              
        # La funzione che cerca le occorrenze è il matchTemplate, di cui si può scegliere anche la modalità
        # Si è scelto TM_CCOEFF_NORMED
        w, h = template.shape[::-1]
        res = cv2.matchTemplate(images_final[n], template, cv2.TM_CCOEFF_NORMED)
        
        # Oltre la soglia c'è l'occorrenza
        loc = np.where( res >= float(threshold))
        pt_old = (0,0)

        for pt in zip(*loc[::-1]):
            if pt[1] - pt_old[1] < (template.shape[0]/num_maglie)*prec:
                continue
            else:
                cv2.rectangle(images[n], pt, (pt[0] + w, pt[1] + h), (0,0,255), 8)
                # Corners
                tl = pt
                tr = (pt[0] + w, pt[1])
                br = (pt[0] + w, pt[1] + h)
                bl = (pt[0], pt[1] + h)
                # Si calcola il punto medio tra le coordinate TL e TR, seguito da quello tra BL e BR
                (tltrX, tltrY) = midpoint(tl, tr)
                (blbrX, blbrY) = midpoint(bl, br)
                # Si calcola il punto medio tra le coordinate TL e BL, seguito da quello tra TR e BR
                (tlblX, tlblY) = midpoint(tl, bl)
                (trbrX, trbrY) = midpoint(tr, br)
                # Si calcola la distanza Euclidea mediante i punti medi
                dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
                dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
            
                # Se i Pixel x Metric non è ben inizializzato allora è il rapporto dei pixel e la matrice fornita
                if pixelsPerMetric is None:
                    pixelsPerMetric = dB / float(width_ref)
                # CALCOLO DIMENSIONE DEGLI OGGETTI
                dimA = dA / pixelsPerMetric
                dimB = dB / pixelsPerMetric
                # Scrive le dimensioni dell'oggetto
            
                cv2.putText(images[n], "{:.1f}cm".format(dimB), (int(tltrX - 150), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX, 3.5, (255, 0, 255), 8)
                cv2.putText(images[n], "{:.1f}cm".format(dimA), (int(trbrX + 10), int(trbrY - 300)), cv2.FONT_HERSHEY_SIMPLEX, 3.5, (255, 0, 255), 8)
                # show the output image
                #cv2.imshow("Image", images[n])
                #cv2.waitKey(0)
            pt_old = pt
        
    # Scrivere le immagini sulla folder tes_catene
    num=0
    for num in range(len(onlyfiles_catene)):
        cv2.imwrite('C:/Users/Andrea Fabrizi/Visual Studio/test_catene/out_metodo1/metodo1_out' + str(num) + '_thresh' + str(threshold) + '_nmag' + str(num_maglie) + '.png', images[num])
        num+=1
    
    print("Metodo Occorrenze - Done")



# Metodo 2: findContours
def find_contours(images, onlyfiles_catene):

    width_ref = 2   # input("Larghezza (in cm) della catena come riferimento: ")

    for n in range(0, len(onlyfiles_catene)):
        # Si carica l'immagine, si converte in scala di grigi e si sfoca leggermente mediante Gauss
        
        # scale_percent = 20 # percent of original size
        # width = int(images[n].shape[1] * scale_percent / 100)
        # height = int(images[n].shape[0] * scale_percent / 100)
        # dim = (width, height)
        # images[n] = cv2.resize(images[n], dim, interpolation = cv2.INTER_AREA)


        images_gray[n] = cv2.cvtColor(images[n], cv2.COLOR_BGR2GRAY)
        images_gray[n] = cv2.GaussianBlur(images_gray[n], (7, 7), 0)
        ret,images_thresh[n] = cv2.threshold(images_gray[n],220,255,cv2.THRESH_OTSU)
        dst[n] = cv2.erode(images_thresh[n], None, iterations=3)
        # plt.imshow(dst[n])
        # plt.show()
        
        
        # Si trovano i contorni nell'immagine "dei contorni" (edged)
        cnts, hierarchy = cv2.findContours(dst[n].copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #print(hierarchy)
        # cnts = imutils.grab_contours(cnts)
        # Si ordinano i contorni da sx a dx
        (cnts, _) = contours.sort_contours(cnts)
        # Si inizializzano i colori del punto del riquadro di delimitazione
        pixelsPerMetric = None
        num_cnts = 0
        orig[n] = images[n].copy()
        # Si esamina con un loop ogni contorno
        for c in cnts:
            # Se un contorno non è sufficientemente grande (a causa del "rumore" nel processo di rilevamento dei bordi)
            # # Si scarta la regione del contorno
            if cv2.contourArea(c) < 70000 or hierarchy[0][num_cnts][3]!=-1:
                continue
            # Si calcola il riquadro di delimitazione ruotato dell'immagine utilizzando cv2.cv.BoxPoints (OpenCV 2.4) e cv2.boxPoints (OpenCV 3.0)  
            
            box = cv2.minAreaRect(c)
            box = cv2.boxPoints(box)
            box = np.array(box, dtype="int")
            # Si ordinano le coordinate dall'angolo in alto a sx in senso orario
            box = perspective.order_points(box)
            # Si disegna il contorno dell'oggetto in verde
            cv2.drawContours(orig[n], [box.astype("int")], -1, (0, 0, 255), 10)
            
            # Si disegnano i vertici del rettangolo del riquadro di delimitazione in piccoli cerchi rossi
            for (x, y) in box:
                cv2.circle(orig[n], (int(x), int(y)), 5, (0, 0, 255), -1)
            
            # SI POSSONO ORA CALCOLARE I PUNTI MEDI
            # Si decomprime il riquadro di delimitazione ordinati
            # Si calcola il punto medio tra le coordinate TL e TR, seguito da quello tra BL e BR
            (tl, tr, br, bl) = box
            (tltrX, tltrY) = midpoint(tl, tr)
            (blbrX, blbrY) = midpoint(bl, br)
            # Si calcola il punto medio tra le coordinate TL e BL, seguito da quello tra TR e BR
            (tlblX, tlblY) = midpoint(tl, bl)
            (trbrX, trbrY) = midpoint(tr, br)
            
            # Si disegnano i punti medi sull'immagine con dei cerchi rossi
            # cv2.circle(orig[n], (int(tltrX), int(tltrY)), 5, (0, 0, 255), -1)
            # cv2.circle(orig[n], (int(blbrX), int(blbrY)), 5, (0, 0, 255), -1)
            # cv2.circle(orig[n], (int(tlblX), int(tlblY)), 5, (0, 0, 255), -1)
            # cv2.circle(orig[n], (int(trbrX), int(trbrY)), 5, (0, 0, 255), -1)
            
            # Si disegnano le linee tra i punti medi
            # cv2.line(orig[n], (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)), (0, 0, 255), 2)
            # cv2.line(orig[n], (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)), (0, 0, 255), 2)
            
            # Si calcola la distanza Euclidea mediante i punti medi
            dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
            dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
            
            # Se i Pixel x Metric non è ben inizializzato allora è il rapporto dei pixel e la matrice fornita
            if pixelsPerMetric is None:
                pixelsPerMetric = dB / float(width_ref)
            
            # CALCOLO DIMENSIONE DEGLI OGGETTI
            dimA = dA / pixelsPerMetric
            dimB = dB / pixelsPerMetric
            
            # Scrive le dimensioni dell'oggetto
            cv2.putText(orig[n], "{:.1f}cm".format(dimB), (int(tltrX - 100), int(tltrY + 100)), cv2.FONT_HERSHEY_SIMPLEX, 3.5, (255, 0, 255), 9)
            cv2.putText(orig[n], "{:.1f}cm".format(dimA), (int(trbrX - 150), int(trbrY + 20)), cv2.FONT_HERSHEY_SIMPLEX, 3.5, (255, 0, 255), 9)
            
            # show the output image
            # print(num_cnts)
            # print(num_catena)
            # Scrivere le immagini sulla folder tes_catene
            # print(num_cnts)
            
            #cv2.imwrite('C:/Users/Andrea Fabrizi/Visual Studio/test_catene/out_metodo2/metodo2_out' + str(n) + '_cnt' + str(num_cnts) + '.png', orig[n])
            
            num_cnts = num_cnts + 1
        cv2.imwrite('C:/Users/Andrea Fabrizi/Visual Studio/test_catene/out_metodo2/metodo2_out' + str(n) + '_cnt' + '.png', orig[n]) 
        # show the output image
        # cv2.imshow("Image", orig[n])
        # cv2.waitKey(0)
    # Scrivere le immagini sulla folder tes_catene
    #num=0
    #for num in range(len(onlyfiles_catene)):
    #    cv2.imwrite('C:/Users/Andrea Fabrizi/Visual Studio/test_catene/out_metodo2/metodo2_out' + str(num) + '.png', orig[num])
    #    num+=1
    print("Metodo Find Contours - Done")



metodo = int(input("Quale metodo ? \n 1: Occorrenze \n 2: Contorni\n    "))

if(metodo==1):
    occurrences(images, template, onlyfiles_catene)
else:
    if(metodo==2):
        find_contours(images, onlyfiles_catene)
    else:
        print("Metodo non valido")