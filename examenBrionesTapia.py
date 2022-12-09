import numpy as np
import matplotlib.pyplot as plt
import math
import cv2

def conversionImagen(imagen):
    imagen= cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
    return imagen

def redimensionImagen(imagen, filas, columnas, tamaño):
    r= tamaño/columnas
    dim= (tamaño, int(filas*r))
    imagenRe=cv2.resize(imagen, dim) #Redimensionamiento
    return imagenRe

def calcular_distancia ( p1 , p2 ) :
    x = p1 [0] - p2 [0] #Diferencia de las coordenadas en x
    y = p1 [1] - p2 [1] #Diferencia de las coordenadas en y
    return math.sqrt( x**2 + y**2)

def max_distancia ( l ) :
    par_max = ()
    val_max = float(0) 
    for i in range (len ( l ) ): #Recorremos la lista
        for j in range ( i +1 , len ( l ) ): #Tomamos una posición adelante para calcular distancia con la anterior
            d = calcular_distancia ( l[i][0] , l[j][0]) #Calculamos distancia
            if( d > val_max ): #Evaluamos si la distancia es la más grande que hemos encontrado 
                val_max = d #Actualizamos nuestra distancia máxima
                par_max = i , j #Actualizamos nuestros indices guardados de los puntos más alejados
    return par_max

#Lectura de la imagen original
image = cv2.imread("Jit1.JPG")
imgMediciones= cv2.imread("Jit1.JPG") #Ocupamos otra imagen igul a la original para dibujarle las líneas
filas, columnas, dimensiones= image.shape #Ontenemos el tamaño de la imagen
tamaño=500 #tamaño para mostrar las ventanitas de salidas obtenidas
imageOriginal = conversionImagen(image) #Cambio de canales BGR a RGB para mostrar la imagen

pixeles = image.reshape((-1, 1)) #Convertimos los pixeles de la imagen en un arreglo de 3 columnas para sus valores RGB
pixeles = np.float32(pixeles)

#Aplicación del algoritmo de Kmeans
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85) 
k = 2 #Creamos 2 clusters, un para los jitomates y otro para las piedritas de fondo
retval, labels, centers = cv2.kmeans(pixeles, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS) 

centers = np.uint8(centers) #centers son nuestros centroides encontrados
segmented_data = centers[labels.flatten()] #Mapeamos los pixeles con su centroide más similar de acuerdo a sus valores de RGB
segmented_image = segmented_data.reshape((image.shape)) # reshape data into the original image dimensions
segmented_image= redimensionImagen(segmented_image, filas, columnas, tamaño) #Redimensión del tamaño para mostrar la imagen

#Obtenemos nuestra imagen segmentada en escala de grises
gray_1= cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)
#Binarizamos a partir de nuestra imagen segmentada
_, binarizada= cv2.threshold(gray_1, 180, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
binarizada= redimensionImagen(binarizada, filas, columnas, tamaño) #Redimensión del tamaño para mostrar la imagen
kernel_morfo = np.ones((25,25),np.uint8) #Creación de un kernel morfológico
#Aplicamos un cierre morfológico para obtener una imagen binarizada unicamente con nuestros objetos
opening= cv2.morphologyEx(binarizada, cv2.MORPH_OPEN, kernel_morfo)
opening= redimensionImagen(opening, filas, columnas, tamaño) #Redimensión del tamaño para mostrar la imagen
# Aplicación del filtro LoG a nuestra imagen que identificó los objetos
dst = cv2.Laplacian(opening, cv2.CV_64F)
abs_dst = cv2.convertScaleAbs(dst)
#En una lista almacenamos los contornos encontrados
contornos,hierarchy1 = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cv2.drawContours(segmented_image, contornos, -1, (255,255,255), 2) #Dibuja los contornos

#En mi lista la celda 0=controno jitomate 4,  celda 2=contorno jitomate 2
#Obtención de puntos extremos del jitomate 2
print("\nPara el jitomate 2 obtenemos los puntos: ")
minimo2=5000
#ESTA ES LA PRIMERA IDEA PARA OBTENER LOS PUNTOS EXTREMOS
#Dado que el jitomate está en horizontal, según mi idea basta con obtener la coordenada con el x más a la izquierda
for i in range(len(contornos[2])): #Recorremos la lista de coordenadas de nuestros puntos de contorno
    if int(contornos[2][i][0][0])<minimo2: #Comparamos cada x de cada coordenada contra el minimo
        minimo2=int(contornos[2][i][0][0]) #Actualizamos el minimo si encontro uno más chiquito
        punto_minimo2=contornos[2][i][0] #Actualizamos nuestra coordenada guardada como la minima
print("Punto extremo izquierdo: ({}, {})".format(int(punto_minimo2[0]), int(punto_minimo2[1])))
maximo2=0
#Dado que el jitomate está en horizontal, según mi idea basta con obtener la coordenada con el x más a la derecha
for i in range(len(contornos[2])): #Recorremos la lista de coordenadas de nuestros puntos de contorno
    if int(contornos[2][i][0][0])>maximo2: #Comparamos cada x de cada coordenada contra el máximo
        maximo2=int(contornos[2][i][0][0]) #Actualizamos el maximo si encontro uno más grande
        punto_maximo2=contornos[2][i][0] #Actualizamos nuestra coordenada guardada como la maxima
print("Punto extremo derecho: ({}, {})".format(int(punto_maximo2[0]), int(punto_maximo2[1])))
#Sacamos la distancia entre nuestros puntos encontrados
distancia = math.sqrt((int(punto_maximo2[0])-int(punto_minimo2[0]))**2+(int(punto_maximo2[1])-int(punto_minimo2[1]))**2)
print("\nLongitud del jitomate 2: {}\n".format(distancia))

# Dibujamos la línea entre los puntos que encontramos en la imagen que creamos para mostrar está parte
imgMediciones= redimensionImagen(imgMediciones, filas, columnas, tamaño) #Redimensión del tamaño para mostrar la imagen
cv2.line(imgMediciones, (int(punto_minimo2[0]), int(punto_minimo2[1])), (int(punto_maximo2[0]),int(punto_maximo2[1])), (0,255,255), 2)

#ESTA ES LA SEGUNDA IDEA PARA OBTENER LOS PUNTOS EXTREMOS
i , j = max_distancia ( contornos[0] ) #Llamamos a nuestra función que basicamente compara cada tupla con todas las demas 
                                    # y nos regresa los indices de las tuplas entre las que encontró mayor distancia
inferior= contornos[0][i] #Punto inferior del jitomate 4
superior= contornos[0][j] #Punto superior del jitomate 4

#Obtención de puntos extremos del jitomate 4
print("\nPara el jitomate 4 obtenemos los puntos: ")
print("Punto extremo inferior: ({}, {})".format(int(inferior[0][0]), int(inferior[0][1])))
print("Punto extremo superior: ({}, {})".format(int(superior[0][0]), int(superior[0][1])))
#Sacamos la distancia entre nuestros puntos encontrados
distancia = math.sqrt((int(superior[0][0])-int(inferior[0][0]))**2+(int(superior[0][1])-int(inferior[0][1]))**2)
print("\nLongitud del jitomate 4: {}\n".format(distancia))

# Dibujamos la línea entre los puntos que encontramos en la imagen que creamos para mostrar está parte
cv2.line(imgMediciones, (int(inferior[0][0]), int(inferior[0][1])), (int(superior[0][0]), int(superior[0][1])), (0,255,255), 2)

image= redimensionImagen(image, filas, columnas, tamaño) #Redimensión del tamaño para mostrar la imagen
#Muestra de los resultados obtenidos en las partes del proceso
cv2.imshow('Imagen original', image)
cv2.imshow('Imagen LoG', abs_dst)
cv2.imshow('Imagen segmentada', segmented_image)
cv2.imshow('Imagen binarizada', binarizada)
cv2.imshow('Imagen con extraccion de objetos', opening)
cv2.imshow('Imagen con mediciones', imgMediciones)
cv2.waitKey(0)
cv2.destroyAllWindows()