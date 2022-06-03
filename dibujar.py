

from copy import copy
import math
from random import random
import matplotlib.pyplot as plt
import numpy as np


def get_dimension_fichero(fichero):
    with open(fichero,'r') as file:
        dimension = 0
        encuentra = False
        for line in file:
            for word in line.split():
                if encuentra:
                    dimension = word
                    encuentra = False
                if word == "DIMENSION:":
                    encuentra = True
    return dimension


def lectura_archivo(fichero):
    dimension = int(get_dimension_fichero(fichero))
    valores = np.loadtxt(fichero,skiprows=6,max_rows=dimension,dtype='int')

    res = list()
    tmp = list()
    for i in range(len(valores)):
        tmp = [valores[i][1],valores[i][2]]
        res.append(tmp)

    return res




datos = lectura_archivo("a280.tsp")


# 280
indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
           26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 
           49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71,
           72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 
           95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114,
           115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133,
           134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 
           153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 172, 
           171, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 
           191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 
           210, 211, 212, 213, 214, 215, 216,217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228,
           229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247,
           248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266,
           267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 0]

indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 
164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 0]

indices_optimo = [1,2,242,243,244,241,240,239,238,237,236,235,234,233,232,231,246,245,247,250,251,230,229,
                  228,227,226,225,224,223,222,221,220,219,218,217,216,215,214,213,212,211,210,207,206,205,
                  204,203,202,201,198,197,196,195,194,193,192,191,190,189,188,187,186,185,184,183,182,181,
                  176,180,179,150,178,177,151,152,156,153,155,154,129,130,131,20,21,128,127,126,125,124,123
                  ,122,121,120,119,157,158,159,160,175,161,162,163,164,165,166,167,168,169,170,172,171,173,
                  174,107,106,105,104,103,102,101,100,99,98,97,96,95,94,93,92,91,90,89,109,108,110,111,112,
                  88,87,113,114,115,117,116,86,85,84,83,82,81,80,79,78,77,76,75,74,73,72,71,70,69,68,67,66,
                  65,64,58,57,56,55,54,53,52,51,50,49,48,47,46,45,44,59,63,62,118,61,60,43,42,41,40,39,38,37,
                  36,35,34,33,32,31,30,29,28,27,26,22,25,23,24,14,15,13,12,11,10,9,8,7,6,5,4,277,276,275,274,
                  273,272,271,16,17,18,19,132,133,134,270,269,135,136,268,267,137,138,139,149,148,147,146,145
                  ,199,200,144,143,142,141,140,266,265,264,263,262,261,260,259,258,257,254,253,208,209,252,255
                  ,256,249,248,278,279,3,280]


# 130

indices_optimo = indices_optimo - np.ones(len(indices_optimo))
print(indices_optimo)

<<<<<<< HEAD
indices_obtenidos = np.array(indices)
=======
indices = np.array(indices)
>>>>>>> modificacion-buena

eje_x = []
eje_y = []
for k in indices:
    eje_x.append(datos[k][0])
    eje_y.append(datos[k][1])

plt.plot(np.array(eje_x),np.array(eje_y))
plt.show()

