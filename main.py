from cProfile import label
from cmath import cos
from copy import copy, deepcopy
from dis import dis
import math
import random
from textwrap import indent
import matplotlib.pyplot as plt
import numpy as np
import time

from sympy import appellf1

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


def calculo_matriz_heuristica(valores):
    """
    Calculo de distancias entre cada punto, todos con todos
    para generar la matriz de heuristicas
    :return:
    """
    
  
    heuristicas = np.zeros((len(valores),len(valores)))
    for puntoA in range(0,len(valores)):
        for puntoB in range(0,len(valores)):
            # print(puntoA , puntoB)
            if puntoA != puntoB:
                # print(valores[puntoA] , " " , valores[puntoB])
                x = valores[puntoA][0] - valores[puntoB][0]
                y = valores[puntoA][1] - valores[puntoB][1]
                value = int(math.sqrt(x*x + y*y))
                if value != 0:
                    heuristicas[puntoA][puntoB] = 1/value
                    heuristicas[puntoB][puntoA] = 1/value
                if value == 0:
                    # Como lo manejamos, debidoa  que son distintos puntos 
                    # pero con una posicion identica
                    heuristicas[puntoA][puntoB] = 1 #0.01
                    heuristicas[puntoB][puntoA] = 1 
            else:
                heuristicas[puntoA][puntoB] = math.inf
                heuristicas[puntoB][puntoA] = math.inf
                


    return heuristicas

def calculo_matriz_distancias(valores):
    """
    Calculo de distancias entre cada punto, todos con todos
    para generar la matriz de heuristicas
    :return:
    """
    
    # distancias = np.zeros((len(valores)-1,len(valores)-1))
    distancias = np.zeros((len(valores),len(valores)))
    
    # se han mod len - 1
    for puntoA in range(0,len(valores)):
        for puntoB in range(0,len(valores)):
            # print(puntoA , puntoB)
            if puntoA != puntoB:
                # print(valores[puntoA] , " " , valores[puntoB])
                x = valores[puntoA][0] - valores[puntoB][0]
                y = valores[puntoA][1] - valores[puntoB][1]
                value = int(math.sqrt(x*x + y*y))
                if value != 0:
                    distancias[puntoA][puntoB] = value
                    distancias[puntoB][puntoA] = value
                if value == 0:
                    # Como lo manejamos, debidoa  que son distintos puntos 
                    # pero con una posicion identica
                    distancias[puntoA][puntoB] = 0
                    distancias[puntoB][puntoA] = 0
            else:
                distancias[puntoA][puntoB] = math.inf

    return distancias


def get_valor_minimo(distancias_segun_punto,visitados):

    agregado = False
    valor_minimo = min(distancias_segun_punto)
    distancias_segun_punto = list(distancias_segun_punto)
    indice = distancias_segun_punto.index(valor_minimo)

    while not agregado:
        if indice in visitados:
            distancias_segun_punto

    return valor_minimo,indice

def greedy_calculo_camino_optimo(problema,punto_partida = 0):
    """
    Cálculo distancia con algoritmo greedy,
    
    Parametros
    ----------
    
    problema : 'Fichero de datos'
    punto_partida : 'Nodo de inicio' 
    
    """

    datos = lectura_archivo(problema)
    distancias = calculo_matriz_distancias(datos)
   
    camino = [punto_partida]
    
    for i in range(len(distancias)):
        dist_minima = 99999
        indice_min = -1    
        for k in range(0,len(distancias)):
            if k not in camino:
                dis_punto = distancias[camino[-1]][k]
                if dis_punto < dist_minima:
                    dist_minima = dis_punto
                    indice_min = k
        camino.append(indice_min)

    coste = 0
    for i in range(len(camino)-1):
        a = camino[i]
        b = camino[i+1]
        coste += distancias[a][b]
        
    coste+= distancias[camino[-1]][camino[0]]


    return 1/(len(distancias)*coste)
    
    
    
    
    
def calcular_punto_siguiente(camino_generado,matriz_feromonas,matriz_heuristica,alpha=1,beta=2):
    # Calculo el siguinte punto teniendo en cuenta los puntos visitados
    # y la heuristica y las feromonas
    
    punto_ultimo = camino_generado[-1]
    probabilidades = []
    indices_validos = []
    todos_los_puntos = np.arange(0,len(matriz_feromonas))
    sumatorio = 0
    
    no_visitados = set(todos_los_puntos) - set(camino_generado)
    no_visitados = list(no_visitados)
    
    for k in (no_visitados):
        feromona_sum = math.pow(matriz_feromonas[punto_ultimo][k],alpha)
        heuristica_sum = math.pow(matriz_heuristica[punto_ultimo][k],beta)
        sumatorio += feromona_sum * heuristica_sum
        

    for indice in no_visitados:
        # Se calculan los valores de probabilidad de cada punto no visitado
        feromona_arco = math.pow(matriz_feromonas[punto_ultimo][indice],alpha)
        heuristica_arco = math.pow(matriz_heuristica[punto_ultimo][indice],beta)
        
        probabilidad = (feromona_arco * heuristica_arco) / sumatorio
        probabilidades.append(probabilidad)
        indices_validos.append(indice)
        
    valor_aleatorio = np.random.uniform(0,1)
    acumulado_probabilidad = 0.0
    indice_punto_elegido = 0
    while valor_aleatorio > acumulado_probabilidad:
        acumulado_probabilidad += probabilidades[indice_punto_elegido]
        indice_punto_elegido += 1
    
    
    return indices_validos[indice_punto_elegido-1]

def calcular_punto_siguiente_sistema_hormigas(camino_generado,matriz_feromonas,matriz_heuristica,valor_feromona_inicial,alpha=1,beta=2):
        
    q = 0.98
    
    r = np.random.uniform(0,1)
        
    if  r <= q :
        # se aplica el arg max de todos los arcos
        # Se aplica la regla de transicion de SCH
        punto_ultimo = camino_generado[-1]
        todos_los_puntos = np.arange(0,len(matriz_feromonas))
        no_visitados = set(todos_los_puntos) - set(camino_generado)
        no_visitados = list(no_visitados)
    
        mejor_valor = 0 # math.inf
        indice_mejor_valor = 0

        
        for k in no_visitados:
            feromona = math.pow(matriz_feromonas[punto_ultimo][k],alpha)
            heuristica = math.pow(matriz_heuristica[punto_ultimo][k],beta)
            producto = feromona * heuristica
            if producto > mejor_valor:
                mejor_valor = producto
                indice_mejor_valor = k
                
        return indice_mejor_valor
    else:
        return calcular_punto_siguiente(camino_generado,matriz_feromonas,matriz_heuristica,alpha=1,beta=2)
    

    

def creacion_de_camino(matriz_feromonas,matriz_heuristica,punto_partida,valor_inicial_feronomas=1,sistema=False):
    
    camino_generado = []
    camino_generado.append(punto_partida)
    
    coste_camino_generado = 0
     
    
    for lon in range(len(matriz_feromonas)-1): 
        punto_ultimo = camino_generado[-1]
        
        if sistema:
            punto_siguiente = calcular_punto_siguiente_sistema_hormigas(camino_generado,matriz_feromonas,matriz_heuristica,valor_inicial_feronomas)
        else:
            punto_siguiente = calcular_punto_siguiente(camino_generado,matriz_feromonas,matriz_heuristica,valor_inicial_feronomas)
        
        
        valor = matriz_heuristica[punto_ultimo][punto_siguiente]

        coste_camino_generado += 1/valor         
        camino_generado.append(punto_siguiente)
    
    # Hay que cerrar ciclo, volver al punto de inicio??
    valor = matriz_heuristica[camino_generado[-1]][punto_partida]
    coste_camino_generado += (1/valor)
    camino_generado.append(punto_partida)
    
   
    
    return coste_camino_generado,camino_generado

def evaporacion_global(matriz_feromonas,mejor_camino_global,coste_mejor_camino_global,valor_evaporacion=0.1):
        

    for k in range(len(mejor_camino_global)-1):
        pos = mejor_camino_global[k]
        pos_sig = mejor_camino_global[k+1]
        aporte = matriz_feromonas[pos][pos_sig]*(1-valor_evaporacion) + valor_evaporacion*(1/coste_mejor_camino_global)
        matriz_feromonas[pos][pos_sig] = aporte
        matriz_feromonas[pos_sig][pos] = aporte
    
    aporte = matriz_feromonas[mejor_camino_global[-1]][mejor_camino_global[0]]*(1-valor_evaporacion) + valor_evaporacion*(1/coste_mejor_camino_global)
    matriz_feromonas[mejor_camino_global[-1]][mejor_camino_global[0]] = aporte
    matriz_feromonas[mejor_camino_global[0]][mejor_camino_global[-1]] = aporte

    return matriz_feromonas

def evaporacion_local(matriz_feromonas,camino,valor_feromona_inicial,valor_evaporacion=0.1):
    
    
    
    for k in range(len(camino)-1):
        pos = camino[k]
        pos_sig = camino[k+1]
        aporte = matriz_feromonas[pos][pos_sig]*(1-valor_evaporacion) + valor_evaporacion*valor_feromona_inicial
        matriz_feromonas[pos][pos_sig] = aporte
        matriz_feromonas[pos_sig][pos] = aporte
    
    aporte = matriz_feromonas[camino[-1]][camino[0]]*(1-valor_evaporacion) + valor_evaporacion*valor_feromona_inicial
    matriz_feromonas[camino[-1]][camino[0]] = aporte
    matriz_feromonas[camino[0]][camino[-1]] = aporte

    return matriz_feromonas

def evaporacion_local_por_arcos(matriz_feromonas,puntoA,puntoB,valor_feromona_inicial,valor_evaporacion=0.1):
    
    
    aporte = matriz_feromonas[puntoA][puntoB]*(1-valor_evaporacion) + valor_evaporacion*valor_feromona_inicial
    matriz_feromonas[puntoA][puntoB] = aporte
    matriz_feromonas[puntoB][puntoA] = aporte

    return matriz_feromonas

def actualizar_matriz_feromonas(matriz_feromonas,caminos_hormigas,costes_caminos_hormigas,mejor_camino_global,coste_mejor_camino_global=1,elite=0,valor_evaporacion=0.1):
    # modificamos los valores de la matriz de feromonas, modificado ambas posiciones  

    variaciones =  np.ones((len(matriz_feromonas),len(matriz_feromonas)))*(1-valor_evaporacion)
    matriz_feromonas = matriz_feromonas*variaciones
    
    matriz_aportaciones = np.zeros((len(matriz_feromonas),len(matriz_feromonas)))

    
    for i in range(len(caminos_hormigas)):
        camino = caminos_hormigas[i]
        aporte = costes_caminos_hormigas[i]
        for k in range(len(camino)-1):
            pos = camino[k]
            pos_sig = camino[k+1]
            matriz_aportaciones[pos][pos_sig] += (1/aporte)  
            matriz_aportaciones[pos_sig][pos] += (1/aporte) 

        matriz_aportaciones[camino[-1]][camino[0]] += (1/aporte) 
        matriz_aportaciones[camino[0]][camino[-1]] += (1/aporte)
        
    matriz_feromonas = np.add(matriz_feromonas,matriz_aportaciones)
    
    if elite != 0:
        for k in range(len(mejor_camino_global)-1):
            pos = mejor_camino_global[k]
            pos_sig = mejor_camino_global[k+1]
            matriz_feromonas[pos][pos_sig] += (elite/coste_mejor_camino_global)  
            matriz_feromonas[pos_sig][pos] += (elite/coste_mejor_camino_global) 
        
        matriz_feromonas[mejor_camino_global[-1]][mejor_camino_global[0]] += (elite/coste_mejor_camino_global)  
        matriz_feromonas[mejor_camino_global[0]][mejor_camino_global[-1]] += (elite/coste_mejor_camino_global)
    
    
    return matriz_feromonas
        
                    

def dibujar(fichero,camino,optimo=False):
    datos = lectura_archivo(fichero)

    eje_x = []
    eje_y = []
    if not optimo:
        for k in camino:
            eje_x.append(datos[k][0])
            eje_y.append(datos[k][1])

        plt.plot(np.array(eje_x),np.array(eje_y))
        plt.xlabel("Coord. X")
        plt.ylabel("Coord. Y")
        plt.title(fichero)
        plt.show()
    else:
        for k in range(len(camino)):
            camino[k] -= 1
        
        for k in camino:
            eje_x.append(datos[k][0])
            eje_y.append(datos[k][1])
        plt.plot(np.array(eje_x),np.array(eje_y))
        plt.xlabel("Coord. X")
        plt.ylabel("Coord. Y")
        plt.title(fichero)
        plt.show()


def dibujar_compuesto(fichero,camino,camino_optimo):
    datos = lectura_archivo(fichero)

    eje_x = []
    eje_y = []
    for k in camino:
        eje_x.append(datos[k][0])
        eje_y.append(datos[k][1])
        
    for k in range(len(camino_optimo)):
        camino_optimo[k] -= 1
        
    eje_x_optimo = []
    eje_y_optimo = []
    for k in camino_optimo:
        eje_x_optimo.append(datos[k][0])
        eje_y_optimo.append(datos[k][1])

    plt.plot(np.array(eje_x),np.array(eje_y),'r',label="camino")
    plt.plot(np.array(eje_x_optimo),np.array(eje_y_optimo),'b',label="camino_optimo")
    
    plt.legend(loc="upper left")
    
    plt.show()

def hormigas(problema = "ch130.tsp",n_hormigas=10,limite_iteracciones = 100_000,minutos_limite=1,punto_partida = 0,elite = 0,verbose = False):
    
    valor_inicial_feronomas = greedy_calculo_camino_optimo(problema)
    
    datos = lectura_archivo(problema)
    matriz_heuristica = calculo_matriz_heuristica(datos)
    matriz_feromonas = np.ones((len(matriz_heuristica),len(matriz_heuristica))) * valor_inicial_feronomas
    
    mejor_camino_global = list()
    coste_mejor_camino_global = math.inf
    
    caminos_hormigas = []
    costes_caminos_hormigas = []    
    
    eje_x = []
    eje_y = []
    inicio = time.time()
  
    iteraccion = 0
    iteracion_mejora_ultima = 0
    while iteraccion < limite_iteracciones and (time.time() - inicio) < 60*minutos_limite:
        
        if verbose:
            eje_x.append(iteraccion)
            eje_y.append(coste_mejor_camino_global)
        
        costes_caminos_hormigas = []
        caminos_hormigas = []
        for hormiga in range(n_hormigas):
            # Cada hormiga crea su recorrido
            
            coste_camino,camino_actual = creacion_de_camino(matriz_feromonas,matriz_heuristica,punto_partida)
            caminos_hormigas.append(camino_actual)
            costes_caminos_hormigas.append(coste_camino)
         
            # Nos quedamos con el camino si mejora al global y su coste
            if coste_camino < coste_mejor_camino_global:
                mejor_camino_global = camino_actual
                coste_mejor_camino_global = coste_camino
                iteracion_mejora_ultima = iteraccion
            
            # print(" tiempo ",time.time() - inicio , " iteracion " , iteraccion)
        
        # Aplicamos la evaporación y el aporte 
        
        matriz_feromonas = actualizar_matriz_feromonas(matriz_feromonas,caminos_hormigas,costes_caminos_hormigas,coste_mejor_camino_global=coste_mejor_camino_global,mejor_camino_global=mejor_camino_global,elite=elite)
        
        iteraccion += 1
        
    if verbose:
        dibujar(problema,mejor_camino_global)
        plt.plot(np.array(eje_x),np.array(eje_y))
        plt.show()
    
    print("ite" , iteracion_mejora_ultima)
    return mejor_camino_global , coste_mejor_camino_global

def sistema_colonia_hormigas(problema = "ch130.tsp",n_hormigas=10,limite_iteracciones = 100_000,minutos_limite=1,punto_partida = 0,verbose = False):
    
    valor_inicial_feronomas = greedy_calculo_camino_optimo(problema)
        
    datos = lectura_archivo(problema)
    matriz_heuristica = calculo_matriz_heuristica(datos)
    matriz_feromonas = np.ones((len(matriz_heuristica),len(matriz_heuristica))) * valor_inicial_feronomas
    
    mejor_camino_global = list()
    coste_mejor_camino_global = math.inf
    
    caminos_hormigas = []
    costes_caminos_hormigas = []    
    
    eje_x = []
    eje_y = []
    inicio = time.time()
  
    iteraccion = 0
    iteracion_mejora_ultima = 0

    
    while iteraccion < limite_iteracciones and (time.time() - inicio) < 60*minutos_limite:
        
        if verbose:
            eje_x.append(iteraccion)
            eje_y.append(coste_mejor_camino_global)
        
        costes_caminos_hormigas = []
        caminos_hormigas = []
        for hormiga in range(n_hormigas):
            # Cada hormiga crea su recorrido
            coste_camino,camino_actual = creacion_de_camino(matriz_feromonas,matriz_heuristica,punto_partida,valor_inicial_feronomas,sistema=True)
            caminos_hormigas.append(camino_actual)
            costes_caminos_hormigas.append(coste_camino)
            matriz_feromonas = evaporacion_local(matriz_feromonas,camino=camino_actual,valor_feromona_inicial=valor_inicial_feronomas)
            
            # Nos quedamos con el camino si mejora al global y su coste
            if coste_camino < coste_mejor_camino_global:
                mejor_camino_global = camino_actual.copy()
                coste_mejor_camino_global = coste_camino
                iteracion_mejora_ultima = iteraccion
            
            
        # print(" tiempo ",time.time() - inicio , " iteracion " , iteraccion)
        # Aplicamos la evaporación y el aporte 
        matriz_feromonas = evaporacion_global(matriz_feromonas,mejor_camino_global,coste_mejor_camino_global)
        
        iteraccion += 1
        
    if verbose:
        dibujar(problema,mejor_camino_global)
        plt.plot(np.array(eje_x),np.array(eje_y))
        plt.show()
    print("ite" , iteracion_mejora_ultima)
    return mejor_camino_global , coste_mejor_camino_global





f =  "a280.tsp"  
# f = "ch130.tsp"

# semilla = random.randint(0,4294967296)

# random.seed(semilla)
# np.random.seed(semilla)

# mejor_camino_global,coste_mejor_camino_global_elite = hormigas(problema=f,minutos_limite=5,elite=0,verbose=True)
# mejor_camino_global,coste_mejor_camino_global_elite = hormigas(problema=f,minutos_limite=5,elite=15,verbose=True)
# # mejor_camino_global,coste_mejor_camino_global_elite = sistema_colonia_hormigas(problema=f,minutos_limite=5,verbose=True)

# print("hormiga " ,coste_mejor_camino_global_elite)  
# print("semilla ", semilla)

# print(mejor_camino_global)
# dibujar(f,mejor_camino_global)


# optimo_130 = [1,41,39,117,112,115,28,62,105,128,16,45,5,11,76,109,61,129,124,64,69,86,88,26,7,97,70,107,127,104,43,34,17,31,27,19,100,15,29,24,116,95,79,87,12,81,103,77,94,89,110,98,68,63,48,25,113,32,36,84,119,111,123,101,82,57,9,56,65,52,75,74,99,73,92,38,106,53,120,58,49,72,91,6,102,10,14,67,13,96,122,55,60,51,42,44,93,37,22,47,40,23,33,21,126,121,78,66,85,125,90,59,30,83,3,114,108,8,18,46,80,118,20,4,35,54,2,50,130,71]
# optimo_280 = [1,2,242,243,244,241,240,239,238,237,236,235,234,233,232,231,246,245,247,250,251,230,229,228,227,226,225,224,223,222,221,220,219,218,217,216,215,214,213,212,211,210,207,206,205,204,203,202,201,198,197,196,195,194,193,192,191,190,189,188,187,186,185,184,183,182,181,176,180,179,150,178,177,151,152,156,153,155,154,129,130,131,20,21,128,127,126,125,124,123,122,121,120,119,157,158,159,160,175,161,162,163,164,165,166,167,168,169,170,172,171,173,174,107,106,105,104,103,102,101,100,99,98,97,96,95,94,93,92,91,90,89,109,108,110,111,112,88,87,113,114,115,117,116,86,85,84,83,82,81,80,79,78,77,76,75,74,73,72,71,70,69,68,67,66,65,64,58,57,56,55,54,53,52,51,50,49,48,47,46,45,44,59,63,62,118,61,60,43,42,41,40,39,38,37,36,35,34,33,32,31,30,29,28,27,26,22,25,23,24,14,15,13,12,11,10,9,8,7,6,5,4,277,276,275,274,273,272,271,16,17,18,19,132,133,134,270,269,135,136,268,267,137,138,139,149,148,147,146,145,199,200,144,143,142,141,140,266,265,264,263,262,261,260,259,258,257,254,253,208,209,252,255,256,249,248,278,279,3,280]






