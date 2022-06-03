from copy import copy, deepcopy
import math
import random
import matplotlib.pyplot as plt
import numpy as np
import time

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
    heuristicas = np.zeros((len(valores)-1,len(valores)-1))
    for puntoA in range(0,len(valores)-1):
        for puntoB in range(0,len(valores)-1):
            # print(puntoA , puntoB)
            if puntoA != puntoB:
                # print(valores[puntoA] , " " , valores[puntoB])
                x = valores[puntoA][0] - valores[puntoB][0]
                y = valores[puntoA][1] - valores[puntoB][1]
                value = int(math.sqrt(x*x + y*y))
                if value != 0:
                    heuristicas[puntoA][puntoB] = 1/value
                if value == 0:
                    # Como lo manejamos, debidoa  que son distintos puntos 
                    # pero con una posicion identica
                    heuristicas[puntoA][puntoB] = 0.01
            else:
                heuristicas[puntoA][puntoB] = math.inf


    return heuristicas

def calculo_matriz_distancias(valores):
    """
    Calculo de distancias entre cada punto, todos con todos
    para generar la matriz de heuristicas
    :return:
    """
    distancias = np.zeros((len(valores)-1,len(valores)-1))
    for puntoA in range(0,len(valores)-1):
        for puntoB in range(0,len(valores)-1):
            # print(puntoA , puntoB)
            if puntoA != puntoB:
                # print(valores[puntoA] , " " , valores[puntoB])
                x = valores[puntoA][0] - valores[puntoB][0]
                y = valores[puntoA][1] - valores[puntoB][1]
                value = int(math.sqrt(x*x + y*y))
                if value != 0:
                    distancias[puntoA][puntoB] = value
                if value == 0:
                    # Como lo manejamos, debidoa  que son distintos puntos 
                    # pero con una posicion identica
                    distancias[puntoA][puntoB] = 0.0
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

def greedy_calculo_camino_optimo(distancias,punto_partida):

    # puntos_visitados_ordenados = list()
    # punto_inicial = 0

    # punto_actual = punto_inicial
    # # while len(puntos_visitados_ordenados) < len(distancias):

    # puntos_visitados_ordenados.append(0)

    # valor,indice = get_valor_minimo(distancias[0],puntos_visitados_ordenados)
    # print(distancias[0])
    # print(valor, indice)
    
    solucion = [0]
    puntos_pendientes = np.arange(0,len(distancias))
    
    
    
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
    
    # for indice in range(len(matriz_heuristica)):
    #     if indice not in camino_generado:
    #         # Se calculan los valores de probabilidad de cada punto no visitado
    #         feromona_arco = math.pow(matriz_feromonas[punto_ultimo][indice],alpha)
    #         heuristica_arco = math.pow(matriz_heuristica[punto_ultimo][indice],beta)
           
    #         probabilidad = (feromona_arco * heuristica_arco) / sumatorio
    #         probabilidades.append(probabilidad)
    #         indices_validos.append(indice)
        

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
    
    # print("long" ,len(indices_validos))
    # print("indice ele",indice_punto_elegido-1)
    
    return indices_validos[indice_punto_elegido-1]


def creacion_de_camino(matriz_feromonas,matriz_heuristica,punto_partida):
    
    camino_generado = []
    camino_generado.append(punto_partida)
    
    coste_camino_generado = 0
     
    
    for lon in range(len(matriz_feromonas)-1): 
    
    # while len(camino_generado) < len(matriz_feromonas):
        punto_ultimo = camino_generado[-1]
        
        punto_siguiente = calcular_punto_siguiente(camino_generado,matriz_feromonas,matriz_heuristica)
        
        valor = matriz_heuristica[punto_ultimo][punto_siguiente]

        coste_camino_generado += 1/valor         
        camino_generado.append(punto_siguiente)
    
    # Hay que cerrar ciclo, volver al punto de inicio??
    valor = matriz_heuristica[camino_generado[-1]][punto_partida]
    coste_camino_generado += (1/valor)
    camino_generado.append(punto_partida)
    
    return coste_camino_generado,camino_generado


def actualizar_matriz_feromonas(matriz_feromonas,caminos_hormigas,costes_caminos_hormigas,valor_evaporacion=0.1):
    # modificamos los valores de la matriz de feromonas, modificado ambas posiciones  
    
    
    variaciones =  np.ones((len(matriz_feromonas),len(matriz_feromonas)))*(1-valor_evaporacion)
    matriz_feromonas = matriz_feromonas*variaciones
    
    matriz_aportaciones = np.zeros((len(matriz_feromonas),len(matriz_feromonas)))
    
    for i in range(len(caminos_hormigas)-1):
        camino = caminos_hormigas[i]
        aporte = costes_caminos_hormigas[i]
        for k in range(len(camino)-1):
            pos = camino[k]
            pos_sig = camino[k+1]
            matriz_aportaciones[pos][pos_sig] += (1/aporte)
            matriz_aportaciones[pos_sig][pos] += (1/aporte)

        matriz_aportaciones[camino[-1]][camino[0]] += (1/aporte)
        matriz_aportaciones[camino[0]][camino[-1]] += (1/aporte)
    
    
    for puntoA in range(0,len(matriz_feromonas)):
        for puntoB in range(0,len(matriz_feromonas)):
            if puntoA != puntoB:
                matriz_feromonas[puntoA][puntoB] += matriz_aportaciones[puntoA][puntoB]
                # matriz_feromonas[puntoA][puntoB] += matriz_aportaciones[puntoB][puntoA]
                
                matriz_feromonas[puntoB][puntoA] += matriz_aportaciones[puntoB][puntoA]
                # matriz_feromonas[puntoB][puntoA] += matriz_aportaciones[puntoA][puntoB]
    
    return matriz_feromonas
        

def hormigas(problema = "ch130.tsp",n_hormigas=10,limite_iteracciones = 1000000000,valor_inicial_feronomas=1,punto_partida = 0):
    
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
    diff = 0
  
    iteraccion = 0
    while iteraccion < limite_iteracciones and (time.time() - inicio) < 60*6:
        eje_x.append(iteraccion)
        eje_y.append(coste_mejor_camino_global)
        
        costes_caminos_hormigas = []
        caminos_hormigas = []
        inicio_hormigas = time.time()
        for hormiga in range(n_hormigas):
            print("ite " , iteraccion , " hormiga " , hormiga)
            # Cada hormiga crea su recorrido
            
            coste_camino,camino_actual = creacion_de_camino(matriz_feromonas,matriz_heuristica,punto_partida)
            
            caminos_hormigas.append(camino_actual)
            costes_caminos_hormigas.append(coste_camino)
            # Nos quedamos con el camino si mejora al global y su coste
            if coste_camino < coste_mejor_camino_global:
                mejor_camino_global = camino_actual
                coste_mejor_camino_global = coste_camino
            
            print("coste " , coste_camino)
            
            diff =  time.time() - inicio
            print(diff , " tiempo")

        print("homtti tt",time.time() - inicio_hormigas)
        
        # Aplicamos la evaporación y el aporte 
        print("timp ",time.time() - inicio)
        matriz_feromonas = actualizar_matriz_feromonas(matriz_feromonas,caminos_hormigas,costes_caminos_hormigas)
        iteraccion += 1
        
    # print(mejor_camino_global , coste_mejor_camino_global)
    plt.plot(np.array(eje_x),np.array(eje_y))
    plt.show()
    return mejor_camino_global , coste_mejor_camino_global
    






random.seed(12385214)
np.random.seed(12385214)



# valor = 1/(280*6958)
# f =  "a280.tsp"  


valor = 1/(130*7579)
f = "ch130.tsp"

mejor_camino_global,coste_mejor_camino_global = hormigas(problema=f,valor_inicial_feronomas = valor)

print(mejor_camino_global,coste_mejor_camino_global)

datos = lectura_archivo(f)
distancias = calculo_matriz_distancias(datos)



p = mejor_camino_global
coste = 0
for i in range(len(p)-1):
    a = p[i]
    b = p[i+1]
    coste += distancias[a][b]
print(coste)





indices_obtenidos = np.array(mejor_camino_global)

eje_x = []
eje_y = []
for k in mejor_camino_global:
    eje_x.append(datos[k][0])
    eje_y.append(datos[k][1])

plt.plot(np.array(eje_x),np.array(eje_y))
plt.show()


