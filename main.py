from cmath import cos
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
                    distancias[puntoA][puntoB] = 1
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
    

    

def creacion_de_camino(matriz_feromonas,matriz_heuristica,punto_partida,valor_inicial_feronomas,sistema=False):
    
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
        
                    

def dibujar(fichero,camino):
    datos = lectura_archivo(fichero)

    eje_x = []
    eje_y = []
    for k in camino:
        eje_x.append(datos[k][0])
        eje_y.append(datos[k][1])

    plt.plot(np.array(eje_x),np.array(eje_y))
    plt.show()

def hormigas(problema = "ch130.tsp",n_hormigas=10,limite_iteracciones = 100_000,minutos_limite=1,valor_inicial_feronomas=1,punto_partida = 0,elite = 0,verbose = False):
    
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
    
    while iteraccion < limite_iteracciones and (time.time() - inicio) < 60*minutos_limite:
        
        # if verbose:
        #     eje_x.append(iteraccion)
        #     eje_y.append(coste_mejor_camino_global)
        
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
            
            print(" tiempo ",time.time() - inicio , " iteracion " , iteraccion)
        
        # Aplicamos la evaporación y el aporte 
        
        matriz_feromonas = actualizar_matriz_feromonas(matriz_feromonas,caminos_hormigas,costes_caminos_hormigas,coste_mejor_camino_global=coste_mejor_camino_global,mejor_camino_global=mejor_camino_global,elite=elite)
        
        iteraccion += 1
        
    if verbose:
        dibujar(problema,mejor_camino_global)
        plt.plot(np.array(eje_x),np.array(eje_y))
        plt.show()
    return mejor_camino_global , coste_mejor_camino_global

def sistema_colonia_hormigas(problema = "ch130.tsp",n_hormigas=10,limite_iteracciones = 100_000,minutos_limite=1,valor_inicial_feronomas=1,punto_partida = 0,verbose = False):
    
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
            
            
        print(" tiempo ",time.time() - inicio , " iteracion " , iteraccion)
        # Aplicamos la evaporación y el aporte 
        matriz_feromonas = evaporacion_global(matriz_feromonas,mejor_camino_global,coste_mejor_camino_global)
        
        iteraccion += 1
        
    if verbose:
        dibujar(problema,mejor_camino_global)
        plt.plot(np.array(eje_x),np.array(eje_y))
        plt.show()
    return mejor_camino_global , coste_mejor_camino_global








semilla = 503546630
semilla = random.randint(41689191,999999999)

random.seed(semilla)
np.random.seed(semilla)


# valor = 1/(130*7579)
# valor = 1/(280*6958)
# f =  "a280.tsp"  


valor = 1/(130*7579)
f = "ch130.tsp"


# mejor_camino_global,coste_mejor_camino_global_elite = hormigas(problema=f,valor_inicial_feronomas = valor,minutos_limite=6,elite=15,verbose=False)

mejor_camino_global,coste_mejor_camino_global_elite = sistema_colonia_hormigas(problema=f,valor_inicial_feronomas = valor,minutos_limite=6,verbose=True)

print("hormiga " ,coste_mejor_camino_global_elite)  

print("semilla utilizada ", semilla)

# dibujar(f,mejor_camino_global)




# datos = lectura_archivo(f)
# distancias = calculo_matriz_distancias(datos)

# p = mejor_camino_global
# coste = 0
# for i in range(len(p)-1):
#     a = p[i]
#     b = p[i+1]
#     coste += distancias[a][b]
# print(coste)





