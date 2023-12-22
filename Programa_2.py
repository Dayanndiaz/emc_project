import numpy as np
import aaaaa as aa

# Datos de la estructura

# Nodos de la estructura

nodos = [
    np.array([0, 0]),  # Nodo 0: coordenadas (x, y)
    np.array([0, 10]),  # Nodo 1
    np.array([20, 10]),  # Nodo 2
    np.array([20, 0]),  # Nodo 3
]
N_nodos = len(nodos)
# print("Numero de nodos: ", N_nodos)
Fuerzas = [
    np.array([np.nan, np.nan]),  # Nodo 0 (F_x, F_y) son las reacciones
    np.array([np.nan, np.nan]),  # Nodo 1
    np.array([5000, 0]),  # Nodo 2
    np.array([5000, 0]),  # Nodo 3
]

Desplazamientos = [
    np.array(
        [0, 0]
    ),  # Nodo 0 (u, v) son iguales a 0 debido a que son los apoyos y se limitan sus desplazamientos
    np.array([0, 0]),  # Nodo 1
    np.array([np.nan, np.nan]),  # Nodo 2
    np.array([np.nan, np.nan]),  # Nodo 3
]

# Definir los elementos triangulares utilizando los nodos
# elemento_1 = (nodos[0], nodos[1], nodos[2])
# elemento_2 = (nodos[0], nodos[2], nodos[3])
# elemento_1 = [0, 1, 2]
# elemento_2 = [0, 2, 3]
elementos = [[0, 1, 2], [0, 2, 3]]
N_elementos = len(elementos)


def calculo_area_elemento(elemento, nodos_list):
    x1, y1 = nodos_list[elemento[0]]
    x2, y2 = nodos_list[elemento[1]]
    x3, y3 = nodos_list[elemento[2]]
    area = 0.5 * np.abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    return area


espesor = 1

area_elementos = []
for i in range(N_elementos):
    area_elementos.append(calculo_area_elemento(elementos[i], nodos))

# print("Areas :", area_elementos)

# area_elementos = [area_elemento(elemento_1, nodos), area_elemento(elemento_2, nodos)]


def calcular_matriz_B(elemento, nodos_list, area):
    x1, y1 = nodos_list[elemento[0]]
    x2, y2 = nodos_list[elemento[1]]
    x3, y3 = nodos_list[elemento[2]]
    # area = 0.5 * np.abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    beta_1 = y3 - y2
    beta_2 = y2 - y1
    beta_3 = y1 - y3
    gamma_1 = x2 - x3
    gamma_2 = x1 - x2
    gamma_3 = x3 - x1
    B = (1 / (2 * area)) * np.array(
        [
            [beta_1, 0, beta_3, 0, beta_2, 0],
            [0, gamma_1, 0, gamma_3, 0, gamma_2],
            [gamma_1, beta_1, gamma_3, beta_3, gamma_2, beta_2],
        ]
    )
    return B


def calcular_matriz_D(E, v):
    D = (E / (1 - v**2)) * np.array([[1, v, 0], [v, 1, 0], [0, 0, ((1 - v) / 2)]])
    return D


D = calcular_matriz_D(30e6, 0.3)  # Esta matriz la da el grupo 3
B = []

for elemento in elementos:
    B.append(calcular_matriz_B(elemento, nodos, calculo_area_elemento(elemento, nodos)))

# print("Vector B", B)
# print("Vector B1", B[0])
# print("Vector B2", B[1])
BT = []

for matriz_B in B:
    BT.append(np.transpose(matriz_B))

print("Vector BT", BT)
print("Vector BT1", BT[0])
print("Vector BT2", BT[1])


# D_elemento_1 = calcular_matriz_D(30e6, 0.3)  # Esta matriz la da el grupo 3
# B_elemento_1 = calcular_matriz_B(elemento_1)
# BT_elemento_1 = np.transpose(B_elemento_1)
# k_elemento = tA[BT][D][B]
def calculo_k(t, a, BT, D, B):
    k = (t * a) * BT @ D @ B
    return k


k_elementos = []  # Vector con k del elementos

for i in range(N_elementos):
    k_elementos.append(calculo_k(espesor, area_elementos[i], BT[i], D, B[i]))

print("Matriz k:", k_elementos)

K = np.zeros((N_nodos * 2, N_nodos * 2))


for n in range(N_elementos):
    i = elementos[n][0]
    j = elementos[n][1]
    k = elementos[n][2]
    matriz_k = k_elementos[n]

    K[i * 2, i * 2] = (
        K[i * 2, i * 2] + matriz_k[0, 0]
    )  # 2 corresponde a las dimennsiones del problema
    K[i * 2, i * 2 + 1] = K[i * 2, i * 2 + 1] + matriz_k[0, 1]
    K[i * 2 + 1, i * 2] = K[i * 2 + 1, i * 2] + matriz_k[1, 0]
    K[i * 2 + 1, i * 2 + 1] = K[i * 2 + 1, i * 2 + 1] + matriz_k[1, 1]

    # print(i, j, k, matriz_k)
    print(matriz_k[0, 0])
print(K)
