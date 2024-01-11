import numpy as np
import matplotlib.pyplot as plt


# Funciones utilizadas:


# Función que permite calcular el área de cada elemento
def calculo_area_elemento(elemento, nodos_list):
    x1, y1 = nodos_list[elemento[0]]
    x2, y2 = nodos_list[elemento[1]]
    x3, y3 = nodos_list[elemento[2]]
    area = 0.5 * np.abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    return area


def calcular_matriz_B(elemento, nodos_list, area):
    xi, yi = nodos_list[elemento[0]]
    xm, ym = nodos_list[elemento[1]]
    xj, yj = nodos_list[elemento[2]]
    beta_i = yj - ym
    beta_j = ym - yi
    beta_m = yi - yj
    gamma_i = xm - xj
    gamma_j = xi - xm
    gamma_m = xj - xi
    B = (1 / (2 * area)) * np.array(
        [
            [beta_i, 0, beta_j, 0, beta_m, 0],
            [0, gamma_i, 0, gamma_j, 0, gamma_m],
            [gamma_i, beta_i, gamma_j, beta_j, gamma_m, beta_m],
        ]
    )
    return B


# Cálculo de Matriz D, esta debe ser entregada por el grupo 3
def calcular_matriz_D(E, v):
    D = (E / (1 - v**2)) * np.array([[1, v, 0], [v, 1, 0], [0, 0, ((1 - v) / 2)]])
    return D


def calculo_k(t, a, BT, D, B):
    k = (t * a) * BT @ D @ B
    return k


# Datos de la estructura
# Nodos de la estructura

nodos = [
    np.array([0, 0]),  # Nodo 0: coordenadas (x, y)
    np.array([0, 10]),  # Nodo 1
    np.array([10, 10]),  # Nodo 2
    np.array([10, 0]),
    np.array([20, 10]),  # Nodo 2
    np.array([20, 0]),  # Nodo 3  # Nodo 3
]
N_nodos = len(nodos)

# Elementos de la estructura
elementos = [[0, 1, 2], [0, 2, 3], [3, 2, 4], [3, 4, 5]]
N_elementos = len(elementos)

# Fuerzas que se aplican sobre la estructura

Fuerzas = [
    np.array([np.nan, -4000 + np.nan]),  # Nodo 0 (F_x, F_y) son las reacciones
    np.array([np.nan, np.nan]),
    np.array([0, -80]),  # Nodo 2
    np.array([0, 0]),  # Nodo  # Nodo 1
    np.array([0, -40]),  # Nodo 2
    np.array([0, 0]),  # Nodo 3

]

vector_f = []
for fuerza in Fuerzas:
    fx, fy = fuerza  # Separar los valores x e y de cada array en Fuerzas
    vector_f.extend([[fx], [fy]])  # Agregar los valores a la lista f
f = np.array(vector_f)

# Desplazamientos en la estructura

Desplazamientos = [
    np.array(
        [0, 0]
    ),  # Nodo 0 (u, v) son iguales a 0 debido a que son los apoyos y se limitan sus desplazamientos
    np.array([0, 0]),  # Nodo 1
    np.array([np.nan, np.nan]),  # Nodo 2
    np.array([np.nan, np.nan]),
    np.array([np.nan, np.nan]),  # Nodo 2
    np.array([np.nan, np.nan]),  # Nodo 3  # Nodo 3
]
vector_u = []
for desplazamiento in Desplazamientos:
    (
        u_x,
        v_y,
    ) = desplazamiento  # Separar los valores x e y de cada array en Desplazamientos
    vector_u.extend([[u_x], [v_y]])  # Agregar los valores a la lista u
u = np.array(vector_u)

# Espesor estructura
espesor = 0.01

area_elementos = []
for i in range(N_elementos):
    area_elementos.append(calculo_area_elemento(elementos[i], nodos))

D = calcular_matriz_D(210e9, 0.3)  # Esta matriz la da el grupo 3

# Matriz B para cada elemento de la estructuras:

B = []

for elemento in elementos:
    B.append(calcular_matriz_B(elemento, nodos, calculo_area_elemento(elemento, nodos)))

BT = []

for matriz_B in B:
    BT.append(np.transpose(matriz_B))

k_elementos = []  # Vector con k del elementos

for i in range(N_elementos):
    k_elementos.append(calculo_k(espesor, area_elementos[i], BT[i], D, B[i]))

K = np.zeros((N_nodos * 2, N_nodos * 2))

K = np.zeros((N_nodos * 2, N_nodos * 2))

for n in range(N_elementos):
    i, j, k = elementos[n]  # Nodos asociados al elemento
    matriz_k = k_elementos[n]  # Matriz de rigidez del elemento n en orden i, k, j

    # Ensamblaje de la matriz de rigidez del elemento en la matriz global K
    for p in range(2):
        for q in range(2):
            K[i * 2 + p, i * 2 + q] += matriz_k[p, q]
            K[i * 2 + p, k * 2 + q] += matriz_k[p, q + 2]
            K[i * 2 + p, j * 2 + q] += matriz_k[p, q + 4]

            K[k * 2 + p, i * 2 + q] += matriz_k[p + 2, q]
            K[k * 2 + p, k * 2 + q] += matriz_k[p + 2, q + 2]
            K[k * 2 + p, j * 2 + q] += matriz_k[p + 2, q + 4]

            K[j * 2 + p, i * 2 + q] += matriz_k[p + 4, q]
            K[j * 2 + p, k * 2 + q] += matriz_k[p + 4, q + 2]
            K[j * 2 + p, j * 2 + q] += matriz_k[p + 4, q + 4]

f_known_indices = np.where(~np.isnan(f[:, 0]))[0]
u_unknown_indices = np.where(np.isnan(u[:, 0]))[0]

# Seleccionar los valores conocidos en f y construir la submatriz K_known
f_known = f[f_known_indices]
K_known = K[np.ix_(f_known_indices, f_known_indices)]

# Resolver el sistema de ecuaciones para los valores desconocidos de u
u_unknown_solution = np.linalg.solve(K_known, f_known)

# Actualizar el vector u con los valores calculados
u[u_unknown_indices] = u_unknown_solution

print("Vector u solución:")
print(u)

desplazamientos_actualizados = []

for i in range(N_nodos):
    desplazamiento_actualizado = np.array([u[2 * i], u[2 * i + 1]])
    desplazamientos_actualizados.append(desplazamiento_actualizado)

nodos_actualizados = [
    nodos[i] + desplazamientos_actualizados[i] for i in range(N_nodos)
]

# Mostramos los desplazamientos actualizados
print("Nodos actualizados:")
for i, desplazamiento in enumerate(nodos_actualizados):
    print(f"Nodo {i}: {desplazamiento}")


Desplazamientos_elemento = []
for n in range(N_elementos):
    i = elementos[n][0]
    j = elementos[n][1]
    k = elementos[n][2]
    vector_ijk = []
    vector_ijk.extend(desplazamientos_actualizados[i])
    vector_ijk.extend(desplazamientos_actualizados[k])
    vector_ijk.extend(desplazamientos_actualizados[j])
    Desplazamientos_elemento.append(vector_ijk)

Deformaciones_elementos = []  # Lista para almacenar las deformaciones de los elementos

# Calcular las deformaciones de cada elemento
for i in range(N_elementos):
    deformaciones_i = (
        B[i] @ Desplazamientos_elemento[i]
    )  # Calcular las deformaciones del elemento i
    Deformaciones_elementos.append(
        deformaciones_i
    )  # Agregar las deformaciones a la lista

Tensiones_elementos = []  # Lista para almacenar las tensiones de los elementos

# Calcular las tensiones de cada elemento
for i in range(N_elementos):
    tensiones_i = (
        D @ Deformaciones_elementos[i]
    )  # Calcular las tensiones del elemento i
    Tensiones_elementos.append(tensiones_i)  # Agregar las tensiones a la lista

tensiones_von_mises = []
for tensiones in Tensiones_elementos:
    sigma_x, sigma_y, tau_xy = tensiones[0], tensiones[1], tensiones[2]
    sigma_vm = np.sqrt(
        sigma_x**2 + sigma_y**2 - sigma_x * sigma_y + 3 * tau_xy**2
    )
    tensiones_von_mises.append(sigma_vm)


print("Deformaciones del los elementos:")
for i, Deformaciones_elementos in enumerate(Deformaciones_elementos):
    print(f"Elemento {i}: {Deformaciones_elementos}")

print("Tensiones de los elementos en MPa:")
for i, Tensiones_elementos in enumerate(Tensiones_elementos):
    ajuste_unidadestension = Tensiones_elementos/100
    print(f"Elemento {i}: {ajuste_unidadestension}")
    
print("Tensiones de Von Mises de los elementos en MPa:")
for i, Tensiones_elementos in enumerate(tensiones_von_mises):
    ajuste_unidadtensionvm = Tensiones_elementos/100
    print(f"Elemento {i}: {ajuste_unidadtensionvm}")

# Crear la figura y el lienzo para la primera serie de gráficos
fig, ax = plt.subplots()

# Graficar los nodos y los elementos para la configuración inicial
for i, nodo in enumerate(nodos):
    ax.plot(nodo[0], nodo[1], "ro")  # Nodos iniciales
    ax.text(nodo[0], nodo[1], f"Nodo {i}", ha="right", va="bottom")  # Etiquetas

# Graficar los elementos con coordenadas iniciales
for elemento in elementos:
    x = [nodos[i][0] for i in elemento] + [nodos[elemento[0]][0]]
    y = [nodos[i][1] for i in elemento] + [nodos[elemento[0]][1]]
    ax.plot(x, y, "b-")

# Configuraciones adicionales del primer gráfico
plt.xlabel("Coordenada X")
plt.ylabel("Coordenada Y")
plt.title("Estructura con nodos y elementos (iniciales)")
plt.grid(True)

# Mostrar el primer gráfico
plt.show()

# Extraer coordenadas x e y de cada nodo
coord_x = [matriz[0][0] for matriz in nodos_actualizados]
coord_y = [matriz[1][1] for matriz in nodos_actualizados]

# Crear la figura y el lienzo para la primera serie de gráficos
fig, ax = plt.subplots(figsize=(6, 6))

# Graficar los nodos
ax.scatter(coord_x, coord_y, color="red")  # Graficar nodos como puntos rojos

# Etiquetar cada punto con el índice del nodo
for i, (x, y) in enumerate(zip(coord_x, coord_y)):
    ax.text(x, y, f"Nodo {i}", ha="right", va="bottom")

# Graficar los elementos con coordenadas iniciales
for elemento in elementos:
    x = [nodos_actualizados[i][0][0] for i in elemento] + [
        nodos_actualizados[elemento[0]][0][0]
    ]
    y = [nodos_actualizados[i][1][1] for i in elemento] + [
        nodos_actualizados[elemento[0]][1][1]
    ]
    ax.plot(x, y, "b-")

plt.xlabel("Coordenada X")
plt.ylabel("Coordenada Y")
plt.title("Gráfico de Nodos y Elementos")
plt.grid(True)
plt.show()

# Crear un mapa de colores para representar las tensiones de Von Mises
colores = plt.cm.plasma(np.array(tensiones_von_mises) / max(tensiones_von_mises))

# Crear la figura y el lienzo para los gráficos
fig, ax = plt.subplots(figsize=(6, 6))

# Graficar los elementos con colores representando las tensiones de Von Mises
for i, elemento in enumerate(elementos):
    # Obtener coordenadas x e y del elemento
    x = [nodos_actualizados[nodo][0][0] for nodo in elemento]
    y = [nodos_actualizados[nodo][1][1] for nodo in elemento]
    ax.fill(
        x, y, color=colores[i], alpha=0.5
    )  # Utilizar colores representativos de las tensiones

plt.xlabel("Coordenada X")
plt.ylabel("Coordenada Y")
plt.title("Gráfico de Nodos y Elementos con Tensiones de Von Mises")
plt.grid(True)

# Crear la barra de colores para mostrar la escala de tensiones
sm = plt.cm.ScalarMappable(
    cmap=plt.cm.plasma,
    norm=plt.Normalize(min(tensiones_von_mises), max(tensiones_von_mises)),
)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax)  # Asignar el eje (Axes) para el colorbar
cbar.set_label("Tensiones de Von Mises MPa")

plt.show()
