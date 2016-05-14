
# coding: utf-8

# Henry Steven Rueda Corredor. Rayos-X 
# 

# In[1]:

import numpy as np
import numpy.linalg as svd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from numpy.linalg import matrix_rank
import ternary


# In[2]:

##Carga el archivo fuente, omite el encabezado y coloca los caracteres como Racionales
Al=np.loadtxt("Al.txt", skiprows= 41, dtype=float)
Na=np.loadtxt("Na.txt", skiprows= 41, dtype= float)
Ca=np.loadtxt("Ca.txt", skiprows= 41, dtype= float)
Fe=np.loadtxt("Fe.txt", skiprows= 41, dtype= float)
Si=np.loadtxt("Si.txt", skiprows= 41, dtype= float)
Mg=np.loadtxt("Mg.txt", skiprows= 41, dtype= float)


# In[3]:

## Visualización de la matriz
print (Al)
print (Na)
print (Ca)
print (Fe)
print (Si)
print (Mg)


# In[4]:

Suma=Al+Na+Ca+Fe+Si+Mg
print (Suma)


# In[5]:

## Da el tamaño de la matriz
print (np.size(Al))


# In[6]:

## Muestra el numero de filas y columnas de la matriz (fila,columna). Del archivo origen leído como (Line,Step)
tamaño = Al.shape
tamaño2 = Na.shape  
print (tamaño,tamaño2)


# In[7]:

fig=plt.figure(figsize=(14,14))
plt.subplot(331)
plt.imshow(Al)
plt.title('Al')
plt.colorbar(orientation='horizontal')
plt.subplot(332)
plt.imshow(Na)
plt.title('Na')
plt.colorbar(orientation='horizontal')
plt.subplot(333)
plt.imshow(Ca)
plt.title('Ca')
plt.colorbar(orientation='horizontal')
plt.subplot(334)
plt.imshow(Fe)
plt.title('Fe')
plt.colorbar(orientation='horizontal')
plt.subplot(335)
plt.imshow(Si)
plt.title('Si')
plt.colorbar(orientation='horizontal')
plt.subplot(336)
plt.imshow(Mg)
plt.title('Mg')
plt.colorbar(orientation='horizontal')
#plt.show()
plt.savefig('graphs.png')


# In[8]:

figure=plt.figure(figsize=(10,10))
imgplot= plt.imshow(Suma,cmap='gray')
plt.colorbar(orientation='horizontal')
plt.title('Compuesta')
plt.savefig('Compuesta.png')


# In[9]:

figure=plt.figure(figsize=(10,10))
imgplot2= plt.imshow(Al)
plt.colorbar(orientation='horizontal')
plt.title('Al')
plt.savefig('Al.png')


# In[10]:

##Realiza la descomposición de una matriz de datos en un solo vector
Al.flatten()


# In[11]:

#Convierte todos los arrays en una nueva matriz 
Matriz_de_datos=(np.array([Al.flatten(),Na.flatten(),Ca.flatten(),Fe.flatten(),Si.flatten(),Mg.flatten()])).T
print (Matriz_de_datos)


# In[12]:

# Se debe saber la dimensión de la matriz nueva para de tal manera crear una nueva matriz para realizar la computación
print (Matriz_de_datos.shape)


# In[13]:

Matriz_creada= np.loadtxt("Matriz creada.txt", dtype=float).T
print (Matriz_creada)


# In[14]:

# Multiplicar la Matriz de datos y la creada 
Matriz_mineral=np.dot(Matriz_de_datos,Matriz_creada)
print (Matriz_mineral)
print (Matriz_mineral.shape)


# In[15]:

#La matriz mineral anterior tiene en primera columna A, segunda C tercera F cuarta Mg y Fe quinta Qz, Sexta Plagioclasa
# Y se debe extraer cada una para devolver al tamaño original de la matriz
A=np.reshape(Matriz_mineral[:,0],Al.shape)
C=np.reshape(Matriz_mineral[:,1],Al.shape)
F=np.reshape(Matriz_mineral[:,2],Al.shape)
Mg_Fe=np.reshape(Matriz_mineral[:,3],Al.shape)
Qz=np.reshape(Matriz_mineral[:,4],Al.shape)
Plg=np.reshape(Matriz_mineral[:,5],Al.shape)


# In[16]:

print (A)
print (C)
print (F)
print (Mg_Fe)
print (Qz)
print (Plg)
print (A.shape)


# In[17]:

fig=plt.figure(figsize=(14,14))
plt.subplot(331)
plt.imshow(A)
plt.title('A')
plt.colorbar(orientation='horizontal')
plt.subplot(332)
plt.imshow(C)
plt.title('C')
plt.colorbar(orientation='horizontal')
plt.subplot(333)
plt.imshow(F)
plt.title('F')
plt.colorbar(orientation='horizontal')
plt.subplot(334)
plt.imshow(abs(Mg_Fe))
plt.title('Mg_Fe')
plt.colorbar(orientation='horizontal')
plt.subplot(335)
plt.imshow(Qz)
plt.title('Qz')
plt.colorbar(orientation='horizontal')
plt.subplot(336)
plt.imshow(Plg)
plt.title('Plg')
plt.colorbar(orientation='horizontal')
#plt.show()
plt.savefig('Graphs minerals.png')


# In[18]:

#Recalcula la matriz A, C y F como porcentaje puntual
Suma_2=(A+C+F)
print (Suma_2)


# In[19]:

A_diagrama=np.divide(A*100,Suma_2)
C_diagrama=np.divide(C*100,Suma_2)
F_diagrama=np.divide(F*100,Suma_2)
print (A_diagrama)
print (C_diagrama)
print (F_diagrama)


# In[20]:

Matriz_ACF=(np.array([F_diagrama.flatten(),A_diagrama.flatten(),C_diagrama.flatten()])).T
print (Matriz_ACF)
print (Matriz_ACF.shape)


# In[21]:

#Ternary diagram, cabe aclarar que para la matriz los datos deben ser puestos como FAC
scale = 100
figure, tax = ternary.figure(scale=scale,)
fontsize=20
tax.set_title("ACF", fontsize=fontsize)
tax.left_axis_label("C",fontsize=fontsize, position=(-0.01,0.05,0),rotation=0)
tax.right_axis_label("A", fontsize=fontsize, position=(-0.06,1.12,0),rotation=0)
tax.bottom_axis_label("F", fontsize=fontsize, position=(0.93,0.05,0))
tax.boundary(linewidth=2.0)
tax.gridlines(multiple=10, color="blue")
tax.scatter(Matriz_ACF,color='purple', label='Qz, Mg y Fe, Plagioclasa', s=0.2)
tax.legend()
tax.ticks(axis='lbr', linewidth=1, multiple=10, clockwise=True)
#tax.show()
plt.savefig('ACF.png')


# Cálculo del Single Value Decomposition

# In[22]:

Matriz_ACF.shape


# In[23]:

U,s,V=np.linalg.svd(A)
S = np.zeros((A.shape))
Vt=np.transpose(V)
S[:672, :672] = np.diag(s)


# In[24]:

U1,s1,V1=np.linalg.svd(C)
S1 = np.zeros((C.shape))
Vt1=np.transpose(V1)
S1[:672, :672] = np.diag(s1)


# In[25]:

U2,s2,V2=np.linalg.svd(F)
S2 = np.zeros((F.shape))
Vt2=np.transpose(V2)
S2[:672, :672] = np.diag(s2)


# In[ ]:



