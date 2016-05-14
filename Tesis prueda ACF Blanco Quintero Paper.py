
# coding: utf-8

# Henry Steven Rueda Corredor. Rayos-X 
# 

# In[1]:

import numpy as np
import numpy.linalg as svd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import ternary
from numpy.linalg import matrix_rank


# In[2]:

##Carga el archivo fuente, omite el encabezado y coloca los caracteres como Racionales
Al=np.loadtxt("Al.txt", skiprows= 41, dtype=float)
Na=np.loadtxt("Na.txt", skiprows= 41, dtype= float)
Ca=np.loadtxt("Ca.txt", skiprows= 41, dtype= float)
Fe=np.loadtxt("Fe.txt", skiprows= 41, dtype= float)
Si=np.loadtxt("Si.txt", skiprows= 41, dtype= float)
Mg=np.loadtxt("Mg.txt", skiprows= 41, dtype= float)
Ba=np.loadtxt("Ba.txt", skiprows= 41, dtype= float)
K=np.loadtxt("K.txt", skiprows= 41, dtype= float)


# In[3]:

## Visualización de la matriz
print (Al)
print (Na)
print (Ca)
print (Fe)
print (Si)
print (Mg)
print (Ba)
print (K)


# In[4]:

Suma=Al+Na+Ca+Fe+Si+Mg+Ba+K
print (Suma)


# In[5]:

## Da el tamaño de la matriz 
print (np.size(Al))
print (Al.shape)


# In[6]:

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
plt.subplot(337)
plt.imshow(Ba)
plt.title('Ba')
plt.colorbar(orientation='horizontal')
plt.subplot(338)
plt.imshow(K)
plt.title('K')
plt.colorbar(orientation='horizontal')
#plt.show()
plt.savefig('graficas.png')


# In[7]:

figure=plt.figure(figsize=(10,10))
imgplot= plt.imshow(Suma,cmap='gray_r')
plt.colorbar(orientation='horizontal')
plt.title('Compuesta')
plt.savefig('Compuesta.png')


# In[8]:

figure=plt.figure(figsize=(10,10))
imgplot2= plt.imshow(K,cmap='gist_rainbow_r')
plt.colorbar(orientation='horizontal')
plt.title('K')
plt.savefig('K.png')


# In[9]:

Knorm=np.loadtxt("K.txt", skiprows= 41, dtype=float)
K.max(), K.min ()
vmax=400
vmin = 200
Knorm[Knorm > vmax] = np.nan
Knorm[Knorm< vmin] = np.nan
my_cmap = plt.cm.get_cmap()
my_cmap.set_under('w')
plt.imshow(Suma,cmap='gray')
plt.imshow(Knorm,cmap=my_cmap, interpolation='none',vmin=0.01)
plt.title('K')
plt.savefig('Knorm.png')


# In[10]:

figure=plt.figure(figsize=(10,10))
imgplot2= plt.imshow(Ba,cmap='gist_rainbow_r')
plt.colorbar(orientation='horizontal')
plt.title('Ba')
plt.savefig('Ba.png')


# In[11]:

Banorm=np.loadtxt("Ba.txt", skiprows= 41, dtype=float)
vmax=320
vmin= 24
Banorm[Banorm > vmax] = np.nan
Banorm[Banorm< vmin] = np.nan
my_cmap = plt.cm.get_cmap()
my_cmap.set_under('w')
plt.imshow(Suma,cmap='gray')
plt.imshow(Banorm,cmap=my_cmap, interpolation='none',vmin=0.01)
plt.title('Ba')
plt.savefig('Banorm.png')


# In[12]:

##Realiza la descomposición de una matriz de datos en un solo vector
Al.flatten()


# In[13]:

#Convierte todos los arrays en una nueva matriz 
Matriz_de_datos=(np.array([Al.flatten(),Na.flatten(),Ca.flatten(),Fe.flatten(),Si.flatten(),Mg.flatten(),Ba.flatten(),K.flatten()])).T
print (Matriz_de_datos)


# In[14]:

# Se debe saber la dimensión de la matriz nueva creada para de tal manera crear una nueva matriz para realizar la computación
print (Matriz_de_datos.shape)


# In[15]:

Matriz_creada= np.loadtxt("Matriz creada.txt", dtype=float)
print (Matriz_creada)


# In[16]:

# Multiplicar la Matriz de datos y la creada 
Matriz_mineral=np.dot(Matriz_de_datos,Matriz_creada)
print (Matriz_mineral)
print (Matriz_mineral.shape)


# In[20]:

#La matriz mineral anterior tiene en primera columna A, segunda C tercera F cuarta Mg y Fe quinta Qz, Sexta Plagioclasa
# Y se debe extraer cada una para devolver al tamaño original de la matriz
A=np.reshape(Matriz_mineral[:,0],Al.shape)
C=np.reshape(Matriz_mineral[:,1],Al.shape)
F=np.reshape(Matriz_mineral[:,2],Al.shape)
Mg_Fe=np.reshape(Matriz_mineral[:,3],Al.shape)
Qz=np.reshape(Matriz_mineral[:,4],Al.shape)
Plg=np.reshape(Matriz_mineral[:,5],Al.shape)
KNa=np.reshape(Matriz_mineral[:,6],Al.shape)
BaAlSiK=np.reshape(Matriz_mineral[:,7],Al.shape)


# In[21]:

print (A)
print (C)
print (F)
print (Mg_Fe)
print (Qz)
print (Plg)
print (KNa)
print (BaAlSiK)
print (A.shape)


# In[22]:

Suma_2=(A+C+F)
A_diagrama=np.divide(A*100,Suma_2)
C_diagrama=np.divide(C*100,Suma_2)
F_diagrama=np.divide(F*100,Suma_2)
Matriz_ACF=(np.array([F_diagrama.flatten(),A_diagrama.flatten(),C_diagrama.flatten()])).T
print (Matriz_ACF)
print (Matriz_ACF.shape)


# Cálculo del Single Value Decomposition

# In[ ]:



