from scipy import datasets, signal
import torch
import numpy as np
import matplotlib.pyplot as plt

shape = [50, 50]
x_data, y_data = np.mgrid[0:shape[0], 0:shape[1]]
x_center, y_center = (shape[0]-1)/2, (shape[1]-1)/2
x, y = torch.tensor(x_data, dtype=torch.float32), torch.tensor(y_data, dtype=torch.float32)

distance = torch.sqrt((x-x_center)**2 + (y-y_center)**2)
dist = distance.unique()
diff = dist[1:] - dist[:-1]
print('min distance = ', diff.min())
print(dist)

fig, axes = plt.subplots(ncols=1, nrows=1, figsize=(50,3), dpi=300, constrained_layout=True)
axes.plot(dist, np.ones(dist.shape[0]), '.', color='red')
axes.vlines(np.linspace(start=0., stop=17.0, num=17*40 +1), ymin=0.5, ymax=1.5, colors='black')
plt.savefig('tmp.png')

Nx, Ny = torch.tensor(25), torch.tensor(25)
overSampling = 30

xp, yp = (Nx - 1)/2, (Ny - 1)/2
maxRadius = torch.round(torch.sqrt(((Nx-1) - xp)**2 + ((Ny-1) - yp)**2)) + 1
R = torch.linspace(start=0, end=maxRadius * overSampling - 1, steps=int(maxRadius * overSampling)) / overSampling

gridx = torch.linspace(start=0, end=Nx-1, steps=Nx)
gridy = torch.linspace(start=0, end=Ny-1, steps=Ny)
Y, X  = torch.meshgrid(gridx, gridy)

rPixel = torch.sqrt((X - xp)**2 + (Y - yp)**2)
index  = torch.floor(rPixel * overSampling).type(torch.int)
print(index)

fig, axes = plt.subplots(ncols=1, nrows=1, figsize=(5,5), dpi=300, constrained_layout=True)
axes.imshow(index)
plt.savefig('tmp.png')
