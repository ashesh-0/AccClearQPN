def plot(dBZ2D):
    import matplotlib.pyplot as plt   
    plt.pcolor(dBZ2D,cmap='jet')
    plt.clim(0,50)
    plt.colorbar()
    plt.show()
    return()

import struct
import numpy as np


nx=561
ny=441
nz=21
dBZ = np.zeros((nx, ny, nz))


f1 = open('MREF3D21L.20180908.1300', 'rb')

head = f1.read(242)
Num = f1.read(4)
Num_i = struct.unpack('i', Num)[0]

for i in range(Num_i):
    sta=f1.read(4)
    
for k in range(nz):
    for i in range(nx):
        for j in range(ny):         
            data = f1.read(2)
            data_float = struct.unpack('h', data)[0]
            dBZ[i][j][k] = data_float

plot(dBZ[:,:,0]/10)
plot(dBZ[:,:,2]/10)
plot(dBZ[:,:,4]/10)
plot(dBZ[:,:,6]/10)
plot(dBZ[:,:,8]/10)
plot(dBZ[:,:,10]/10)

plt.pcolor(dBZ[300,:,:].transpose()/10)
plt.clim(0,50)
plt.colorbar()
plt.show()