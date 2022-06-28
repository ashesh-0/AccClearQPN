import numpy as np
from netCDF4 import Dataset  # http://code.google.com/p/netcdf4-python/
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap


file='QPESUMS_rain_2018/201809/20180908_1300.nc'
#file='20140101_1400.nc'
data=Dataset(file, 'r')
RR=data.variables['qperr'][:]
lon=data.variables['lon'][:]
lat=data.variables['lat'][:]

lon2d, lat2d = np.meshgrid(lon, lat)

m = Basemap(projection='merc', llcrnrlat=21, urcrnrlat=26,\
            llcrnrlon=119, urcrnrlon=123, resolution='h', lat_ts=20)
m.drawcoastlines()
m.drawmapboundary()
x, y = m(lon2d, lat2d)
m.drawmeridians([118,120,122,124])
m.drawparallels([19,21,23,25,27])
m.contourf(x, y, RR, cmap=plt.cm.jet)
plt.colorbar()
plt.show()
