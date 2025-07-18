# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 20:57:37 2019

@author: Neoooli
"""

import gdal

list1 = ["byte","uint8","uint16","int16","uint32","int32","float32","float64","cint16","cint32","cfloat32","cfloat64"]
list2 = [gdal.GDT_Byte,gdal.GDT_Byte,gdal.GDT_UInt16,gdal.GDT_Int16,gdal.GDT_UInt32,gdal.GDT_Int32,gdal.GDT_Float32,gdal.GDT_Float64,gdal.GDT_CInt16,gdal.GDT_CInt32,gdal.GDT_CFloat32,gdal.GDT_CFloat64]

def imgread(path):
    img = gdal.Open(path)
    # col = img.RasterXSize #col
    # row = img.RasterYSize #row
    # img_arr = img.ReadAsArray(0,0,col,row) #设置读取图像的范围(宽/高)
    c = img.RasterCount  
    img_arr = img.ReadAsArray() #读取整幅图像
    if c>1:
        img_arr = img_arr.swapaxes(1,0)
        img_arr = img_arr.swapaxes(2,1)
    del img
    return img_arr.reshape(img_arr.shape[0],img_arr.shape[1],-1)
def imgwrite(path,narray):
    s=narray.shape
    narray=narray.reshape((s[0], s[1],-1))
    s=narray.shape
    dt_name=narray.dtype.name
    name=list(set(list1)&set([dt_name.lower()]))+["byte"]
    datatype=list2[list1.index(name[0])]

    driver = gdal.GetDriverByName('GTiff')
    dataset = driver.Create(path,s[1],s[0],s[2], datatype)

    for i in range(s[2]):
        dataset.GetRasterBand(i + 1).WriteArray(narray[:,:,i])
    del dataset
