from pyomeca.thirdparty import pyorbdl


m = pyorbdl.new("data/pyomecaman.s2mMod")
b = pyorbdl.nMarkers(m)
print(b)
