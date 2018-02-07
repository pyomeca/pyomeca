from thirdparty import btk
# import thirdparty.btk as btk

reader = btk.btkAcquisitionFileReader()
reader.SetFilename("testc3d.c3d")
reader.Update()
acq = reader.GetOutput()

for marker in btk.Iterate(acq.GetPoints()):
    print(marker.GetValues())
