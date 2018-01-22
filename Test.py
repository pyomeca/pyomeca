import pyomeca
import numpy as np

# pyomeca.data.load_data('coucou.csv')


a = pyomeca.types.Vectors3d('coucou', np.array(3))
print(type(a))