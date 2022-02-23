# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 19:11:17 2020

@author: mendez
"""

from importlib import import_module

fail = False
required_modules = ['latex' ,'shutil', 'numpy', 'matplotlib', 'pandas', 'scipy', 'imageio',\
                   'tensorflow', 'torch' ,'deap' ,'sklearn' ,'gym', 'skopt']

for mod in required_modules:
    try:
        import_module(mod)
        print("{} available".format(mod))
    except ImportError:
        print("{} is not available".format(mod))
        fail = True

print()

if fail:
    print("Test set failed")
else:
    print('Test set passed')
