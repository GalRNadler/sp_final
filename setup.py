from setuptools import Extension, setup

module = Extension("symnmfmodule", sources=['symnmfmodule.c', 'symnmf.c'])
setup(name='symnmfmodule',
     version='1.0',
     description='Python wrapper for C extension',
     ext_modules=[module])
