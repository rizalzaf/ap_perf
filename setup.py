from setuptools import setup

setup(name='ap_perf',
      version='0.1.0',
      description='AP-Perf: Incorporating Generic Performance Metrics in Differentiable Learning',
      url='http://github.com/rizalzaf/ap_perf',
      author='Rizal Fathony',
      author_email='rizal@fathony.com',
      license='MIT',
      packages=['ap_perf'],
      install_requires=[
          'numpy',
          'scipy',
          'torch',
          'numba'
      ],
      zip_safe=False)