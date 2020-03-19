from setuptools import setup

setup(name='ap_perf',
      version='0.2.0',
      description='AP-Perf: Incorporating Generic Performance Metrics in Differentiable Learning',
      url='http://github.com/rizalzaf/ap_perf',
      author='Rizal Fathony',
      author_email='rfathony@cs.cmu.edu',
      license='MIT',
      packages=['ap_perf'],
      install_requires=[
          'numpy',
          'scipy',
          'torch',
          'numba'
      ],
      zip_safe=False)