from setuptools import setup, find_packages
# the analysis was done with Python version 3.7.2.

install_requires = ['numpy',
                    'matplotlib',
                    'tqdm',
                    'joblib',
                    'scikit-image',
                    'tensorflow',
                    'spams',  # not available for Windows
                    ]

setup(name='stain_san',
      version='0.0.1',
      description='Code to reproduce Stain SAN',
      author='Taebin Kim',
      author_email='taebinkim@unc.edu',
      license='MIT',
      packages=find_packages(),
      python_requires=">=3.7",
      install_requires=install_requires,
      zip_safe=False)