from setuptools import setup, find_packages

setup(
   name='nnetart',
   version='0.1',
   description='Generate random art with neural networks',
   author='Tuan Le',
   author_email='tuanle@hotmail.de',
   packages=find_packages(), install_requires=['torch', 'numpy', 'matplotlib', 'typing', 'seaborn']
  )
