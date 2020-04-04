from setuptools import setup

def readme():
    with open('README.rst') as f:
        return f.read()

setup(
    name='berp',
    version='.01',
    description='Module for handling the batch effect in sequencing data.',
    url='http://github.com/thekuhninator/berp',
    author='Roman Kuhn',
    author_email='roman.kuhn1@gmail.com',
    license='MIT',
    py_modules=['berp'],
    #packages=['berp'],
    install_requires=[
        'Click==7.0',
        'pandas==0.24.2',
        'scipy==1.3.0',
        'numpy==1.16.3',
        'sklearn==0.0',
        'matplotlib==3.1.0',
        'seaborn==0.10.0'
    ],
    entry_points='''
        [console_scripts]
        fart=berp_pipeline:cli
    ''',
    # dependency_links='' # this is where R packages will go?
    zip_safe=False)
