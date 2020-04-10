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
    #dependency_links=['https://github.com/theislab/kBET'],
#packages=['berp'],
    install_requires=[
        'Click==7.0',
        'pandas==0.24.2',
        'scipy==1.3.0',
        'numpy==1.16.3',
        'sklearn==0.0',
        'seaborn==0.8.1',
        'matplotlib==3.0.1',
        'rpy2'
        # need to fix this fucking garbage. maybe have them just install it with anaaconda
    ],
    entry_points='''
        [console_scripts]
        berp=berp_pipeline:cli
    ''',
    # dependency_links='' # this is where R packages will go?
    zip_safe=False)