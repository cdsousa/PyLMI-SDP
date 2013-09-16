from os.path import exists
from setuptools import setup

setup(
    name='PyLMI-SDP',
    version='0.2',
    author='Cristovao D. Sousa',
    author_email='crisjss@gmail.com',
    description=('Symbolic linear matrix inequalities (LMI) and semi-definite'
                 'programming (SDP) tools for Python'),
    license='BSD',
    keywords='LMI SDP',
    url='http://github.com/cdsousa/PyLMI-SDP',
    packages=['lmi_sdp'],
    long_description=open('README.md').read() if exists('README.md') else '',
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Manufacturing',
        'Intended Audience :: Science/Research',
    ],
)
