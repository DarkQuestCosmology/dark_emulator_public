import re,os,sys

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

def get_requirements():
    fname = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    with open(fname, 'r') as f:
        requirements = [line.rstrip() for line in f]
    return requirements

def find_from_doc(what='version'):
    f = open(os.path.join(os.path.dirname(__file__), 'dark_emulator/__init__.py')).read()
    match = re.search(r"^__%s__ = ['\"]([^'\"]*)['\"]"%(what), f, re.M)
    if match:
        return match.group(1)
    raise RuntimeError("Unable to find %s string."%what)

packages = ['dark_emulator',
            'dark_emulator/darkemu','dark_emulator/darkemu/gp',
            'dark_emulator/model_hod',
            'dark_emulator/pyfftlog_interface']

setup(
    name='dark_emulator',
    version=find_from_doc('version'),
    description='dark emulator package',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url=find_from_doc('url'),
    author=find_from_doc('author'),
    author_email='takahiro.nishimichi@yukawa.kyoto-u.ac.jp, hironao.miyatake@iar.nagoya-u.ac.jp',
    keywords=['cosmology', 'large scale structure', 'halo', 'gaussian process', 'machine learning'],
    packages=packages,
    include_package_data=True,
    install_requires=get_requirements(),
    classifiers=['Programming Language :: Python :: 3.6'],
)
