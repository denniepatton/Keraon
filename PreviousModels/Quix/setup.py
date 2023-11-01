import pathlib
from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).parent

VERSION = '0.1.0'
PACKAGE_NAME = 'Quix'
AUTHOR = 'Robert D Patton'
AUTHOR_EMAIL = 'rpatton@fredhutch.org'
URL = 'https://github.com/denniepatton/Quix'

LICENSE = 'GNU General Public License 3.0'
DESCRIPTION = 'Quix: a tool for estimating mixture proportions.'
LONG_DESCRIPTION = (HERE / "README.md").read_text()
LONG_DESC_TYPE = "text/markdown"

INSTALL_REQUIRES = ['numpy', 'pandas', 'scipy']

setup(name=PACKAGE_NAME,
      version=VERSION,
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      long_description_content_type=LONG_DESC_TYPE,
      author=AUTHOR,
      license=LICENSE,
      author_email=AUTHOR_EMAIL,
      url=URL,
      install_requires=INSTALL_REQUIRES,
      packages=find_packages())
