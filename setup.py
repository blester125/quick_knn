import re
from setuptools import setup, find_packages

def get_version(project_name):
    regex = re.compile(r"^__version__ = '(\d+\.\d+\.\d+(?:a|b|rc)?(?:\d)*?)'$")
    with open(f"{project_name}/__init__.py") as f:
        for line in f:
            m = regex.match(line)
            if m is not None:
                return m.groups(1)[0]

class About(object):
    NAME='quick_knn'
    VERSION=get_version(NAME)
    AUTHOR='blester125'
    EMAIL=f'{AUTHOR}@gmail.com'
    URL=f'https://github.com/{AUTHOR}/{NAME}'
    DL_URL=f'{URL}/archive/{VERSION}.tar.gz'
    LICENSE='MIT'
    DESCRIPTION='LSH for Jaccard and Cosine Similarity'

ext_modules = [
]

setup(
    name=About.NAME,
    version=About.VERSION,
    description=About.DESCRIPTION,
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    author=About.AUTHOR,
    author_email=About.EMAIL,
    url=About.URL,
    download_url=About.DL_URL,
    license=About.LICENSE,
    python_requires='>=3.6',
    packages=find_packages(),
    package_data={
        'quick_knn': [
        ],
    },
    include_package_data=True,
    install_requires=[
        'numpy'
    ],
    setup_requires=[
    ],
    extras_requires={
        'test': ['pytest'],
    },
    keywords=["Data Mining"],
    ext_modules=ext_modules,
    classifiers={
        'Development Status :: 3 - Alpha',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Scientific/Engineering',
    },
)
