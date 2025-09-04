from setuptools import setup, find_packages

setup(
    name='arDCA_paths',
    version='0.1.1',
    author='Jeanne Trinquier, Lorenzo Rosset, Francesco Zamponi, Martin Weigt',
    maintainer='Lorenzo Rosset',
    author_email='rosset.lorenzo@gmail.com',
    description='Python implementation of autoregressive Direct Coupling Analysis',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/spqb/arDCA',
    packages=find_packages(include=['arDCA_paths', 'arDCA_paths.*']),
    include_package_data=True,
    python_requires='>=3.10',
    license_files=["LICENSE"],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
    ],
    entry_points={
        'console_scripts': [
            'arDCA_paths=arDCA_paths.cli:main',
        ],
    },
    install_requires=[
        "adabmDCA>=0.3.3",
    ],
)