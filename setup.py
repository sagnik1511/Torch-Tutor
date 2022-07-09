from setuptools import setup, find_packages

VERSION = '0.0.2'
DESCRIPTION = 'Simple & Efficient PyTorch Trainer'

# Setting up
setup(
    name="torch_tutor",
    version=VERSION,
    author="sagnik1511 (Sagnik Roy)",
    author_email="<sagnik.jal00@gmail.com>",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=["numpy", "pandas"],
    keywords=['python', 'pytorch', 'training', 'logging'],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)
