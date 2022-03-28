import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="causal_feature_selection",
    version="0.2",
    author="Christopher Tran",
    author_email="ctran29@uic.edu",
    description="Python implementation of causal feature selection for heterogeneous treatment effect estimation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/edgeslab/causal_feature_selection",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=['numpy',
                      'scikit-learn',
                      'scipy',
                      "cdt",
                      "fcit",
                      "pandas",
                      "networkx",
                      "pyCausalFS",
                      ],
    python_requires='>=3.6',
)