import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="causal_feature_selection",
    version="0.1",
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
                      ],
    python_requires='>=3.6',
)

# setup(
#     name="causal_tree_learn",
#     version="2.42",
#     author="Christopher Tran",
#     author_email="ctran29@uic.edu",
#     description="Python implementation of causal feature selection for heterogeneous treatment effect estimation",
#     long_description=long_description,
#     long_description_content_type="text/markdown",
#     url="https://github.com/edgeslab/CTL",
#     packages=find_packages(),
#     classifiers=[
#         "Programming Language :: Python :: 3",
#         "License :: OSI Approved :: MIT License",
#         "Operating System :: OS Independent",
#     ],
#     install_requires=['numpy',
#                       'scikit-learn',
#                       'scipy',
#                       "cdt",
#                       "fcit",
#                       "pandas",
#                       ],
#     python_requires='>=3.6',
#     ext_modules=ext_modules,
#     # cmdclass={'build_ext': build_ext},
#     cmdclass=cmdclass,
#     setup_requires=["cython", "numpy"],
#     package_data={"CTL.causal_tree": ["util_c.c", "util_c.pyx"]}
# )
