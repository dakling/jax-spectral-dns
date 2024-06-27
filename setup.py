from setuptools import setup

setup(
    name="jax-spectral-dns",
    version="0.1.0",
    description="A spectral DNS solver supporting automatic differentiation",
    url="https://gitlab.com/dakling/jax-optim",
    author="Dario Klingenberg",
    author_email="dsk34@cam.ac.uk",
    license="AGPL3",
    package_data={"jax_spectral_dns": ["py.typed"]},
    packages=["jax_spectral_dns"],
    install_requires=[
        "jax",
        "numpy<2",  # TODO change this at some point
        "scipy",
        "matplotlib",
        "pyvista",
        "jaxopt",
        "optax",
        "humanfriendly",
        "pyyaml",
    ],
    extras_require={
        "notifications": ["PyGObject"],
        "contributing": [
            "mypy",
            "pytest",
            "pytest-xdist",
            "black",
            "pre-commit",
        ],
    },
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GPL License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
