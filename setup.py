from setuptools import setup

setup(
    name="diffbayes",
    version="0.0",
    description="differentiable bayesian filtering",
    url="http://github.com/brentyi/diffbayes",
    author="brentyi",
    author_email="brentyi@berkeley.edu",
    license="BSD",
    packages=["diffbayes"],
    install_requires=["fannypack", "pytest", "scipy", "torch", "tqdm"],
)
