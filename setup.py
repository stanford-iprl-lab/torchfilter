from setuptools import find_packages, setup

setup(
    name="torchfilter",
    version="0.0",
    description="differentiable bayesian filtering",
    url="http://github.com/stanford-iprl-lab/torchfilter",
    author="brentyi",
    author_email="brentyi@berkeley.edu",
    license="BSD",
    packages=find_packages(),
    install_requires=[
        "fannypack",
        "hypothesis",
        "overrides",
        "pytest",
        "scipy",
        "torch",
        "tqdm",
    ],
)
