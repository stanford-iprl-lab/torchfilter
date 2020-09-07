from setuptools import setup

setup(
    name="torchfilter",
    version="0.0",
    description="differentiable bayesian filtering",
    url="http://github.com/stanford-iprl-lab/torchfilter",
    author="brentyi",
    author_email="brentyi@berkeley.edu",
    license="BSD",
    packages=["torchfilter"],
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
