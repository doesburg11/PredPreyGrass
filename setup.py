from setuptools import setup, find_packages

setup(
    name="PredPreyGrass",
    version="0.1",
    packages=find_packages(),
    description="Predator-Prey-Grass gridworld environment using PettingZoo, with dynamic deletion and spawning of partially observant agents.",
    author="P. van Doesburg",
    author_email="petervandoesburg11@gmail.com",
    url="https://github.com/doesburg11/predpreygrass",
    install_requires=[
        "pettingzoo",
        "stable_baselines3",
        "numpy",
        "pygame",
    ],
)