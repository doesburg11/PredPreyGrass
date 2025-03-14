from setuptools import setup, find_packages

setup(
    name="predpreygrass",
    version="0.1",
    packages=find_packages(),
    description="Predator-Prey-Grass gridworld environment, with dynamic deletion and spawning of partially observant agents.",
    author="Peter van Doesburg",
    author_email="petervandoesburg11@gmail.com",
    url="https://github.com/doesburg11/predpreygrass",
    install_requires=[
        "pettingzoo",
        "stable_baselines3",
        "numpy",
        "pygame",
    ],
)