from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='RESCO',
    version='1.1.5',
    packages=find_packages(),
    install_requires=open('requirements.txt').read().splitlines(),
    url='https://github.com/KMASAHIRO/RESCO',
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    description='Evaluating Traffic Light RL Controller using RESCO benchmark.',
    python_requires=">=3.6.9"
)
