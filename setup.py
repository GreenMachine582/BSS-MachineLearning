from setuptools import setup, find_packages
import subprocess
import os


def get_version():
    latest_version = (
        subprocess.run(["git", "describe", "--tags"], stdout=subprocess.PIPE)
        .stdout.decode("utf-8")
        .strip()
    )

    if "-" in latest_version:
        x = latest_version.split("-")
        v, i, s = x[0], x[-2], x[-1]
        if len(x) == 2:
            i = 0
        return f"{v}+{i}.git.{s}"
    return latest_version


version = get_version()

assert "-" not in version
assert "." in version

assert os.path.isfile("BSS\\version.py")
with open("BSS\\VERSION", "w", encoding="utf-8") as fh:
    fh.write("%s\n" % version)

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='bss-machinelearning',
    version=version,
    author='Matthew Johnson, Noel Williams, Yisha Chen, Yuki-Jiayi Li',
    author_email='greenchicken1902@gmail.com',
    maintainer='Matthew Johnson',
    maintainer_email='greenchicken1902@gmail.com',
    description='Forecasting Demand for Bike Sharing Systems and Analysis with Python',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/GreenMachine582/BSS-MachineLearning',
    packages=find_packages(),
    package_data={'BSS': ['VERSION']},
    include_package_data=True,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Programming Language :: Python',
        'Operating System :: OS Independent',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3 :: Only'
    ],
    keywords='bss, machine-learning, cross-validation, neural-network, regression, clustering',
    python_requires='>=3.10, <4',
    entry_points={},
    install_requires=[],
)
