from setuptools import setup


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='bss-machinelearning',
    version='v0.4.1',
    author='Matthew Johnson, Noel Williams, Yisha Chen, Yuki-Jiayi Li',
    author_email='greenchicken1902@gmail.com, noelosdub@gmail.com',
    maintainer='Matthew Johnson',
    maintainer_email='greenchicken1902@gmail.com',
    description='Forecasting Demand for Bike Sharing Systems and Analysis with Python',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/GreenMachine582/BSS-MachineLearning',
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
)
