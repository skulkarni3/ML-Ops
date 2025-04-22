from setuptools import setup, find_packages

setup(
    name="mlops", 
    version="0.1",
    packages=find_packages(),  
    install_requires=[  
        'mlflow==2.15.1',
        'numpy==1.26.4',
        'pandas==2.2.2',
        'scikit-learn==1.5.1'
    ],
    description="Sets up a virtual pyenv environment for ML Ops course", 
)
