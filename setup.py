from setuptools import setup, find_packages

setup(
    name="weather_client",
    version="0.1.0",
    description="A simple weather API client",
    packages=find_packages(),
    install_requires=[
        "requests>=2.25.0",
    ],
    python_requires=">=3.6",
)
