from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="deepguard",
    version="1.0.0",
    author="DeepGuard Team",
    author_email="deepguard@example.com",
    description="AI-powered deepfake detection system for images and videos",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/deepguard",
    packages=find_packages(where=".", exclude=["tests*", "notebooks*", "scripts*"]),
    python_requires=">=3.10",
    install_requires=requirements,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    entry_points={
        "console_scripts": [
            "deepguard-train=scripts.train:main",
            "deepguard-eval=scripts.evaluate:main",
            "deepguard-api=api.main:run",
        ],
    },
)
