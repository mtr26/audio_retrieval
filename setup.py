from setuptools import setup, find_packages

setup(
    name="audio_retrieval",
    version="0.1.0",
    author="Mohamed Traore",
    author_email="mohamed.trapro@gmail.com",
    description="A neural audio fingerprinting system for content-based audio retrieval",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.9.0",
        "librosa>=0.8.0",
        "faiss-cpu>=1.7.0",
        "numpy>=1.19.0",
        "tqdm>=4.60.0",
    ],
    extras_require={
        "gpu": ["faiss-gpu>=1.7.0"],
        "dev": ["pytest", "black", "flake8"],
    },
)
