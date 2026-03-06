"""
Setup script for Video Anonymizer Pro
"""

from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="video-anonymizer-pro",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Professional video anonymization tool for adult women in videos with children",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/video-anonymizer-pro",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Multimedia :: Video",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "video-anonymizer=main:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)