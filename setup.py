#!/usr/bin/env python3
"""
Setup script for SMART-TRIP sistema multi-paradigma
"""

from setuptools import setup, find_packages

# Read requirements
with open('requirements.txt', 'r', encoding='utf-8') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

# Read README  
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="smart-trip",
    version="1.0.0",
    author="Antonio Colamartino", 
    author_email="a.colamartino6@studenti.uniba.it",
    description="Sistema Multi-paradigma per Analisi e Raccomandazione Tragitti Intelligenti",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Tony0380/Smart_Trip",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education", 
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9", 
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'smart-trip=intelligent_travel_planner:main',
        ],
    },
    include_package_data=True,
    package_data={
        'prolog_kb': ['*.pl'],
    },
    keywords="artificial intelligence, machine learning, search algorithms, bayesian networks, prolog, travel planning",
    project_urls={
        "Source": "https://github.com/Tony0380/Smart_Trip",
    },
)