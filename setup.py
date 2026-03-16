import os
import toml
from setuptools import setup, find_packages

# Load pyproject.toml
pyproject = toml.load("pyproject.toml")
project = pyproject["project"]

# Metadata
name = project["name"]
version = project["version"]
description = project["description"]
long_description = open(project["readme"]).read()
long_description_content_type = "text/markdown"
author = project["authors"][0]["name"] if "authors" in project else "Your Name"
author_email = project["authors"][0]["email"] if "authors" in project else "you@example.com"
requires_python = project["requires-python"]
url = project.get("homepage", "https://github.com/yourusername/neuralvoid")

# Determine environment (dev or prod)
env = os.environ.get("INSTALL_ENV", "prod")  # default to production

# Base dependencies
install_requires = list(project.get("dependencies", []))

# Add optional dependencies based on environment
optional_deps = project.get("optional-dependencies", {})
install_requires += optional_deps.get(env, [])

# Scripts
scripts = project.get("scripts", {})

setup(
    name=name,
    version=version,
    description=description,
    long_description=long_description,
    long_description_content_type=long_description_content_type,
    author=author,
    author_email=author_email,
    url=url,
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=install_requires,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=requires_python,
    scripts=scripts,
)