from setuptools import setup, find_packages

setup(
    name="mcp-client-tx",
    version="0.1.0",
    description="中文MCP客户端",
    author="Xin Tang",
    author_email="xtang@example.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "httpx>=0.23.0",
        "pydantic>=1.9.0",
    ],
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)