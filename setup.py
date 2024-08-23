from setuptools import setup, find_packages

setup(
    name="autodistill-rt-detr",
    version="0.1.0",
    description="A package for RT-DETR integration with autodistill",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/venkatram-dev/autodistill-rt-detr",  # Replace with your actual repo URL
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchvision",
        "transformers",
        "pytorch_lightning",
        "supervision",
        "opencv-python",
        "autodistill",
        "roboflow",
        "supervision",
        "pycocotools",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)

