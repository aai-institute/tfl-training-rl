from setuptools import find_packages, setup


def read_requirements(filename: str):
    return [line for line in open(filename).readlines() if not line.startswith("--")]


test_requirements = read_requirements("requirements-test.txt")


setup(
    name="training_rl",
    python_requires=">=3.9",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    include_package_data=True,
    version="1.0.0",
    description="TransferLab training",
    install_requires=read_requirements("requirements.txt"),
    extras_require={"test": test_requirements},
    author="appliedAI TransferLab",
)
