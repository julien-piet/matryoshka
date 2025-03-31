import os

from setuptools import find_packages, setup


# Read requirements from requirements.txt
def read_requirements():
    requirements_path = os.path.join(
        os.path.dirname(__file__), "requirements.txt"
    )
    if os.path.exists(requirements_path):
        with open(requirements_path, "r", encoding="utf-8") as f:
            requirements = []
            for line in f:
                line = line.strip()
                # Skip empty lines and comments
                if line and not line.startswith("#"):
                    requirements.append(line)
            return requirements
    return []


setup(
    name="matryoshka",
    version="0.1",
    description="LLM-based log parser",
    packages=find_packages("src", exclude=["tests*"]),
    classifiers=[
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: Apache Software License",
    ],
    python_requires="~=3.9",
    include_package_data=True,
    package_dir={"": "src"},
    install_requires=read_requirements(),
    entry_points={
        "console_scripts": [
            "matryoshka-syntax=matryoshka.entrypoints.syntax:main",
            "matryoshka-schema=matryoshka.entrypoints.schema:main",
            "matryoshka-map=matryoshka.entrypoints.map:main",
            "matryoshka=matryoshka.entrypoints.all:main",
            "matryoshka-edit=matryoshka.curation.app:main",
            "matryoshka-eval=matryoshka.evaluation.eval_full:main",
            "matryoshka-ingest-loghub-baseline=matryoshka.evaluation.compare_to_loghub:main",
            "matryoshka-ingest-loghub-relwork=matryoshka.evaluation.ingest_relworks:main",
            "matryoshka-eval-syntax=matryoshka.evaluation.eval_syntax:main",
            "matryoshka-ingest=matryoshka.entrypoints.ingest:main",
            "matryoshka-validate-syntax=matryoshka.entrypoints.validate_syntax:main",
            "matryoshka-save-baseline=matryoshka.evaluation.save_baseline_as_json:main",
            "matryoshka-convert-lilac=matryoshka.evaluation.convert_lilac_to_tree:main",
        ]
    },
    zip_safe=False,
)
