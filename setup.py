from setuptools import find_packages, setup

setup(
    name="logparser",
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
    entry_points={
        "console_scripts": [
            "logparser-parse=logparser.run_parsing:main",
            "logparser-type=logparser.run_typing:main",
            "logparser-event=logparser.run_event_matching:main",
            "logparser-map=logparser.run_mapping:main",
            "logparser-create=logparser.run_creation:main",
            "logparser-norm=logparser.run_norm:main",
            "logparser-fill=logparser.run_fill:main",
            "logparser-ocsf-parse=logparser.run_ocsf_type_parsing:main",
            "logparser-convert=logparser.convert:main",
            "logparser-viz=logparser.curation.app:main",
            "logparser-run-loghub=logparser.evaluation.parse_loghub:main",
            "logparser-compare=logparser.evaluation.compute_metrics:main",
            "logparser-ingest=logparser.evaluation.ingest_relwork:main",
            "logparser-reparse=logparser.run_parse:main",
            "logparser-eval=logparser.evaluation.full_eval:main",
        ]
    },
    zip_safe=False,
)
