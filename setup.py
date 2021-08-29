"""
    Setup file for sliceoptim.
    Use setup.cfg to configure your project.

    This file was generated with PyScaffold 4.0.2.
    PyScaffold helps you to put up the scaffold of your new Python project.
    Learn more under: https://pyscaffold.org/
"""
from setuptools import setup
from platform import machine

if __name__ == "__main__":
    try:
        if not machine() in ["armv7l", "armv6l"]:
            print("\n\nNo ARM platform detected.\n\n")
            setup(
                use_scm_version={"version_scheme": "no-guess-dev"},
            )
        else:
            print(
                "\n\nARM platform detected! Using pywheels for compiled dependencies.\n\n"
            )
            setup(
                use_scm_version={"version_scheme": "no-guess-dev"},
                dependency_links=[
                    "https://www.piwheels.org/simple/pandas",
                    "https://www.piwheels.org/simple/numpy",
                    "https://www.piwheels.org/simple/scipy",
                    "https://www.piwheels.org/simple/scikit-learn",
                    "https://www.piwheels.org/simple/scikit-optimize",
                ],
            )
    except:  # noqa
        print(
            "\n\nAn error occurred while building the project, "
            "please ensure you have the most updated version of setuptools, "
            "setuptools_scm and wheel with:\n"
            "   pip install -U setuptools setuptools_scm wheel\n\n"
        )
        raise
