from pathlib import Path
from enum import IntEnum
import platform
import os
import sys
import urllib.request
import tarfile
from zipfile import ZipFile

__all__ = ["check_setup"]


class ArchiveType(IntEnum):
    ZIP = 1
    GZ = 2


INSTALL_STRING = "SCAII is not detected in the canonical location (<home>/.scaii), you may " \
                 "install it yourself from https://github.com/SCAII/Sky-install or we can " \
                 "install the latest release for you. May we install files to your home " \
                 "directory? [y/n]: "

REMOVE_STRING = "SCAII appears to be present but corrupted, may we delete <home>/.scaii " \
                "and attempt to re-extract and install it from the latest release? [y/n]:"

INPUT_ERROR = "Enter 'y' or 'n': "


def check_setup():
    scaii_root = Path.home() / ".scaii"
    if not scaii_root.exists():
        install_scaii()
    else:
        bin_dir = scaii_root / "bin"
        if not bin_dir.exists():
            reinstall_fallback()

        backends_dir = scaii_root / "backends"
        if not backends_dir.exists():
            reinstall_fallback()

        glue_dir = scaii_root / "glue/python"
        if not glue_dir.exists():
            reinstall_fallback()

    check_amend_path()


def reinstall_fallback() -> None:
    print("The .scaii directory appears to be present but corrupted, "
          "proceeding to reinstallation fallback")

    reinstall_permission = get_permission(REMOVE_STRING)

    install_scaii(reinstall_permission)


def get_permission(msg: str) -> bool:
    if 'TRAVIS' in os.environ:
        return True # no input in Travis
    else:
        try:
            permission = input(msg)

            while permission.lower() not in ['y', 'n', 'yes', 'no']:
                permission = input(INPUT_ERROR)

            return permission.lower() in ['y', 'yes']
        # We're probably on a cluster or something
        except OSError:
            return True
        except IOError:
            return True


def install_scaii(pre_permission: bool=False) -> None:
    if platform.system() == 'Windows':
        archive_name = "Windows10_x86_64.scaii.zip"
        archive_type = ArchiveType.ZIP
    elif platform.system() == "Linux":
        archive_name = "Linux-x86_64.scaii.tar.gz"
        archive_type = ArchiveType.GZ
    elif platform.system() == 'Darwin':
        raise Exception("OSX cannot auto-downloaded SCAII, please see https://github.com/SCAII/Sky-install for "
                        "instructions on how to build and install yourself")
    else:
        raise Exception("SCAII is unsupported on " + os.name + " you may "
                        "consider visiting https://github.com/SCAII/Sky-install to "
                        "see if you can build it yourself.")

    if not pre_permission:
        permission = get_permission(INSTALL_STRING)

    if not (pre_permission or permission):
        raise Exception("SCAII not found")

    download_extract(archive_name, archive_type)


def download_extract(archive_name: str, archive_type: ArchiveType) -> None:
    print("Downloading archive for your platform...")
    urllib.request.urlretrieve("https://github.com/SCAII/SCAII/releases/latest/download/" + archive_name,
                       Path.home() / archive_name)

    print()
    print("Extracting...")
    try:
        if archive_type.value == ArchiveType.ZIP.value:
            file = ZipFile(Path.home() / archive_name)
            file.extractall(path=Path.home())
            file.close()
        elif archive_type.value == ArchiveType.GZ.value:
            file = tarfile.open(Path.home() / archive_name, mode='r:gz')
            file.extractall(Path.home())
            file.close()
    finally:
        os.remove(str(Path.home() / archive_name))

    print("Done")


def check_amend_path() -> None:
    # ignore the errors while editing, they're intended
    try:
        from scaii.env.sky_rts.env.scenarios.city_attack import CityAttack
        from scaii.env.explanation import Explanation, BarChart, BarGroup, Bar
    except ImportError:
        glue_path = Path.home() / ".scaii" / "glue" / "python"
        sys.path.append(str(glue_path))

    try:
        from scaii.env.sky_rts.env.scenarios.city_attack import CityAttack
        from scaii.env.explanation import Explanation, BarChart, BarGroup, Bar
    except ImportError:
        raise Exception("Cannot import SCAII by setting sys.path, aborting")

