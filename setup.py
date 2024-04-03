# this is actually a download and install script
# it appears in `pip` style `setup.py` file, to be easily installable with `pip install`
# and files can be removed with `pip uninstall`
# unfortunately, this brings nccl into the wheel, which will exceed the 100MB limit of PyPI.
# so we don't upload this to PyPI, but instead, we can install it from github
# its argument is derived from environment variable `VLLM_INSTALL_NCCL`,
# e.g. `VLLM_INSTALL_NCCL=2.18+cu12 pip install https://github.com/vllm-project/vllm-nccl.git`
# after installation, files are available in `{sys.prefix}/vllm_nccl` directory

from setuptools import setup, find_packages
import platform
import os
from dataclasses import dataclass

# for reference, we can download nccl from the following links

@dataclass
class DistInfo:
    cuda_version: str
    full_version: str
    public_version: str
    filename_linux: str

    def get_url(self, architecture: str) -> str:
        url_temp = "https://developer.download.nvidia.com/compute/redist/nccl/v{}/{}".format(
            self.public_version, self.filename_linux)
        return url_temp.replace("x86_64", architecture)

# taken from https://developer.download.nvidia.com/compute/redist/nccl/
available_dist_info = [
    # nccl 2.16.5
    DistInfo('11.8', '2.16.5', '2.16.5', 'nccl_2.16.5-1+cuda11.8_x86_64.txz'),
    DistInfo('12.0', '2.16.5', '2.16.5', 'nccl_2.16.5-1+cuda12.0_x86_64.txz'),
    # nccl 2.17.1
    DistInfo('11.0', '2.17.1', '2.17.1', 'nccl_2.17.1-1+cuda11.0_x86_64.txz'),
    DistInfo('12.0', '2.17.1', '2.17.1', 'nccl_2.17.1-1+cuda12.0_x86_64.txz'),
    # nccl 2.18.1
    DistInfo('11.0', '2.18.1', '2.18.1', 'nccl_2.18.1-1+cuda11.0_x86_64.txz'),
    DistInfo('12.0', '2.18.1', '2.18.1', 'nccl_2.18.1-1+cuda12.0_x86_64.txz'),
    # nccl 2.20.3
    DistInfo('11.0', '2.20.3', '2.20.3', 'nccl_2.20.3-1+cuda11.0_x86_64.txz'),
    DistInfo('12.2', '2.20.3', '2.20.3', 'nccl_2.20.3-1+cuda12.2_x86_64.txz'),
]

package_name = "vllm_nccl_cu12"
cuda_name = package_name[-4:]
nccl_version = "2.18.1"
vllm_nccl_verion = "0.1.0"
version = ".".join([nccl_version, vllm_nccl_verion])

assert nccl_version == "2.18.1", f"only support nccl 2.18.1, got {version}"

url = f"https://storage.googleapis.com/vllm-public-assets/nccl/{cuda_name}/libnccl.so.{nccl_version}"

import urllib.request
import os

# desination path is ~/.config/vllm/nccl/cu12/libnccl.so.2.18.1
destination = os.path.expanduser(f"~/.config/vllm/nccl/cu12/libnccl.so.{nccl_version}")

os.makedirs(os.path.dirname(destination), exist_ok=True)

if os.path.exists(destination):
    print(f"nccl package already exists at {destination}")
else:
    print(f"Downloading nccl package from {url}")
    import urllib.request
    urllib.request.urlretrieve(url, destination)
    print(f"nccl package downloaded to {destination}")

setup(
    name=package_name,
    version=version,
    packages=["vllm_nccl"],
)
