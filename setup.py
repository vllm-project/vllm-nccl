# this is actually a download and install script
# it appears in `pip` style `setup.py` file, to be easily installable with `pip install`
# and files can be removed with `pip uninstall`
# its argument is derived from environment variable `VLLM_INSTALL_NCCL`,
# e.g. `VLLM_INSTALL_NCCL=2.18+cu11 pip install vllm_nccl`
# after installation, files are available in `{sys.prefix}/vllm_nccl` directory

from setuptools import setup, find_packages
import platform
import os
from dataclasses import dataclass

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

# # test code to check if the files are available
# for each in available_dist_info:
#     url_temp = "https://developer.download.nvidia.com/compute/redist/nccl/v{}/{}".format(
#         each.public_version, each.filename_linux)
#     for architecture in ["x86_64", "aarch64", "ppc64le"]:
#         url = each.get_url(architecture)
#         # print(url)
#         import requests
#         # don't download the file, just check if it exists
#         response = requests.head(url)
#         if response.status_code == 200:
#             pass
#             # print(f"nccl package found at {url}")
#         else:
#             print(f"nccl package not found at {url}")

# download and install nccl package specific to vLLM

architecture = platform.machine()
assert architecture in ["x86_64", "aarch64", "ppc64le"], f"Unsupported architecture: {architecture}"

assert "VLLM_INSTALL_NCCL" in os.environ, "Environment variable VLLM_INSTALL_NCCL is not set"

nccl_major_version, cuda_major_version = os.environ["VLLM_INSTALL_NCCL"].split("+")
cuda_major_version = cuda_major_version[2:] # remove "cu" prefix

assert nccl_major_version in ["2.20", "2.18", "2.17", "2.16"], f"Unsupported nccl major version: {nccl_major_version}"

assert cuda_major_version in ["11", "12"], f"Unsupported cuda major version: {cuda_major_version}"

url = None

for each in available_dist_info:
    if each.cuda_version.split(".")[0] == cuda_major_version and each.full_version.startswith(nccl_major_version):
        url = each.get_url(architecture)
        break

assert url is not None, f"Could not find a suitable nccl package for cuda {cuda_major_version} and nccl {nccl_major_version}"

print(f"Downloading nccl package from {url}")

# download from url
if not os.path.exists("nccl"):
    import requests
    import os
    import shutil
    import subprocess

    # download with progress bar
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open("nccl.txz", "wb") as f:
        shutil.copyfileobj(response.raw, f)

    # extract the file to a temporary location, using python's built-in tarfile module
    import tarfile
    with tarfile.open("nccl.txz") as f:
        f.extractall("nccl")

# list all the files in the extracted directory
files = []
for root, dirs, filenames in os.walk("nccl"):
    for filename in filenames:
        files.append(os.path.relpath(os.path.join(root, filename), "."))

setup(
    name='vllm_nccl',
    version='0.1',
    packages=find_packages(),
    # this directory lies in either `sys.prefix` or `site.USER_BASE`
    # according to https://setuptools.pypa.io/en/latest/deprecated/distutils/setupscript.html#installing-additional-files
    # usually it is under `sys.prefix`
    data_files=[('vllm_nccl', files)],
)
