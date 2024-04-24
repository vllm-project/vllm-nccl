# this is actually a download and install script
# it appears in `pip` style `setup.py` file, to be easily installable with `pip install`

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

import hashlib

def get_md5_hash(file_path):
    hash_md5 = hashlib.md5()  # Create MD5 hash object
    with open(file_path, "rb") as f:  # Open file in binary read mode
        for chunk in iter(lambda: f.read(4096), b""):  # Read file in 4KB chunks
            hash_md5.update(chunk)  # Update the hash with the chunk
    return hash_md5.hexdigest()  # Return the final hash as a hexadecimal string

package_name = "vllm_nccl_cu11"
cuda_name = package_name[-4:]
nccl_version = "2.18.1"
vllm_nccl_verion = "0.4.0"
version = ".".join([nccl_version, vllm_nccl_verion])

file_hash = {
    "cu11": "5129e4e7e671cc7ce072aaeea870bee8",
    "cu12": "296c4de7fbdb0f7fd8501fb63bd0cb40",
}[cuda_name]

assert nccl_version == "2.18.1", f"only support nccl 2.18.1, got {version}"

url = f"https://github.com/vllm-project/vllm-nccl/releases/download/v0.1.0/{cuda_name}-libnccl.so.{nccl_version}"

import urllib.request
import os

# desination path is ~/.config/vllm/nccl/cu12/libnccl.so.2.18.1
destination = os.path.expanduser(f"~/.config/vllm/nccl/{cuda_name}/libnccl.so.{nccl_version}")

os.makedirs(os.path.dirname(destination), exist_ok=True)

while True:
    if os.path.exists(destination):
        print(f"nccl package already exists at {destination}")
    else:
        print(f"Downloading nccl package from {url}")
        try:
            import urllib.request
            urllib.request.urlretrieve(url, destination)
            print(f"nccl package downloaded to {destination}")
        except Exception as e:
            print(f"Failed to download nccl package from {url}")
            print(e)
    if get_md5_hash(destination) != file_hash:
        print(f"md5 hash of downloaded file does not match expected hash, retrying")
        os.remove(destination)
    else:
        print(f"md5 hash of downloaded file matches expected hash")
        break

os.chmod(destination, 0o777)

setup(
    name=package_name,
    version=version,
    packages=["vllm_nccl"],
)
