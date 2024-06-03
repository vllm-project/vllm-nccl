**NOTE: This repo is deprecated with [this fix](https://github.com/vllm-project/vllm/pull/5091) to the main vLLM repo.**

# vllm-nccl

Manages vllm-nccl dependency

1. Define `package_name`, `nccl_version`, `vllm_nccl_verion`
2. run `python setup.py sdist`
3. run `twine upload dist/*`
