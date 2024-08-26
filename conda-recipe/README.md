# conda recipe README

## How to build

```sh
./build_conda 3.11 2.2.1 cu121  #Python verion, torch version, CUDA version
```

## Tips

- Do not rename `build_conda.sh` to `build.sh`. As the `build.sh` will be recognized as part of conda build instruction file.
