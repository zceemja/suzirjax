# Suzirjax [_suzïr'yax_]
UCL Optical Networks Group (ONG) Real-Time Constellation Shaper

### Reason for naming
It's a word "constellation" in Ukrainian (сузір'я) with "jax" suffix because 
python side computation is implemented in [JAX](https://github.com/google/jax).

### Project structure
* _suzirjax_ - python part with GUI (in pyqt5), most computation, simulations and equipment interfaces 
* _resources_ - arbitrary data binary files and images

### How to run?

Preparation
```shell
python -m venv env
pip install -r requirements.txt

# for CUDA acceleration
pip install "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

Run GUI
```shell
python -m suzirjax
```

Run Animation generator
```shell
python -m suzirjax.animation
```

