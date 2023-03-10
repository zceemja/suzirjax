# Suzirjax [_suzïr'yax_]
UCL Optical Networks Group (ONG) Real-Time Constellation Shaper

![demo.mp4](resources%2Fdemo.mp4)

## What's here?
Here you can find code for constellation shaping for AWGN/PC-AWGN simulation including pyQT5 gui, 
or headless "animator" to produce lovely mp4 animations.

It also includes "remove channel" that connects to the lab and does experimental transmission.
DSP is mostly handled by [QAMpy](http://qampy.org/) library.

## Reason for naming
It's a word "constellation" in Ukrainian (сузір'я) with "jax" suffix because 
python side computation is implemented in [JAX](https://github.com/google/jax).

## OFC 2023
Find out more in 
[OFC23 Demo video](https://mediacentral.ucl.ac.uk/player?autostart=n&videoId=8ad172H5&captions=y&chapterId=0&playerJs=n)

## Experimental setup
Experimental setup includes single channel transmission over 1550nm with 
real-time updatable channel and launch power parameters. Encoding and DSP is done remotely (server side), 
GMI calculations and optimisation are done locally (gui client) with new constellations 
being sent back to server.

![diagram.png](resources%2Fdiagram.png)


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

