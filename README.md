# PyAngstrom (new and improved!)

Try the fastest way of performing the Angstrom method, yet!

PyAngstrom is a
fully-featured Python library with which you can take thermal camera videos and
extrapolate the thermal conductivity of the material that was shot. It comes
with several solutions to common governing equations, and you also get many
options for how you want to fit your models to your experimental data. If these
built-in functions do not measure up to your needs, you can also easily extend
PyAngstrom by writing custom solutions and fitting methods to match exactly what
you are looking for.

Let's make the Angstrom method great again!

## Installation

This repository can be directly installed through pip using the command:
```
pip install git+https://github.com/RuralBrick/Angstrom-Method-Rewrite.git
```
The `pyangstrom` package will be installed into your current environment.

## Usage

Instructions, examples, and much more can all be found in this repository’s
[wiki](https://github.com/RuralBrick/Angstrom-Method-Rewrite/wiki)! Also, source
code documentation can be found in this repository’s
[Pages](https://ruralbrick.github.io/Angstrom-Method-Rewrite/pyangstrom/index.html),
for those interested in all the nitty-gritty details.

## Citation + Previous Work

This repository is a rewrite of the code found at
https://github.com/yuanyuansjtu/High-T-Angstrom-Method and
https://github.com/yuanyuansjtu/Angstrom-method. If you use this code in your
research, please cite the following sources:
- https://doi.org/10.5281/zenodo.4587868
- https://doi.org/10.5281/zenodo.4587863
- https://doi.org/10.1115/1.4053108
- https://doi.org/10.1115/1.4047145

## Other Credits

This code was written for the [Nano Transport Research Group (NTRG) at
UCLA](https://ntrg.seas.ucla.edu/) through a collaboration between PhD Student
Min Jong Kil and Undergraduate Research Assistant Theodore Lau under the
direction of Professor Timothy S. Fisher. The majority of this work is inspired
by and derived from the work of Yuan Hu under NTRG.

## Works Referenced

This code is based off of techniques and theory described in the following
papers:
- Hu, Y., Abuseada, M., Alghfeli, A., Holdheim, S., and Fisher, T. S. (December
  20, 2021). "High-Temperature Thermal Diffusivity Measurements Using a Modified
  Ångström's Method With Transient Infrared Thermography." ASME. J. Heat
  Transfer. February 2022; 144(2): 023502. https://doi.org/10.1115/1.4053108
- Hu, Y., and Fisher, T. S. (May 29, 2020). "Accurate Thermal Diffusivity
  Measurements Using a Modified Ångström's Method With Bayesian Statistics."
  ASME. J. Heat Transfer. July 2020; 142(7): 071401.
  https://doi.org/10.1115/1.4047145
- Lopez-Baeza, Ernesto & Rubia, J & Goldsmid, Hiroshi. (1987). Angstrom's
  thermal diffusivity method for short samples. Journal of Physics D Applied
  Physics. 20. 1156. 10.1088/0022-3727/20/9/011.
