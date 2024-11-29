# ome-zarr-models-py

A Python package that provides a Python object representation of an OME-zarr dataset.
It aims to be a reference implementation of the OME-zarr specification.

> ⚠️ This is still under construction!
> We welcome feedback, suggestions, and contributions!
> But beware, this package may change without warning for now

## Installing

Currently this package is only available in the GitHub repo.
Install it using:

```sh
pip install git+https://github.com/BioImageTools/ome-zarr-models-py.git@main
```

When we do a first release, it will be available on PyPI and conda-forge.

## Getting help

Developers of this package are active on our [Zulip chat channel](https://imagesc.zulipchat.com/#narrow/channel/469152-ome-zarr-models-py), which is a great place for asking questions and getting help.

## Design

This package is designed with the following guiding principles:

- Strict adherence to the OME-zarr specification
- A usable set of Python classes for reading and writing and interacting with OME-zarr metadata
- A separate object for each version of the OME-zarr specification
- Array reading and writing operations are out of scope

We are trying to make this as usable and useful as possible while still complying with the OME-zarr specification.

## Roadmap

This is our draft roadmap.

### v1

- A working validator for OME-zarr 0.4 datasets

### v2

- Ability to write metadata after creation/modification
- A working validator for OME-zarr 0.5 datasets
