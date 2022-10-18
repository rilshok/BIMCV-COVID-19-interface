#!/usr/bin/env bash

stubgen ./bimcvcovid19i -o .
python setup.py sdist bdist_wheel
twine upload --repository pypi dist/*
find bimcvcovid19i -name "*.pyi" -type f -delete
rm -r dist build
