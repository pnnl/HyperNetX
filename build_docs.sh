#!/bin/bash

rm -rf docs/build
rm -rf docs/source/classes
rm -rf docs/source/algorithms 
rm -rf docs/source/drawing 
rm -rf docs/source/reports

sphinx-apidoc -o docs/source/classes hypernetx/classes
sphinx-apidoc -o docs/source/algorithms hypernetx/algorithms
sphinx-apidoc -o docs/source/drawing hypernetx/drawing
sphinx-apidoc -o docs/source/reports hypernetx/reports
sphinx-build -b html docs/source docs/build