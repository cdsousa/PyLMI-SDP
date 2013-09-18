#!/bin/sh
nosetests --with-doctest --doctest-extension=md --with-coverage --cover-package=. --cover-inclusive
