#!/bin/sh
nosetests --with-doctest --doctest-extension=md --with-coverage --cover-package=lmi_sdp --cover-inclusive --cover-tests $@
