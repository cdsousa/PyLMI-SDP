#!/bin/sh
pytest --doctest-glob="*.md" --cov=lmi_sdp "$@"
