#!/usr/bin/env bash

if INVALID_COLUMNS=$(xsv headers -j data/pivoted.csv | grep -v -E "^[0-9a-zA-Z_\-]+$")
then
    printf 'Column names may only include alphanumeric characters, dashes, or underscores.\n\n'
    printf 'Invalid column names:\n\n%s\n' "$INVALID_COLUMNS"
    exit 1
else
    cp data/subsampled.csv data/validated.csv
fi
