#!/bin/bash
gphoto2 --capture-image-and-download --force-overwrite
exif capt0000.jpg > exif.txt
