#!/bin/bash

argc=$#
src_dir=()

if (( $argc > 0 ))
then
    if [ "$1" = "-all" ]
    then
        src_dir=$(ls src/)
    elif [ "$1" = "-list" ]
    then
        src_dir=$(ls src/)
        for dir in $src_dir
        do
            echo " - $dir"
        done
        exit 0
    else
        for arg in $@
        do
            if ! [ -d src/$arg ]
            then
                echo "Wrong name"
                exit 1
            fi
            src_dir+=("$arg")
        done
    fi

    if ! [ -d "bin/" ]
    then
        mkdir bin
    fi

    for dir in $src_dir
    do
        g++ src/$dir/*.cpp -I . -std=c++23 -o bin/$dir.o
    done
else
    src_dir=$(ls src/)
    echo "! Please, type name of project to build or one of avaliable options"

    echo "  Options:"
    echo "  - build.sh all"
    echo "  - build.sh list"
    echo "  - build.sh <project_name>"

    echo "  Projects:"
    for dir in $src_dir
    do
        echo "  - $dir"
    done
fi
    

