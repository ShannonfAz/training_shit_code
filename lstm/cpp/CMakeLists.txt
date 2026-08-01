cmake_minimum_required(VERSION 3.10)
project(lstm_infer CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(lstm_infer
    main.cpp
    lstmrun.cpp
)
