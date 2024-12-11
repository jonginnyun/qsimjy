cmake_minimum_required(VERSION 3.15)
project(qsimjy_c_binding LANGUAGES CXX)

# Locate pybind11
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 REQUIRED)

# Set the target for pybind11 module
pybind11_add_module(_cxx_magcalc c_magcalc_pybind.cpp)
pybind11_add_module(_cxx_potcalc c_potcalc_pybind.cpp)

# Compiler options
if(MSVC)
    target_compile_options(_cxx_magcalc PRIVATE /W4 /std:c++17)
    target_compile_options(_cxx_potcalc PRIVATE /W4 /std:c++17)
else()
    target_compile_options(_cxx_magcalc PRIVATE -Wall -Wextra -std=c++17)
    target_compile_options(_cxx_potcalc PRIVATE -Wall -Wextra -std=c++17)
endif()
