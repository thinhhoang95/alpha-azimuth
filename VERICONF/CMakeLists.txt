# CMakeLists.txt
cmake_minimum_required(VERSION 3.4...3.18)
project(conflict_checker)

find_package(pybind11 REQUIRED)
find_package(OpenMP)

pybind11_add_module(conflict_checker conflict_checker.cpp)

if(OpenMP_CXX_FOUND)
    target_link_libraries(conflict_checker PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(conflict_checker PRIVATE _OPENMP)
endif()

# Set C++ standard
set_target_properties(conflict_checker PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)
