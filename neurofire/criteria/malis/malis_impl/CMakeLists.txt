cmake_minimum_required(VERSION 3.1)

project(malis_impl)


####################
# set c++ 11 support
####################

set(CMAKE_CXX_FLAG ${CMAKE_CXX_FLAG} "-std=c++11")

#-------------------------------------------------------------------------------------------------------------------
# Set a default build configuraton if none is specified. 'Release' produces the fastest code
#-------------------------------------------------------------------------------------------------------------------
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to 'Release' as none was specified.")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
        "MinSizeRel" "RelWithDebInfo")
endif()
string(TOUPPER "${CMAKE_BUILD_TYPE}" U_CMAKE_BUILD_TYPE)


#-------------------------------------------------------------------------------------------------------------------
# thread libs
#-------------------------------------------------------------------------------------------------------------------
find_package(Threads)


#-------------------------------------------------------------------------------------------------------------------
# rt lib
#-------------------------------------------------------------------------------------------------------------------
if(UNIX AND NOT APPLE)
  find_library(RT rt)
  set(LINK_RT true)
else()
  set(LINK_RT false)
endif()


##################
# nifty and pybind
##################

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/nifty/include")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/nifty/externals/pybind11/include")

add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/nifty/externals/pybind11")


###############
# include dirs
###############

file(GLOB_RECURSE headers include/*.hxx)
include_directories(include)


###########################################################################
# macro to simplyfy the adding of a submodule
###########################################################################
macro(addPythonModule)

    set(options "")
    set(oneValueArgs NESTED_NAME)
    set(multiValueArgs SOURCES LIBRRARIES)
    cmake_parse_arguments(ADD_PY_MOD "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN} )

    #message(STATUS "BRAAA"  ${ADD_PY_MOD_NESTED_NAME})

    # get name of the module
    string(REPLACE "/" ";" MOD_NESTING_LIST "${ADD_PY_MOD_NESTED_NAME}")
    list(LENGTH MOD_NESTING_LIST nesting_len)
    LIST(GET MOD_NESTING_LIST ${LENGTH}-1 MODULE_NAME)


    #Create the binding library
    pybind11_add_module(_${MODULE_NAME}
        ${ADD_PY_MOD_SOURCES}
    )
    #if (APPLE AND NOT ${CMAKE_BUILD_TYPE} MATCHES Debug)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
      target_compile_options(_${MODULE_NAME} PRIVATE "-fvisibility=default")
    endif()

    #endif()
    set_target_properties(_${MODULE_NAME}  PROPERTIES PREFIX "")

    #message(STATUS ********Module: ${MODULE_NAME})
    #message(STATUS ********Libs: ${ADD_PY_MOD_LIBRRARIES})



    # link additional libraries
    target_link_libraries(_${MODULE_NAME}
        PUBLIC
        ${PYTHON_LIBRARIES}
        ${ADD_PY_MOD_LIBRRARIES}
    )

    # create module dir
    #file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/python/${ADD_PY_MOD_NESTED_NAME})

    # copy the module file
    ADD_CUSTOM_COMMAND(
        TARGET _${MODULE_NAME}
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} ARGS -E copy_if_different $<TARGET_FILE:_${MODULE_NAME}>
        ${CMAKE_BINARY_DIR}
        COMMENT "Copying target ${MODULE_NAME} to temporary module directory")

    # add_dependencies(malis_impl _${MODULE_NAME})
endmacro()



############
# src folder
############

add_subdirectory(src)

