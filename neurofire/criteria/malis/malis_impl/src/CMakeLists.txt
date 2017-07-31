# build the pybind module
addPythonModule(
    NESTED_NAME
        malis_impl
    SOURCES
        malis.cxx
        malis_impl.cxx
)

# copy the __init__.py
add_custom_command(
    TARGET _malis_impl
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
    ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py
    ${CMAKE_BINARY_DIR}/__init__.py
)
