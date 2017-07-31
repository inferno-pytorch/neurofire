#include <pybind11/pybind11.h>

namespace py = pybind11;


namespace malis_impl {
    void exportMalisLoss(py::module &);
}


PYBIND11_PLUGIN(_malis) {
    
    py::module malisModule("_malis", "C++ implementation of malis loss");

    using namespace malis_impl;

    exportMalisLoss(malisModule);

    return malisModule.ptr();
}
