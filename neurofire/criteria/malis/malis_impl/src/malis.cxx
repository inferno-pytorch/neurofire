#include <pybind11/pybind11.h>

namespace py = pybind11;


namespace malis_impl {
    void exportMalisLoss(py::module &);
}


PYBIND11_PLUGIN(_malis_impl) {

    py::module malisModule("_malis_impl", "C++ implementation of malis loss");

    using namespace malis_impl;

    exportMalisLoss(malisModule);

    return malisModule.ptr();
}
