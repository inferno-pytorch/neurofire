#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <nifty/python/converter.hxx>

#include "malis_impl.hxx"

namespace py = pybind11;

namespace malis_impl {

    using namespace py;

    template<unsigned DIM, class DATA_TYPE, class LABEL_TYPE>
    void exportMalisLossT(py::module & malisModule) {

        malisModule.def("malis_impl", []
            (
                nifty::marray::PyView<DATA_TYPE, DIM+1> affinities,
                nifty::marray::PyView<LABEL_TYPE, DIM> groundtruth,
                const bool pos
            ) {

                // shape of affinities
                std::vector<int64_t> shape(DIM+1);
                for(size_t d = 0; d < DIM+1; ++d) {
                    shape[d] = affinities.shape(d);
                }

                // return data
                nifty::marray::PyView<DATA_TYPE, DIM+1> gradients(shape.begin(), shape.end());
                DATA_TYPE loss, classficationError, randIndex;

                // call c++ function
                {
                    py::gil_scoped_release allowThreads;
                    compute_malis_gradient<DIM>(
                        affinities,
                        groundtruth,
                        pos,
                        gradients,
                        loss,
                        classficationError,
                        randIndex
                    );
                }

                // FIXME this needs to copy the data, so it is not the most efficient thing to do...
                return std::make_tuple(gradients, loss, classficationError, randIndex);
            },
            py::arg("affinities"), py::arg("groundtruth"), py::arg("pos")
        );
    }


    void exportMalisLoss(py::module & malisModule) {

        exportMalisLossT<2, float, uint32_t>(malisModule);
        exportMalisLossT<3, float, uint32_t>(malisModule);

    };
}
