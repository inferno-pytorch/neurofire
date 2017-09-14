#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <nifty/python/converter.hxx>

#include "malis_impl.hxx"

namespace py = pybind11;

namespace malis_impl {

    using namespace py;

    template <class ARRAY>
    void inline initPyView(ARRAY & data) {

        // FIXME this would be the easiest and fastest, but for some reason won't compile
        // although operator= is overloaded for marray::View
        //data = static_cast<typename ARRAY::value_type>(0.);

        // iterators for multiarray are slow :(
        // get the pointer to 0 and fill up to data.size() with zeros - should be much faster
        std::fill(data.begin(), data.end(), 0.);
    }

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

                // standard malis
                std::vector<int> ranges(DIM, -1);
                std::vector<int> axes(DIM);
                std::iota(axes.begin(), axes.end(), 0);
                initPyView(gradients);
                compute_malis_gradient<DIM>(
                    affinities,
                    groundtruth,
                    ranges,
                    axes,
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

    template<unsigned DIM, class DATA_TYPE, class LABEL_TYPE>
    void exportConstrainedMalisLossT(py::module & malisModule) {

        malisModule.def("constrained_malis_impl", []
            (
                nifty::marray::PyView<DATA_TYPE, DIM+1> affinities,
                nifty::marray::PyView<LABEL_TYPE, DIM> groundtruth
            ) {

                // shape of affinities
                std::vector<int64_t> shape(DIM+1);
                for(size_t d = 0; d < DIM+1; ++d) {
                    shape[d] = affinities.shape(d);
                }

                // return data
                nifty::marray::PyView<DATA_TYPE, DIM+1> gradients(shape.begin(), shape.end());
                //nifty::marray::Marray<DATA_TYPE> gradients(shape.begin(), shape.end());
                DATA_TYPE loss = 0;

                // call c++ function
                {
                    py::gil_scoped_release allowThreads;

                    initPyView(gradients);
                    // standard malis
                    std::vector<int> ranges(DIM, -1);
                    std::vector<int> axes(DIM);
                    std::iota(axes.begin(), axes.end(), 0);
                    compute_constrained_malis_gradient<DIM>(
                        affinities,
                        groundtruth,
                        ranges,
                        axes,
                        gradients,
                        loss
                    );
                }

                // we don't return the loss for now, as it can't be used anyway
                return gradients;
            },
            py::arg("affinities"), py::arg("groundtruth")
        );

        malisModule.def("constrained_malis_custom_nh_impl", []
            (
                nifty::marray::PyView<DATA_TYPE, DIM+1> affinities,
                nifty::marray::PyView<LABEL_TYPE, DIM> groundtruth,
                std::vector<int> ranges,
                std::vector<int> axes
            ) {

                // shape of affinities
                std::vector<int64_t> shape(DIM+1);
                for(size_t d = 0; d < DIM+1; ++d) {
                    shape[d] = affinities.shape(d);
                }

                // return data
                nifty::marray::PyView<DATA_TYPE, DIM+1> gradients(shape.begin(), shape.end());
                //nifty::marray::Marray<DATA_TYPE> gradients(shape.begin(), shape.end());
                DATA_TYPE loss = 0;

                // call c++ function
                {
                    py::gil_scoped_release allowThreads;

                    initPyView(gradients);
                    compute_constrained_malis_gradient<DIM>(
                        affinities,
                        groundtruth,
                        ranges,
                        axes,
                        gradients,
                        loss
                    );
                }

                // we don't return the loss for now, as it can't be used anyway
                return gradients;
            },
            py::arg("affinities"), py::arg("groundtruth"), py::arg("ranges"), py::arg("axes")
        );
    }


    void exportMalisLoss(py::module & malisModule) {

        // export malis loss
        exportMalisLossT<2, float, uint32_t>(malisModule);
        exportMalisLossT<3, float, uint32_t>(malisModule);
        exportMalisLossT<2, float, int64_t>(malisModule);
        exportMalisLossT<3, float, int64_t>(malisModule);

        // export constrained malis loss
        exportConstrainedMalisLossT<2, float, uint32_t>(malisModule);
        exportConstrainedMalisLossT<3, float, uint32_t>(malisModule);
        exportConstrainedMalisLossT<2, float, int64_t>(malisModule);
        exportConstrainedMalisLossT<3, float, int64_t>(malisModule);

    };
}
