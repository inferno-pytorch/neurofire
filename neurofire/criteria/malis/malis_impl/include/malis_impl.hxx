#pragma once

#include <algorithm>
#include <map>

#include "nifty/marray/marray.hxx"
#include "nifty/tools/for_each_coordinate.hxx"
#include "nifty/ufd/ufd.hxx"

namespace malis_impl{

// TODO implement different nhoods
template<unsigned DIM, typename DATA_TYPE, typename LABEL_TYPE>
void compute_malis_gradient(
    const nifty::marray::View<DATA_TYPE> & affinities,
    const nifty::marray::View<LABEL_TYPE> & groundtruth,
    const bool pos,
    nifty::marray::View<DATA_TYPE> & gradientsOut,
    DATA_TYPE & lossOut,
    DATA_TYPE & classificationErrorOut,
    DATA_TYPE & randIndexOut
) {

    typedef nifty::array::StaticArray<int64_t,DIM>   Coord;
    typedef nifty::array::StaticArray<int64_t,DIM+1> AffinityCoord;
    typedef LABEL_TYPE LabelType;
    typedef DATA_TYPE DataType;

    // check that number of affinity channels matches the dimensions
    NIFTY_CHECK_OP(affinities.shape(DIM),==,DIM,"Number of affinity channels does not match the dimension!");
    NIFTY_CHECK_OP(gradientsOut.shape(DIM),==,DIM,"Number of gradient channels must match !");
    // check that shapes match
    for(int d = 0; d < DIM; ++d) {
        NIFTY_CHECK_OP(affinities.shape(d),==,groundtruth.shape(d),"Affinity shape does not match gt shape!");
        NIFTY_CHECK_OP(affinities.shape(d),==,gradientsOut.shape(d),"Affinity shape does not match gradients shape!");
    }

    // 1.) Initialize the union-find and the overlap vector, which stores for every pixel
    // the overlap with ground-truth segments in a map
    // also count the gt segment sizes

    // number of nodes and edges in the affinity graph
    const int numberOfNodes = groundtruth.size(); // the number of node corresponds to the number of pixels / voxels
    const int numberOfEdges = affinities.size();  // the number of edges corresponds to the number of affinities

    Coord pixelShape;
    for(int d = 0; d < DIM; ++d) {
        pixelShape[d] = groundtruth.shape(d);
    }

    // union find, overlaps and segment sizes
    nifty::ufd::Ufd<LabelType> sets(numberOfNodes);
    std::vector<std::map<LabelType, size_t>> overlaps(numberOfNodes);
    std::map<LabelType, size_t> segmentSizes;

    // number of positive pairs and labeled nodes for normalization
    size_t numberOfLabeledNodes = 0, nPairPos = 0;

    size_t nodeIndex = 0;
    nifty::tools::forEachCoordinate(pixelShape, [&](Coord coord) {
        auto gtId = groundtruth(coord.asStdArray());

        if(gtId != 0) {
            overlaps[nodeIndex].insert( std::make_pair(gtId,1) );
            ++segmentSizes[gtId];
            ++numberOfLabeledNodes;
            nPairPos += (segmentSizes[gtId] - 1);
        }

        ++nodeIndex;
    });

    // compute normalizations
    size_t nPairNorm;
    if (pos) {
        nPairNorm = nPairPos;
    }
    else {
        size_t nPairTot = (numberOfLabeledNodes * (numberOfLabeledNodes - 1)) / 2;
        nPairNorm = nPairTot - nPairPos;
    }

    // 2.) Sort all affinity edges in increasing order of weight

    AffinityCoord affinityShape;
    for(int d = 0; d < DIM+1; ++d) {
        affinityShape[d] = affinities.shape(d);
    }

    // get a flattened view to the marray
    size_t flatShape[] = {affinities.size()};
    auto flatView = affinities.reshapedView(flatShape, flatShape+1);

    // initialize the pqueu as [0,1,2,3,...,numberOfEdges]
    std::vector<size_t> pqueue(numberOfEdges);
    std::iota(pqueue.begin(), pqueue.end(), 0);

    // sort pqueue in increasing order
    std::sort(pqueue.begin(), pqueue.end(), [&flatView](const size_t ind1, const size_t ind2){
        return (flatView(ind1)>flatView(ind2));
    });

    // 3.) Run kruskal - for each min spanning tree edge,
    // we compute the loss and gradient

    size_t edgeIndex, channel;
    LabelType setU, setV;
    size_t nPair = 0, nPairIncorrect = 0 ;
    double loss = 0, gradient = 0;
    Coord gtCoordU, gtCoordV;
    AffinityCoord affCoord;
    typename std::map<LabelType,size_t>::iterator itU, itV;
    DATA_TYPE affinity;

    // iterate over the pqueue
    for(auto edgeIndex : pqueue) {

        // translate edge index to coordinate
        affCoord[0] = edgeIndex / affinities.strides(0) ;
        for(int d = 1; d < DIM+1; ++d) {
            affCoord[d] = (edgeIndex % affinities.strides(d-1) ) / affinities.strides(d);
        }

        // first, we copy the spatial coordinates of the affinity pixel for both gt coords
        for(int d = 0; d < DIM; ++d) {
            gtCoordU[d] = affCoord[d];
            gtCoordV[d] = affCoord[d];
        }

        // we increase the V coordinate for the given channel (=corresponding coordinate)
        // only if this results in a valid coordinate
        channel = affCoord[DIM];
        if(gtCoordV[channel] < pixelShape[channel] - 1) {
            ++gtCoordV[affCoord[DIM]];
        }
        else {
            continue;
        }

        setU = sets.find( groundtruth(gtCoordU.asStdArray()) ) ;
        setV = sets.find( groundtruth(gtCoordV.asStdArray()) ) ;

        // only do stuff if the two segments are not merged yet
        if(setU != setV) {
            sets.merge(setU, setV);

            // compute the number of pairs merged by this edge
            for (itU = overlaps[setU].begin(); itU != overlaps[setU].end(); ++itU) {
                for (itV = overlaps[setV].begin(); itV != overlaps[setV].end(); ++itV) {

                    // the number of pairs that are joind by this edge are given by the
                    // number of pix associated with U times pix associated with V
                    nPair = itU->second * itV->second;

                    // for pos, TODO 
                    // we add nPairs if we join two nodes in the same gt segment
                    if (pos && (itU->first == itV->first)) {
                        affinity = affinities(affCoord.asStdArray());
                        gradient = 1. - affinity;
                        loss += gradient * gradient * nPair;
                        gradientsOut(affCoord.asStdArray()) += gradient * nPair;

                        // if the affinity for this edge is smaller than 0.5, although the two nodes are connected in the
                        // groundtruth, this is a classification error
                        if(affinity <= .5) {
                            nPairIncorrect += nPair;
                        }
                    }

                    // for !pos, TODO
                    // we add nPairs if we join two nodes in different gt segments
                    else if (!pos && (itU->first != itV->first)) {
                        affinity = affinities(affCoord.asStdArray());
                        gradient = -affinity;
                        loss += gradient * gradient * nPair;
                        gradientsOut(affCoord.asStdArray()) += gradient * nPair;

                        // if the affinity for this edge is bigger than 0.5, although the two nodes are not connected in the
                        // groundtruth, this is a classification error
                        if(affinity > .5) {
                            nPairIncorrect += nPair;
                        }
                    }
                }
            }

            // normalize the gradients
            if (nPairNorm > 0) {
                gradientsOut(affCoord.asStdArray()) /= nPairNorm;
            }
            else {
                gradientsOut(affCoord.asStdArray()) = 0;
            }

            // move the pixel bags of the non-representative to the representative
            if (sets.find(setU) == setV) // make setU the rep to keep and setV the rep to empty
                std::swap(setU,setV);

            itV = overlaps[setV].begin();
            while (itV != overlaps[setV].end()) {
                itU = overlaps[setU].find(itV->first);
                if (itU == overlaps[setU].end()) {
                    overlaps[setU].insert( std::make_pair(itV->first,itV->second) );
                }
                else {
                    itU->second += itV->second;
                }
                overlaps[setV].erase(itV++);
            }
        }
    }

    // 4.) Outputs

    // return the loss
    if (nPairNorm > 0) {
        loss /= nPairNorm;
    }
    else {
        loss = 0;
    }
    lossOut = loss;

    // return the classification error and rand index
    classificationErrorOut = static_cast<DATA_TYPE>(nPairIncorrect) / static_cast<DATA_TYPE>(nPairNorm);
    randIndexOut = 1. - static_cast<DATA_TYPE>(nPairIncorrect) / static_cast<DATA_TYPE>(nPairNorm);

}

}
