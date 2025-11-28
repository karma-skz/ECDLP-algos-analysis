/* 
 * File:   MPI_UTILS.hpp
 * Author: abdullah
 *
 * Created on 7 December, 2020, 17:28 PM
 */

#ifndef MPI_UTILS_HPP
#define MPI_UTILS_HPP

#include <mpi.h>

void MPI_get_MPI_rank_size_name(int &processorId, int &totalNumberOfProcessors, char *nodeName);


#endif /* MPI_UTILS_HPP */