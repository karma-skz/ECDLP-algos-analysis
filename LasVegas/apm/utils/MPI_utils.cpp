#include "MPI_utils.hpp"

void MPI_get_MPI_rank_size_name(int &processorId, int &totalNumberOfProcessors, char *nodeName){
    int NodeNameLen;
    
    MPI_Comm_rank(MPI_COMM_WORLD, &processorId);
    MPI_Comm_size(MPI_COMM_WORLD, &totalNumberOfProcessors);
    MPI::Get_processor_name(nodeName, NodeNameLen);
}