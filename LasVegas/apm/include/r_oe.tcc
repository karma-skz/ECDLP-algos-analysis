
template <class _mat_, class FiniteField>
void r_oe(_mat_ mat)
{
    ulong matRows = mat.NumRows() - 1;
    for (ulong i = 0; i < matRows; ++i)
    {
        for (int j = i; j >= 0; --j)
        {
            FiniteField e = mat[j][i - j] / mat[j + 1][i - j];
            mat[j] = mat[j] - (e * mat[j + 1]);
        }
    }
}

template <class _mat_, class FiniteField>
bool reverse_obliqueElimination(ZZ ordP)
{
    int processorId, totalNumberOfProcessors;
    char *NodeName = new char[1000];
    MPI_get_MPI_rank_size_name(processorId, totalNumberOfProcessors, NodeName);

    _mat_ ker = getKernelFromFile<_mat_>(processorId, totalNumberOfProcessors, ordP);
    ulong kerColCnt = conv<ulong>(ker.NumCols());

    r_oe<_mat_, FiniteField>(ker);
    return false;
}
