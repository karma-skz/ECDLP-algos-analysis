#include "RowCol.hpp"

RowCol::RowCol(ulong dimension)
{
    this->dimension = dimension;
    rows = new ulong[dimension];
    cols = new ulong[dimension];
}

/**
 * Initialization of Rows and Columns
 */
void RowCol::init()
{
    for (int i = 0; i < dimension; i++)
    {
        this->rows[i] = i;
        this->cols[i] = i;
    }
}

/**
 * Copy Constructor...
 */
RowCol::RowCol(const RowCol &rc)
{
    dimension = rc.dimension;

    rows = new ulong[rc.dimension];
    cols = new ulong[rc.dimension];

    // Initialization of Rows and Columns
    for (int i = 0; i < rc.dimension; i++)
    {
        rows[i] = rc.rows[i];
        cols[i] = rc.cols[i];
    }
}

void RowCol::print(std::string msg = "") const
{
    if (this->dimension == 0)
        return;

    std::cout << msg << std::endl;
    for (ulong i = 0; i < this->dimension; ++i)
        std::cout << this->rows[i] << "\t";
    std::cout << std::endl;
    for (ulong i = 0; i < this->dimension; ++i)
        std::cout << this->cols[i] << "\t";
}

RowCol &RowCol::operator=(const RowCol &obj)
{
    if (this->dimension != obj.dimension)
    {
        std::cerr << " Dimensions mismatch " << this->dimension << "\t & " << obj.dimension << "\n";
        exit(0);
    }

    for (int i = 0; i < obj.dimension; i++)
    {
        this->rows[i] = obj.rows[i];
        this->cols[i] = obj.cols[i];
    }

    return *this;
}