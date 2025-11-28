/* 
 * File:   RowCol.hpp
 * Author: abdullah
 *
 * Created on 4 March, 2021, 14:20 PM
 */

#ifndef ROWCOL_HPP
#define ROWCOL_HPP

#include <NTL/ZZ.h>
#include <NTL/ZZ_p.h>

class RowCol
{
public:
    ulong dimension;
    ulong *rows;
    ulong *cols;

    RowCol() {}
    RowCol(ulong dimension);
    RowCol(const RowCol &rc);
    void init();
    void print(std::string msg) const;
    RowCol &operator=(const RowCol &);
};

#endif /* ROWCOL_HPP */