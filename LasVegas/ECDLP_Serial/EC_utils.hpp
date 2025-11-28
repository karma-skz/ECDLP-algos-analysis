/* 
 * File:   utils.hpp
 * Author: abdullah
 *
 * Created on 26 December, 2017, 11:11 AM
 */

#ifndef EC_UTILS_HPP
#define EC_UTILS_HPP

#define B_RED_START "\033[1;31m"

#define B_WHITW_START "\033[1;37m"

#define RESET_TERM "\033[0m"

#define MASTER_NODE 0

#define verbosePrint 0
#define v_cout if(verbosePrint) cout
#define v_print if(verbosePrint)

#define masterPrint(id) if(1 && id ==MASTER_NODE )
#define masterCout(id) if(1) cout

void permute_randomNumbers(NTL::ZZ *PQ_randomNumbers, std::string permutation);

#endif /* UTILS_HPP */