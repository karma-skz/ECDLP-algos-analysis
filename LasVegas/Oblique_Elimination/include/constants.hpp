#ifndef CONSTANTS_HPP
#define CONSTANTS_HPP

const ulong TEN_MILLION = 10000000;
const ulong ONE_MILLION = 1000000;
const ulong HUNDRED_THOUSAND = 100000;
const ulong TEN_THOUSAND = 10000;
const ulong ONE_THOUSAND = 1000;
const ulong ONE_HUNDRED = 100;
const ulong FIFTY = 50;
const ulong FOURTY = 40;
const ulong THIRTY = 30;

#define B_RED_START "\033[1;31m"

#define B_WHITW_START "\033[1;37m"

#define B_MAGENTA_START "\u001b[35m"

#define B_YELLOW_START "\u001b[33m"

#define B_CYAN_START "\u001b[36m"

#define RESET_TERM "\033[0m"

#define MASTER_NODE 0

#define verbosePrint 0
#define v_cout        \
    if (verbosePrint) \
    cout

#define v_print if (verbosePrint)

#define masterPrint(id)    \
    if (id == MASTER_NODE) \
    cout

#define ENABLE_LOGGING 1

#define Kth_combination_InFile "./include/in.txt"
#define Kth_combination_OutFile "./include/out.txt"

#endif