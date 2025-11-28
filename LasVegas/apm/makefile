IDIR=./include
CC=mpic++
CFLAGS=-I$(IDIR) -lntl -lgmp -O0 -w

ODIR=./obj

_DEPS =  EC_GF2E_Point.hpp EC_GF2E.hpp dlp_input_2m.hpp \
		 lasVegas_utils.hpp EC_impl.tcc MPI_utils.hpp   \
		 dlp_input.hpp EC_ZZp.hpp EC_ZZp_Point.hpp constants.hpp \
		 lasVegas.tcc EC_lasVegas_impl.tcc searchAllTwoMinors.tcc \
		 utils.hpp schurComplement.tcc schurComplement_serial.tcc containment.tcc \
		 circularSwap.tcc eliminationTest.tcc Logger.hpp processBigMinor.tcc makeKernelDatabase.tcc \
		 RowCol.hpp processExternalMinors.tcc processBigMinors_parallel.tcc playground.tcc LU_Circular_principleMinor.tcc \
		 gaussianElimination_multiple.tcc GE_allRowAllPivot.tcc obliqueElimination.tcc almostPrinciple_deviation.tcc \
		 r_oe.tcc almostPrinciple_deviation_v2.tcc almostPrinciple_deviation_serial.tcc bruteForceAllMinor_Parallel.tcc \
		 apm_v2_small.tcc \

# $(info $(_DEPS))

DEPS = $(patsubst %,$(IDIR)/%,$(_DEPS))

_OBJ = main.o EC_GF2E_Point.o EC_GF2E.o dlp_input_2m.o dlp_input.o \
       randomNumber.o EC_ZZp.o EC_ZZp_Point.o Logger.o RowCol.o playground.o\
	   ./utils/kernel_utils.o ./utils/lasVegas_utils.o ./utils/MPI_utils.o \
	   ./utils/utils.o

OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ));

$(ODIR)/%.o: %.cpp $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

lasVegas: $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS) $(LIBS)

.PHONY: clean

clean:
	rm -f $(ODIR)/*.o *~ core $(INCDIR)/*~  kernel/*.txt randomNumbers/*.txt obj/utils/*.o output/*.*
