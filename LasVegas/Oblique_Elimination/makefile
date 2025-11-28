IDIR=./include
CC=mpic++
CFLAGS=-I$(IDIR) -lntl -lgmp -O0 -w

ODIR=./obj

_DEPS =  EC_GF2E_Point.hpp EC_GF2E.hpp dlp_input_2m.hpp \
		 lasVegas_utils.hpp EC_impl.tcc MPI_utils.hpp   \
		 dlp_input.hpp EC_ZZp.hpp EC_ZZp_Point.hpp constants.hpp \
		 lasVegas.tcc EC_lasVegas_impl.tcc searchAllTwoMinors.tcc \
		 utils.hpp schurComplement.tcc schurComplement_serial.tcc containment.tcc \
		 eliminationTest.tcc Logger.hpp RowCol.hpp obliqueElimination.tcc \
		 
# $(info $(_DEPS))

DEPS = $(patsubst %,$(IDIR)/%,$(_DEPS))

_OBJ = main.o EC_GF2E_Point.o EC_GF2E.o dlp_input_2m.o dlp_input.o \
       randomNumber.o EC_ZZp.o EC_ZZp_Point.o Logger.o RowCol.o \
	   ./utils/kernel_utils.o ./utils/lasVegas_utils.o ./utils/MPI_utils.o \
	   ./utils/utils.o

OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ)) ;

$(ODIR)/%.o: %.cpp $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

lasVegas: $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS) $(LIBS)

.PHONY: clean

clean:
	rm -f $(ODIR)/*.o *~ core $(INCDIR)/*~  kernel/*.txt randomNumbers/*.txt obj/utils/*.o output/*.*
