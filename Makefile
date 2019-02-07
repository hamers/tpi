#HDF_INSTALL = /usr/local/
#EXTLIB = -L$(HDF_INSTALL)/lib
CXX = g++
CFLAGS = -O3 -g 
LIBS = -fopenmp
#-L/$(HDF_INSTALL)/lib $(HDF_INSTALL)/lib/libhdf5_hl_cpp.a $(HDF_INSTALL)/lib/libhdf5_cpp.a $(HDF_INSTALL)/lib/libhdf5_hl.a $(HDF_INSTALL)/lib/libhdf5.a -L/$(HDF_INSTALL)/lib
SRCS = tpi.cpp helper_routines.cpp
OBJS = $(SRCS:.c=.o)
MAIN = tpi
#INCLUDE_HDF   = -I$(HDF_INSTALL)/include
#LIBS_HDF   = $(EXTLIB) $(HDF_INSTALL)/lib/libhdf5.a 

all:	$(MAIN)

$(MAIN): $(OBJS) 
	$(CXX) $(CFLAGS) -o $(MAIN) $(OBJS) $(LIBS) 
	#$(INCLUDE_HDF) $(LIBS_HDF)

.c.o:
	$(CXX) $(CFLAGS) -c $< -o $@

clean:
	$(RM) *.o *~ $(MAIN)


#tpi: tpi.cpp helper_functions.cpp
#	$(CXX) $(LIBS) $(CFLAGS) -o $@

#helper_functions: helper_functions.cpp
#	$(CXX) $(LDFLAGS) -o helper_functions

#helper_function: helper_functions.cpp
