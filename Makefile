# Makefile

CXX = /usr/local/bin/g++-9
CXXFLAGS = -c -Wall -std=c++11 #-fopenmp

ORG = ./GibbsFE_minimization
SRC = ./GibbsFE_minimization

condensation: main.o grid.o diffuse.o 
	$(CXX) -Wall -fopenmp -o condensation main.o grid.o diffuse.o
main.o: main.cpp
	$(CXX) $(CXXFLAGS) main.cpp
grid.o: grid.cpp
	$(CXX) $(CXXFLAGS) grid.cpp
diffuse.o: diffuse.cpp
	$(CXX) $(CXXFLAGS) diffuse.cpp
clean:
	rm -f *.o condensation
