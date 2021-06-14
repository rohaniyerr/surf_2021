# Makefile

CXX = /usr/bin/g++
CXXFLAGS = -c -Wall -std=c++11 #-fopenmp

ORG = ./GibbsFE_minimization
SRC = ./GibbsFE_minimization

condensation: main.o grid.o diffuse.o grain.o pebble.o vertical.o element.o gibbse.o initcomp.o massb.o solution.o CGgibbsmin.o
	$(CXX) -Wall -fopenmp -o condensation main.o grid.o diffuse.o grain.o pebble.o vertical.o element.o gibbse.o initcomp.o massb.o solution.o CGgibbsmin.o
main.o: main.cpp
	$(CXX) $(CXXFLAGS) main.cpp
grid.o: grid.cpp
	$(CXX) $(CXXFLAGS) grid.cpp
diffuse.o: diffuse.cpp
	$(CXX) $(CXXFLAGS) diffuse.cpp
grain.o: grain.cpp
	$(CXX) $(CXXFLAGS) grain.cpp
pebble.o: pebble.cpp
	$(CXX) $(CXXFLAGS) pebble.cpp
vertical.o: vertical.cpp
	$(CXX) $(CXXFLAGS) vertical.cpp
element.o: $(ORG)/element.cpp
	$(CXX) $(CXXFLAGS) $(ORG)/element.cpp
gibbse.o: $(SRC)/gibbse.cpp
	$(CXX) $(CXXFLAGS) $(SRC)/gibbse.cpp
initcomp.o: $(ORG)/initcomp.cpp
	$(CXX) $(CXXFLAGS) $(ORG)/initcomp.cpp
massb.o: $(SRC)/massb.cpp
	$(CXX) $(CXXFLAGS) $(ORG)/massb.cpp
solution.o: $(SRC)/solution.cpp
	$(CXX) $(CXXFLAGS) $(ORG)/solution.cpp
CGgibbsmin.o: $(SRC)/CGgibbsmin.cpp
	$(CXX) $(CXXFLAGS) $(SRC)/CGgibbsmin.cpp
clean:
	rm -f *.o condensation
