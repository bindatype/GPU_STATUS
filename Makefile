#This is a comment to disprove the claim that I do not use comments -- Glen
CC = nvcc
CFLAGS = -I ../../include -I /usr/include -lnvidia-ml -L /usr/lib/
LFLAGS = -Xcompiler -static

all: gpu_status

usage.o: usage.cpp
	$(CC) $(CFLAGS) -c usage.cpp

getters.o: getters.cpp
	$(CC) $(CFLAGS) -c getters.cpp

printHumanReadableResults.o: printHumanReadableResults.cpp
	$(CC) $(CFLAGS) -c printHumanReadableResults.cpp

printSQLOutput.o: printSQLOutput.cpp
	$(CC) $(CFLAGS) -c printSQLOutput.cpp

gpu_status: gpu_status.cpp usage.o getters.o printHumanReadableResults.o printSQLOutput.o
	$(CC) $(CFLAGS)  gpu_status.cpp -o gpu_status usage.o printHumanReadableResults.o printSQLOutput.o getters.o

clean:
	rm -f gpu_status usage.o getters.o printHumanReadableResults.o printSQLOutput.o

