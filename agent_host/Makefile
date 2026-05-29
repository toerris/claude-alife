CC      ?= cc
CFLAGS  ?= -O2 -Wall -Wextra -std=c11
LDLIBS  := -lm

agent_host: agent_host.o brain.o
	$(CC) $(CFLAGS) -o $@ agent_host.o brain.o $(LDLIBS)

agent_host.o: agent_host.c brain.h
brain.o: brain.c brain.h

clean:
	rm -f agent_host *.o

.PHONY: clean
