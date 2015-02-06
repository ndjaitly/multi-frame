# See README.txt.

.PHONY: all python clean

all: protoc

clean:
	rm -f dbn_pb2.py
	rm -f *.pyc

protoc: dbn.proto
	protoc --python_out=../ dbn.proto

