# See README.txt.

.PHONY: all python clean

all: protoc

clean:
	rm -f nn_pb2.py
	rm -f *.pyc

protoc: nn.proto
	protoc --python_out=../ nn.proto
	protoc --python_out=../ run_params.proto

