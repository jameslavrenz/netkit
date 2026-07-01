# netkit Makefile
#
# Targets:
#   make              — netkit CLI + libnetkit.a
#   make test         — C++ API tests, then C API tests
#   make example-cpp  — C++26 usage demo (examples/infer_cpp)
#   make example-c    — C23 usage demo (examples/infer_c)
#
# See README.md and docs/GETTING_STARTED.md for full documentation.

CC = clang
CXX = clang++
CFLAGS = -fcolor-diagnostics -fansi-escape-codes -g -std=c23 -Wall -Wextra -Iinclude
CXXFLAGS = -fcolor-diagnostics -fansi-escape-codes -g -std=c++26 -Wall -Wextra -Iinclude
TARGET = netkit
LIB = libnetkit.a

CORE_SOURCES = src/arena.cpp src/tensor_factory.cpp src/tensor_access.cpp src/ops.cpp \
               src/conv2d.cpp src/mlp.cpp src/cnn.cpp src/json_parser.cpp \
               src/model_loader.cpp src/vectors_loader.cpp src/netkit_api.cpp \
               src/cli.cpp src/test.cpp
CLI_SOURCES = src/main.cpp

CORE_OBJECTS = $(CORE_SOURCES:.cpp=.o)
CLI_OBJECTS = $(CLI_SOURCES:.cpp=.o)

EXAMPLE_C = examples/infer_c
EXAMPLE_C_SRC = examples/infer_c.c
EXAMPLE_C_OBJ = examples/infer_c.o

EXAMPLE_CPP = examples/infer_cpp
EXAMPLE_CPP_SRC = examples/infer_cpp.cpp
EXAMPLE_CPP_OBJ = examples/infer_cpp.o

TEST_C = tests/test_c_api
TEST_C_SRC = tests/test_c_api.c
TEST_C_OBJ = tests/test_c_api.o

all: $(TARGET)

lib: $(LIB)

$(LIB): $(CORE_OBJECTS)
	ar rcs $@ $^

$(TARGET): $(LIB) $(CLI_OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $(CLI_OBJECTS) $(LIB)

$(EXAMPLE_C): $(LIB) $(EXAMPLE_C_OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $(EXAMPLE_C_OBJ) $(LIB)

$(EXAMPLE_CPP): $(LIB) $(EXAMPLE_CPP_OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $(EXAMPLE_CPP_OBJ) $(LIB)

$(TEST_C): $(LIB) $(TEST_C_OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $(TEST_C_OBJ) $(LIB)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(EXAMPLE_CPP_OBJ): $(EXAMPLE_CPP_SRC)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(EXAMPLE_C_OBJ): $(EXAMPLE_C_SRC) include/netkit.h
	$(CC) $(CFLAGS) -c $< -o $@

$(TEST_C_OBJ): $(TEST_C_SRC) include/netkit.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(CORE_OBJECTS) $(CLI_OBJECTS) $(EXAMPLE_C_OBJ) $(EXAMPLE_CPP_OBJ) $(TEST_C_OBJ) \
	      $(TARGET) $(LIB) $(EXAMPLE_C) $(EXAMPLE_CPP) $(TEST_C)

rebuild: clean all

# C++ API regression (primary CLI test path)
test-cpp: $(TARGET)
	./$(TARGET) test

# C API regression (C23 test harness only)
test-c: $(TEST_C)
	./$(TEST_C)

# Full suite: C++ API tests then C API tests
test: test-cpp test-c

run: test

example-c: $(EXAMPLE_C)

example-cpp: $(EXAMPLE_CPP)

examples: example-cpp example-c

.PHONY: all lib clean rebuild test test-cpp test-c run example-c example-cpp examples
