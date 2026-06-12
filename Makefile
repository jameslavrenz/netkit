CXX = clang++
CXXFLAGS = -fcolor-diagnostics -fansi-escape-codes -g -std=c++11 -Iinclude
TARGET = main
SOURCES = src/main.cpp src/test.cpp src/arena.cpp src/tensor_factory.cpp src/tensor_access.cpp src/ops.cpp src/conv2d.cpp src/inference.cpp src/mlp.cpp src/cnn.cpp
OBJECTS = $(SOURCES:.cpp=.o)

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJECTS) $(TARGET)

rebuild: clean all

run: $(TARGET)
	./$(TARGET)

.PHONY: all clean rebuild run
