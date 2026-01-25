CXX = g++
CXXFLAGS = -std=c++11 -O3 -Wall

TARGET = inference_uint8
SRC = inference_uint8.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC)

clean:
	rm -f $(TARGET) *.bin *.txt *.pth

.PHONY: all clean
