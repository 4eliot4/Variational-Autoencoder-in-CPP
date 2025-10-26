# ---- toolchain ---------------------------------------------------------------
EIGEN_DIR := /opt/homebrew/opt/eigen/include/eigen3
CXX        ?= clang++            # or g++
PKG_CONFIG ?= pkg-config
PKGS       := gtkmm-4.0

# ---- flags -------------------------------------------------------------------
CPPFLAGS := -I$(EIGEN_DIR)
CXXFLAGS := -std=c++17 -Wall -Wextra -Wpedantic
LDFLAGS  :=
#LDLIBS   := $(shell $(PKG_CONFIG) --libs $(PKGS))

# Build type: make [all] BUILD=debug | release
BUILD ?= debug
ifeq ($(BUILD),debug)
  CXXFLAGS += -O0 -g
  SANITIZERS ?= address,undefined
  CXXFLAGS += -fsanitize=$(SANITIZERS) -fno-omit-frame-pointer
  LDFLAGS  += -fsanitize=$(SANITIZERS)
else ifeq ($(BUILD),release)
  CXXFLAGS += -O3 -DNDEBUG
endif

# ---- layout ------------------------------------------------------------------
BUILD_DIR := build
TARGET := $(BUILD_DIR)/main

SRCS := $(wildcard *.cpp)
OBJS := $(patsubst %.cpp,$(BUILD_DIR)/%.o,$(SRCS))
DEPS := $(OBJS:.o=.d)

# Default goal
.DEFAULT_GOAL := all

# ---- targets -----------------------------------------------------------------
.PHONY: all clean run
all: $(TARGET)


# Link step
$(TARGET): $(OBJS) | $(BUILD_DIR)
	$(CXX) $(LDFLAGS) $^ -o $@ $(LDLIBS)

# Compile step
$(BUILD_DIR)/%.o: %.cpp | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -MMD -MP -c $< -o $@

# Create build dir
$(BUILD_DIR):
	@mkdir -p $@
# Convenience
run: $(TARGET)
	@$(TARGET)

clean:
	@echo "Cleaning build/"
	@rm -rf build

# Include auto-generated deps if they exist
-include $(DEPS)