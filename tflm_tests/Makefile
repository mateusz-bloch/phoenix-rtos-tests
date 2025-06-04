TEST_SRCS  := $(wildcard tflm_tests/*_test.cc)

define build_one_test
NAME        := $(basename $(notdir $(1)))
LOCAL_SRCS  := $(notdir $(1))
LOCAL_LDLIBS+= -ltflm -lm -lstdc++ 
include $(binary.mk)
endef

$(foreach t,$(TEST_SRCS),$(eval $(call build_one_test,$(t))))
