CC      = gcc
CFLAGS  = -g
VIVADO_DIR := /apps/Xilinx/Vivado/2022.2
VIVADO_VERILOG_SRC = ${VIVADO_DIR}/data/verilog/src/
PROJECT_DIR = $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
PROJECT_NAME := suzirjax
SYNTH_NAME := synth_1
IMPL_NAME := impl_1
JOBS := $(shell nproc)

VIVADO_SRC = glbl.v unisim_comp.v
#retarget_comp.v unisim_retarget_comp.v
IP_NETLIST = $(wildcard ${PROJECT_NAME}.gen/sources_1/ip/**/*_netlist.v) $(addprefix ${VIVADO_VERILOG_SRC},${VIVADO_SRC})
# SV_FLAGS := -DXIL_XECLIB
IVERILOG_FLAG = $(addprefix -l,${IP_NETLIST})
VERILOG_SRC = ${PROJECT_NAME}.src/sources_1/new

TCLSH := scripts/vivado_proxy.py
TCL_RUN_ARGS = -jobs $(JOBS)
VIVADO_LCK := .vivado_proxy.lck

# ZCU111 fabric
DEVICE := xczu28dr_0

# SYNTH_FILE = $(PROJECT_NAME).runs/$(SYNTH_NAME)/project.wdf
# IMPL_FILE = $(PROJECT_NAME).runs/$(IMPL_NAME)/project.wdf
BITSTREAM_FILE = $(PROJECT_NAME).runs/$(IMPL_NAME)/root_zcu111.bit

build/%: %.c
	$(CC) $(CFLAGS) -o $@ $^

build/%.vvp: ${PROJECT_NAME}.srcs/sources_1/new/%.sv
	iverilog -o $@ -g2012 -I${VERILOG_SRC} ${IVERILOG_FLAG} -s $(notdir $(basename $^)) $^

build/%.vcd: build/%.vvp
	cd build; vvp $(notdir $^)


vivado_proxy:
	$(TCLSH) -d "/tmp/vivado" -l "$(VIVADO_LCK)" -s -b "$(VIVADO_DIR)/bin/vivado"

$(VIVADO_LCK):
	$(error "Server is not running, run it with 'make vivado_proxy'")

$(BITSTREAM_FILE): $(VIVADO_LCK)
	scripts/check_project.sh "$(PROJECT_DIR)$(PROJECT_NAME).xpr"
	$(TCLSH) "reset_run $(IMPL_NAME)"
	$(TCLSH) "launch_runs $(IMPL_NAME) -to_step write_bitstream ${TCL_RUN_ARGS}"
	scripts/filewait.sh $@

program: $(VIVADO_LCK) $(BITSTREAM_FILE)
	scripts/check_project.sh "$(PROJECT_DIR)$(PROJECT_NAME).xpr"
	$(TCLSH) "open_hw_manager"
	$(TCLSH) "connect_hw_server -quiet"
	$(TCLSH) "set_property PROBES.FILE {} [get_hw_devices $(DEVICE)]"
	$(TCLSH) "set_property FULL_PROBES.FILE {} [get_hw_devices $(DEVICE)]"
	$(TCLSH) "set_property PROGRAM.FILE {$(PROJECT_DIR)$(BITSTREAM_FILE)} [get_hw_devices $(DEVICE)]"
	$(TCLSH) "program_hw_devices [get_hw_devices $(DEVICE)]"
	$(TCLSH) "refresh_hw_device [lindex [get_hw_devices $(DEVICE)] 0]"

# clean:
# 	@rm -rfv build/*
