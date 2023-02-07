############################################################################
##
##   ZCU111 Rev1.0 - Master XDC 05/30/2018
##
############################################################################
#Other net   PACKAGE_PIN AA16     - SYSMON_DXN                Bank   0 - DXN
#Other net   PACKAGE_PIN AA17     - SYSMON_DXP                Bank   0 - DXP
#Other net   PACKAGE_PIN V17      - SYSMON_AGND               Bank   0 - GNDADC
#Other net   PACKAGE_PIN AF12     - N16942408                 Bank   0 - POR_OVERRIDE
#Other net   PACKAGE_PIN AF14     - N16943469                 Bank   0 - PUDC_B
#Other net   PACKAGE_PIN V16      - FPGA_SYSMON_AVCC          Bank   0 - VCCADC
#Other net   PACKAGE_PIN Y16      - SYSMON_VN_R               Bank   0 - VN
#Other net   PACKAGE_PIN W17      - SYSMON_VP_R               Bank   0 - VP
#Other net   PACKAGE_PIN W16      - SYSMON_AGND               Bank   0 - VREFN
#Other net   PACKAGE_PIN Y17      - SYSMON_VREFP              Bank   0 - VREFP
set_property PACKAGE_PIN A16      [get_ports "PL_DDR4_CKE"] ;# Bank  69 VCCO - VCC1V2   - IO_L24N_T3U_N11_69
set_property IOSTANDARD  SSTL12   [get_ports "PL_DDR4_CKE"] ;# Bank  69 VCCO - VCC1V2   - IO_L24N_T3U_N11_69
set_property PACKAGE_PIN A17      [get_ports "PL_DDR4_RESET_B"] ;# Bank  69 VCCO - VCC1V2   - IO_L24P_T3U_N10_69
set_property IOSTANDARD  LVCMOS12 [get_ports "PL_DDR4_RESET_B"] ;# Bank  69 VCCO - VCC1V2   - IO_L24P_T3U_N10_69
set_property PACKAGE_PIN D15      [get_ports "PL_DDR4_CAS_B"] ;# Bank  69 VCCO - VCC1V2   - IO_L23N_T3U_N9_69
set_property IOSTANDARD  SSTL12   [get_ports "PL_DDR4_CAS_B"] ;# Bank  69 VCCO - VCC1V2   - IO_L23N_T3U_N9_69
set_property PACKAGE_PIN D16      [get_ports "PL_DDR4_CS_B"] ;# Bank  69 VCCO - VCC1V2   - IO_L23P_T3U_N8_69
set_property IOSTANDARD  SSTL12   [get_ports "PL_DDR4_CS_B"] ;# Bank  69 VCCO - VCC1V2   - IO_L23P_T3U_N8_69
set_property PACKAGE_PIN B17      [get_ports "PL_DDR4_WE_B"] ;# Bank  69 VCCO - VCC1V2   - IO_L22N_T3U_N7_DBC_AD0N_69
set_property IOSTANDARD  SSTL12   [get_ports "PL_DDR4_WE_B"] ;# Bank  69 VCCO - VCC1V2   - IO_L22N_T3U_N7_DBC_AD0N_69
set_property PACKAGE_PIN B18      [get_ports "PL_DDR4_ALERT_B"] ;# Bank  69 VCCO - VCC1V2   - IO_L22P_T3U_N6_DBC_AD0P_69
set_property IOSTANDARD  POD12    [get_ports "PL_DDR4_ALERT_B"] ;# Bank  69 VCCO - VCC1V2   - IO_L22P_T3U_N6_DBC_AD0P_69
set_property PACKAGE_PIN C16      [get_ports "PL_DDR4_BG0"] ;# Bank  69 VCCO - VCC1V2   - IO_L21N_T3L_N5_AD8N_69
set_property IOSTANDARD  SSTL12   [get_ports "PL_DDR4_BG0"] ;# Bank  69 VCCO - VCC1V2   - IO_L21N_T3L_N5_AD8N_69
set_property PACKAGE_PIN C17      [get_ports "PMOD0_0_LS"] ;# Bank  69 VCCO - VCC1V2   - IO_L21P_T3L_N4_AD8P_69
set_property IOSTANDARD  POD12   [get_ports "PMOD0_0_LS"] ;# Bank  69 VCCO - VCC1V2   - IO_L21P_T3L_N4_AD8P_69
set_property PACKAGE_PIN A19      [get_ports "PL_DDR4_ACT_B"] ;# Bank  69 VCCO - VCC1V2   - IO_L20N_T3L_N3_AD1N_69
set_property IOSTANDARD  SSTL12   [get_ports "PL_DDR4_ACT_B"] ;# Bank  69 VCCO - VCC1V2   - IO_L20N_T3L_N3_AD1N_69
set_property PACKAGE_PIN B19      [get_ports "PL_DDR4_ODT"] ;# Bank  69 VCCO - VCC1V2   - IO_L20P_T3L_N2_AD1P_69
set_property IOSTANDARD  SSTL12   [get_ports "PL_DDR4_ODT"] ;# Bank  69 VCCO - VCC1V2   - IO_L20P_T3L_N2_AD1P_69
set_property PACKAGE_PIN C18      [get_ports "PL_DDR4_RAS_B"] ;# Bank  69 VCCO - VCC1V2   - IO_L19N_T3L_N1_DBC_AD9N_69
set_property IOSTANDARD  SSTL12   [get_ports "PL_DDR4_RAS_B"] ;# Bank  69 VCCO - VCC1V2   - IO_L19N_T3L_N1_DBC_AD9N_69
set_property PACKAGE_PIN D18      [get_ports "PL_DDR4_A0"] ;# Bank  69 VCCO - VCC1V2   - IO_L19P_T3L_N0_DBC_AD9P_69
set_property IOSTANDARD  SSTL12   [get_ports "PL_DDR4_A0"] ;# Bank  69 VCCO - VCC1V2   - IO_L19P_T3L_N0_DBC_AD9P_69
set_property PACKAGE_PIN D19      [get_ports "PL_DDR4_PARITY"] ;# Bank  69 VCCO - VCC1V2   - IO_T3U_N12_69
set_property IOSTANDARD  POD12    [get_ports "PL_DDR4_PARITY"] ;# Bank  69 VCCO - VCC1V2   - IO_T3U_N12_69
set_property PACKAGE_PIN E19      [get_ports "PL_DDR4_A1"] ;# Bank  69 VCCO - VCC1V2   - IO_T2U_N12_69
set_property IOSTANDARD  SSTL12   [get_ports "PL_DDR4_A1"] ;# Bank  69 VCCO - VCC1V2   - IO_T2U_N12_69
set_property PACKAGE_PIN E17      [get_ports "PL_DDR4_A2"] ;# Bank  69 VCCO - VCC1V2   - IO_L18N_T2U_N11_AD2N_69
set_property IOSTANDARD  SSTL12   [get_ports "PL_DDR4_A2"] ;# Bank  69 VCCO - VCC1V2   - IO_L18N_T2U_N11_AD2N_69
set_property PACKAGE_PIN E18      [get_ports "PL_DDR4_A3"] ;# Bank  69 VCCO - VCC1V2   - IO_L18P_T2U_N10_AD2P_69
set_property IOSTANDARD  SSTL12   [get_ports "PL_DDR4_A3"] ;# Bank  69 VCCO - VCC1V2   - IO_L18P_T2U_N10_AD2P_69
set_property PACKAGE_PIN E16      [get_ports "PL_DDR4_A4"] ;# Bank  69 VCCO - VCC1V2   - IO_L17N_T2U_N9_AD10N_69
set_property IOSTANDARD  SSTL12   [get_ports "PL_DDR4_A4"] ;# Bank  69 VCCO - VCC1V2   - IO_L17N_T2U_N9_AD10N_69
set_property PACKAGE_PIN F16      [get_ports "PL_DDR4_A5"] ;# Bank  69 VCCO - VCC1V2   - IO_L17P_T2U_N8_AD10P_69
set_property IOSTANDARD  SSTL12   [get_ports "PL_DDR4_A5"] ;# Bank  69 VCCO - VCC1V2   - IO_L17P_T2U_N8_AD10P_69
set_property PACKAGE_PIN F19      [get_ports "PL_DDR4_A6"] ;# Bank  69 VCCO - VCC1V2   - IO_L16N_T2U_N7_QBC_AD3N_69
set_property IOSTANDARD  SSTL12   [get_ports "PL_DDR4_A6"] ;# Bank  69 VCCO - VCC1V2   - IO_L16N_T2U_N7_QBC_AD3N_69
set_property PACKAGE_PIN G19      [get_ports "PL_DDR4_A7"] ;# Bank  69 VCCO - VCC1V2   - IO_L16P_T2U_N6_QBC_AD3P_69
set_property IOSTANDARD  SSTL12   [get_ports "PL_DDR4_A7"] ;# Bank  69 VCCO - VCC1V2   - IO_L16P_T2U_N6_QBC_AD3P_69
set_property PACKAGE_PIN F15      [get_ports "PL_DDR4_A8"] ;# Bank  69 VCCO - VCC1V2   - IO_L15N_T2L_N5_AD11N_69
set_property IOSTANDARD  SSTL12   [get_ports "PL_DDR4_A8"] ;# Bank  69 VCCO - VCC1V2   - IO_L15N_T2L_N5_AD11N_69
set_property PACKAGE_PIN G15      [get_ports "PL_DDR4_A9"] ;# Bank  69 VCCO - VCC1V2   - IO_L15P_T2L_N4_AD11P_69
set_property IOSTANDARD  SSTL12   [get_ports "PL_DDR4_A9"] ;# Bank  69 VCCO - VCC1V2   - IO_L15P_T2L_N4_AD11P_69
set_property PACKAGE_PIN G18      [get_ports "PL_DDR4_A10"] ;# Bank  69 VCCO - VCC1V2   - IO_L14N_T2L_N3_GC_69
set_property IOSTANDARD  SSTL12   [get_ports "PL_DDR4_A10"] ;# Bank  69 VCCO - VCC1V2   - IO_L14N_T2L_N3_GC_69
set_property PACKAGE_PIN H18      [get_ports "PL_DDR4_A11"] ;# Bank  69 VCCO - VCC1V2   - IO_L14P_T2L_N2_GC_69
set_property IOSTANDARD  SSTL12   [get_ports "PL_DDR4_A11"] ;# Bank  69 VCCO - VCC1V2   - IO_L14P_T2L_N2_GC_69
set_property PACKAGE_PIN F17      [get_ports "PL_DDR4_CK0_C"] ;# Bank  69 VCCO - VCC1V2   - IO_L13N_T2L_N1_GC_QBC_69
set_property IOSTANDARD  DIFF_POD12 [get_ports "PL_DDR4_CK0_C"] ;# Bank  69 VCCO - VCC1V2   - IO_L13N_T2L_N1_GC_QBC_69
set_property PACKAGE_PIN G17      [get_ports "PL_DDR4_CK0_T"] ;# Bank  69 VCCO - VCC1V2   - IO_L13P_T2L_N0_GC_QBC_69
set_property IOSTANDARD  DIFF_POD12 [get_ports "PL_DDR4_CK0_T"] ;# Bank  69 VCCO - VCC1V2   - IO_L13P_T2L_N0_GC_QBC_69
set_property PACKAGE_PIN J18      [get_ports "USER_SI570_N"] ;# Bank  69 VCCO - VCC1V2   - IO_L12N_T1U_N11_GC_69
set_property IOSTANDARD  DIFF_POD12 [get_ports "USER_SI570_N"] ;# Bank  69 VCCO - VCC1V2   - IO_L12N_T1U_N11_GC_69
set_property PACKAGE_PIN J19      [get_ports "USER_SI570_P"] ;# Bank  69 VCCO - VCC1V2   - IO_L12P_T1U_N10_GC_69
set_property IOSTANDARD  DIFF_POD12 [get_ports "USER_SI570_P"] ;# Bank  69 VCCO - VCC1V2   - IO_L12P_T1U_N10_GC_69
set_property PACKAGE_PIN H16      [get_ports "PMOD0_2_LS"] ;# Bank  69 VCCO - VCC1V2   - IO_L11N_T1U_N9_GC_69
set_property IOSTANDARD  LVCMOS12 [get_ports "PMOD0_2_LS"] ;# Bank  69 VCCO - VCC1V2   - IO_L11N_T1U_N9_GC_69
set_property PACKAGE_PIN H17      [get_ports "PMOD0_3_LS"] ;# Bank  69 VCCO - VCC1V2   - IO_L11P_T1U_N8_GC_69
set_property IOSTANDARD  LVCMOS12 [get_ports "PMOD0_3_LS"] ;# Bank  69 VCCO - VCC1V2   - IO_L11P_T1U_N8_GC_69
set_property PACKAGE_PIN K18      [get_ports "PL_DDR4_BA0"] ;# Bank  69 VCCO - VCC1V2   - IO_L10N_T1U_N7_QBC_AD4N_69
set_property IOSTANDARD  SSTL12   [get_ports "PL_DDR4_BA0"] ;# Bank  69 VCCO - VCC1V2   - IO_L10N_T1U_N7_QBC_AD4N_69
set_property PACKAGE_PIN K19      [get_ports "PL_DDR4_BA1"] ;# Bank  69 VCCO - VCC1V2   - IO_L10P_T1U_N6_QBC_AD4P_69
set_property IOSTANDARD  SSTL12   [get_ports "PL_DDR4_BA1"] ;# Bank  69 VCCO - VCC1V2   - IO_L10P_T1U_N6_QBC_AD4P_69
set_property PACKAGE_PIN J16      [get_ports "PMOD0_4_LS"] ;# Bank  69 VCCO - VCC1V2   - IO_L9N_T1L_N5_AD12N_69
set_property IOSTANDARD  LVCMOS12 [get_ports "PMOD0_4_LS"] ;# Bank  69 VCCO - VCC1V2   - IO_L9N_T1L_N5_AD12N_69
set_property PACKAGE_PIN K16      [get_ports "PMOD0_5_LS"] ;# Bank  69 VCCO - VCC1V2   - IO_L9P_T1L_N4_AD12P_69
set_property IOSTANDARD  LVCMOS12 [get_ports "PMOD0_5_LS"] ;# Bank  69 VCCO - VCC1V2   - IO_L9P_T1L_N4_AD12P_69
set_property PACKAGE_PIN K17      [get_ports "PL_DDR4_A12"] ;# Bank  69 VCCO - VCC1V2   - IO_L8N_T1L_N3_AD5N_69
set_property IOSTANDARD  SSTL12   [get_ports "PL_DDR4_A12"] ;# Bank  69 VCCO - VCC1V2   - IO_L8N_T1L_N3_AD5N_69
set_property PACKAGE_PIN L17      [get_ports "PL_DDR4_A13"] ;# Bank  69 VCCO - VCC1V2   - IO_L8P_T1L_N2_AD5P_69
set_property IOSTANDARD  SSTL12   [get_ports "PL_DDR4_A13"] ;# Bank  69 VCCO - VCC1V2   - IO_L8P_T1L_N2_AD5P_69
set_property PACKAGE_PIN H15      [get_ports "PMOD0_6_LS"] ;# Bank  69 VCCO - VCC1V2   - IO_L7N_T1L_N1_QBC_AD13N_69
set_property IOSTANDARD  LVCMOS12 [get_ports "PMOD0_6_LS"] ;# Bank  69 VCCO - VCC1V2   - IO_L7N_T1L_N1_QBC_AD13N_69
set_property PACKAGE_PIN J15      [get_ports "PMOD0_7_LS"] ;# Bank  69 VCCO - VCC1V2   - IO_L7P_T1L_N0_QBC_AD13P_69
set_property IOSTANDARD  LVCMOS12 [get_ports "PMOD0_7_LS"] ;# Bank  69 VCCO - VCC1V2   - IO_L7P_T1L_N0_QBC_AD13P_69
set_property PACKAGE_PIN M18      [get_ports "PMOD0_1_LS"] ;# Bank  69 VCCO - VCC1V2   - IO_T1U_N12_69
set_property IOSTANDARD  LVCMOS12 [get_ports "PMOD0_1_LS"] ;# Bank  69 VCCO - VCC1V2   - IO_T1U_N12_69
set_property PACKAGE_PIN L16      [get_ports "VRP_69"] ;# Bank  69 VCCO - VCC1V2   - IO_T0U_N12_VRP_69
set_property IOSTANDARD  LVCMOSxx [get_ports "VRP_69"] ;# Bank  69 VCCO - VCC1V2   - IO_T0U_N12_VRP_69
set_property PACKAGE_PIN M17      [get_ports "MSP430_GPIO_PL_0_LS"] ;# Bank  69 VCCO - VCC1V2   - IO_L6N_T0U_N11_AD6N_69
set_property IOSTANDARD  LVCMOS12 [get_ports "MSP430_GPIO_PL_0_LS"] ;# Bank  69 VCCO - VCC1V2   - IO_L6N_T0U_N11_AD6N_69
set_property PACKAGE_PIN N17      [get_ports "MSP430_GPIO_PL_1_LS"] ;# Bank  69 VCCO - VCC1V2   - IO_L6P_T0U_N10_AD6P_69
set_property IOSTANDARD  LVCMOS12 [get_ports "MSP430_GPIO_PL_1_LS"] ;# Bank  69 VCCO - VCC1V2   - IO_L6P_T0U_N10_AD6P_69
set_property PACKAGE_PIN L12      [get_ports "MSP430_GPIO_PL_2_LS"] ;# Bank  69 VCCO - VCC1V2   - IO_L5N_T0U_N9_AD14N_69
set_property IOSTANDARD  LVCMOS12 [get_ports "MSP430_GPIO_PL_2_LS"] ;# Bank  69 VCCO - VCC1V2   - IO_L5N_T0U_N9_AD14N_69
set_property PACKAGE_PIN M12      [get_ports "MSP430_GPIO_PL_3_LS"] ;# Bank  69 VCCO - VCC1V2   - IO_L5P_T0U_N8_AD14P_69
set_property IOSTANDARD  LVCMOS12 [get_ports "MSP430_GPIO_PL_3_LS"] ;# Bank  69 VCCO - VCC1V2   - IO_L5P_T0U_N8_AD14P_69
set_property PACKAGE_PIN L14      [get_ports "PMOD1_0_LS"] ;# Bank  69 VCCO - VCC1V2   - IO_L4N_T0U_N7_DBC_AD7N_69
set_property IOSTANDARD  LVCMOS12 [get_ports "PMOD1_0_LS"] ;# Bank  69 VCCO - VCC1V2   - IO_L4N_T0U_N7_DBC_AD7N_69
set_property PACKAGE_PIN L15      [get_ports "PMOD1_1_LS"] ;# Bank  69 VCCO - VCC1V2   - IO_L4P_T0U_N6_DBC_AD7P_69
set_property IOSTANDARD  LVCMOS12 [get_ports "PMOD1_1_LS"] ;# Bank  69 VCCO - VCC1V2   - IO_L4P_T0U_N6_DBC_AD7P_69
set_property PACKAGE_PIN M13      [get_ports "PMOD1_2_LS"] ;# Bank  69 VCCO - VCC1V2   - IO_L3N_T0L_N5_AD15N_69
set_property IOSTANDARD  LVCMOS12 [get_ports "PMOD1_2_LS"] ;# Bank  69 VCCO - VCC1V2   - IO_L3N_T0L_N5_AD15N_69
set_property PACKAGE_PIN N13      [get_ports "PMOD1_3_LS"] ;# Bank  69 VCCO - VCC1V2   - IO_L3P_T0L_N4_AD15P_69
set_property IOSTANDARD  LVCMOS12 [get_ports "PMOD1_3_LS"] ;# Bank  69 VCCO - VCC1V2   - IO_L3P_T0L_N4_AD15P_69
set_property PACKAGE_PIN M15      [get_ports "PMOD1_4_LS"] ;# Bank  69 VCCO - VCC1V2   - IO_L2N_T0L_N3_69
set_property IOSTANDARD  LVCMOS12 [get_ports "PMOD1_4_LS"] ;# Bank  69 VCCO - VCC1V2   - IO_L2N_T0L_N3_69
set_property PACKAGE_PIN N15      [get_ports "PMOD1_5_LS"] ;# Bank  69 VCCO - VCC1V2   - IO_L2P_T0L_N2_69
set_property IOSTANDARD  LVCMOS12 [get_ports "PMOD1_5_LS"] ;# Bank  69 VCCO - VCC1V2   - IO_L2P_T0L_N2_69
set_property PACKAGE_PIN M14      [get_ports "PMOD1_6_LS"] ;# Bank  69 VCCO - VCC1V2   - IO_L1N_T0L_N1_DBC_69
set_property IOSTANDARD  LVCMOS12 [get_ports "PMOD1_6_LS"] ;# Bank  69 VCCO - VCC1V2   - IO_L1N_T0L_N1_DBC_69
set_property PACKAGE_PIN N14      [get_ports "PMOD1_7_LS"] ;# Bank  69 VCCO - VCC1V2   - IO_L1P_T0L_N0_DBC_69
set_property IOSTANDARD  LVCMOS12 [get_ports "PMOD1_7_LS"] ;# Bank  69 VCCO - VCC1V2   - IO_L1P_T0L_N0_DBC_69
#Other net   PACKAGE_PIN N16      - N21437344                 Bank  69 - VREF_69
set_property PACKAGE_PIN A14      [get_ports "PL_DDR4_DQ11"] ;# Bank  68 VCCO - VCC1V2   - IO_L24N_T3U_N11_68
set_property IOSTANDARD  POD12    [get_ports "PL_DDR4_DQ11"] ;# Bank  68 VCCO - VCC1V2   - IO_L24N_T3U_N11_68
set_property PACKAGE_PIN A15      [get_ports "PL_DDR4_DQ9"] ;# Bank  68 VCCO - VCC1V2   - IO_L24P_T3U_N10_68
set_property IOSTANDARD  POD12    [get_ports "PL_DDR4_DQ9"] ;# Bank  68 VCCO - VCC1V2   - IO_L24P_T3U_N10_68
set_property PACKAGE_PIN A11      [get_ports "PL_DDR4_DQ14"] ;# Bank  68 VCCO - VCC1V2   - IO_L23N_T3U_N9_68
set_property IOSTANDARD  POD12    [get_ports "PL_DDR4_DQ14"] ;# Bank  68 VCCO - VCC1V2   - IO_L23N_T3U_N9_68
set_property PACKAGE_PIN A12      [get_ports "PL_DDR4_DQ10"] ;# Bank  68 VCCO - VCC1V2   - IO_L23P_T3U_N8_68
set_property IOSTANDARD  POD12    [get_ports "PL_DDR4_DQ10"] ;# Bank  68 VCCO - VCC1V2   - IO_L23P_T3U_N8_68
set_property PACKAGE_PIN B15      [get_ports "PL_DDR4_DQS1_C"] ;# Bank  68 VCCO - VCC1V2   - IO_L22N_T3U_N7_DBC_AD0N_68
set_property IOSTANDARD  DIFF_POD12    [get_ports "PL_DDR4_DQS1_C"] ;# Bank  68 VCCO - VCC1V2   - IO_L22N_T3U_N7_DBC_AD0N_68
set_property PACKAGE_PIN C15      [get_ports "PL_DDR4_DQS1_T"] ;# Bank  68 VCCO - VCC1V2   - IO_L22P_T3U_N6_DBC_AD0P_68
set_property IOSTANDARD  DIFF_POD12    [get_ports "PL_DDR4_DQS1_T"] ;# Bank  68 VCCO - VCC1V2   - IO_L22P_T3U_N6_DBC_AD0P_68
set_property PACKAGE_PIN B13      [get_ports "PL_DDR4_DQ8"] ;# Bank  68 VCCO - VCC1V2   - IO_L21N_T3L_N5_AD8N_68
set_property IOSTANDARD  POD12    [get_ports "PL_DDR4_DQ8"] ;# Bank  68 VCCO - VCC1V2   - IO_L21N_T3L_N5_AD8N_68
set_property PACKAGE_PIN B14      [get_ports "PL_DDR4_DQ13"] ;# Bank  68 VCCO - VCC1V2   - IO_L21P_T3L_N4_AD8P_68
set_property IOSTANDARD  POD12    [get_ports "PL_DDR4_DQ13"] ;# Bank  68 VCCO - VCC1V2   - IO_L21P_T3L_N4_AD8P_68
set_property PACKAGE_PIN C13      [get_ports "PL_DDR4_DQ15"] ;# Bank  68 VCCO - VCC1V2   - IO_L20N_T3L_N3_AD1N_68
set_property IOSTANDARD  POD12    [get_ports "PL_DDR4_DQ15"] ;# Bank  68 VCCO - VCC1V2   - IO_L20N_T3L_N3_AD1N_68
set_property PACKAGE_PIN D13      [get_ports "PL_DDR4_DQ12"] ;# Bank  68 VCCO - VCC1V2   - IO_L20P_T3L_N2_AD1P_68
set_property IOSTANDARD  POD12    [get_ports "PL_DDR4_DQ12"] ;# Bank  68 VCCO - VCC1V2   - IO_L20P_T3L_N2_AD1P_68
set_property PACKAGE_PIN B12      [get_ports "SYSMON_SCL_LS"] ;# Bank  68 VCCO - VCC1V2   - IO_L19N_T3L_N1_DBC_AD9N_68
set_property IOSTANDARD  LVCMOS12 [get_ports "SYSMON_SCL_LS"] ;# Bank  68 VCCO - VCC1V2   - IO_L19N_T3L_N1_DBC_AD9N_68
set_property PACKAGE_PIN C12      [get_ports "PL_DDR4_DM1_B"] ;# Bank  68 VCCO - VCC1V2   - IO_L19P_T3L_N0_DBC_AD9P_68
set_property IOSTANDARD  POD12    [get_ports "PL_DDR4_DM1_B"] ;# Bank  68 VCCO - VCC1V2   - IO_L19P_T3L_N0_DBC_AD9P_68
set_property PACKAGE_PIN C11      [get_ports "No Connect"] ;# Bank  68 VCCO - VCC1V2   - IO_T3U_N12_68
set_property IOSTANDARD  LVCMOSxx [get_ports "No Connect"] ;# Bank  68 VCCO - VCC1V2   - IO_T3U_N12_68
set_property PACKAGE_PIN D11      [get_ports "SYSMON_SDA_LS"] ;# Bank  68 VCCO - VCC1V2   - IO_T2U_N12_68
set_property IOSTANDARD  LVCMOS12 [get_ports "SYSMON_SDA_LS"] ;# Bank  68 VCCO - VCC1V2   - IO_T2U_N12_68
set_property PACKAGE_PIN D14      [get_ports "PL_DDR4_DQ0"] ;# Bank  68 VCCO - VCC1V2   - IO_L18N_T2U_N11_AD2N_68
set_property IOSTANDARD  POD12    [get_ports "PL_DDR4_DQ0"] ;# Bank  68 VCCO - VCC1V2   - IO_L18N_T2U_N11_AD2N_68
set_property PACKAGE_PIN E14      [get_ports "PL_DDR4_DQ4"] ;# Bank  68 VCCO - VCC1V2   - IO_L18P_T2U_N10_AD2P_68
set_property IOSTANDARD  POD12    [get_ports "PL_DDR4_DQ4"] ;# Bank  68 VCCO - VCC1V2   - IO_L18P_T2U_N10_AD2P_68
set_property PACKAGE_PIN E11      [get_ports "PL_DDR4_DQ1"] ;# Bank  68 VCCO - VCC1V2   - IO_L17N_T2U_N9_AD10N_68
set_property IOSTANDARD  POD12    [get_ports "PL_DDR4_DQ1"] ;# Bank  68 VCCO - VCC1V2   - IO_L17N_T2U_N9_AD10N_68
set_property PACKAGE_PIN F12      [get_ports "PL_DDR4_DQ3"] ;# Bank  68 VCCO - VCC1V2   - IO_L17P_T2U_N8_AD10P_68
set_property IOSTANDARD  POD12    [get_ports "PL_DDR4_DQ3"] ;# Bank  68 VCCO - VCC1V2   - IO_L17P_T2U_N8_AD10P_68
set_property PACKAGE_PIN E12      [get_ports "PL_DDR4_DQS0_C"] ;# Bank  68 VCCO - VCC1V2   - IO_L16N_T2U_N7_QBC_AD3N_68
set_property IOSTANDARD  DIFF_POD12    [get_ports "PL_DDR4_DQS0_C"] ;# Bank  68 VCCO - VCC1V2   - IO_L16N_T2U_N7_QBC_AD3N_68
set_property PACKAGE_PIN E13      [get_ports "PL_DDR4_DQS0_T"] ;# Bank  68 VCCO - VCC1V2   - IO_L16P_T2U_N6_QBC_AD3P_68
set_property IOSTANDARD  DIFF_POD12    [get_ports "PL_DDR4_DQS0_T"] ;# Bank  68 VCCO - VCC1V2   - IO_L16P_T2U_N6_QBC_AD3P_68
set_property PACKAGE_PIN F14      [get_ports "PL_DDR4_DQ2"] ;# Bank  68 VCCO - VCC1V2   - IO_L15N_T2L_N5_AD11N_68
set_property IOSTANDARD  POD12    [get_ports "PL_DDR4_DQ2"] ;# Bank  68 VCCO - VCC1V2   - IO_L15N_T2L_N5_AD11N_68
set_property PACKAGE_PIN G14      [get_ports "PL_DDR4_DQ6"] ;# Bank  68 VCCO - VCC1V2   - IO_L15P_T2L_N4_AD11P_68
set_property IOSTANDARD  POD12    [get_ports "PL_DDR4_DQ6"] ;# Bank  68 VCCO - VCC1V2   - IO_L15P_T2L_N4_AD11P_68
set_property PACKAGE_PIN H12      [get_ports "PL_DDR4_DQ5"] ;# Bank  68 VCCO - VCC1V2   - IO_L14N_T2L_N3_GC_68
set_property IOSTANDARD  POD12    [get_ports "PL_DDR4_DQ5"] ;# Bank  68 VCCO - VCC1V2   - IO_L14N_T2L_N3_GC_68
set_property PACKAGE_PIN H13      [get_ports "PL_DDR4_DQ7"] ;# Bank  68 VCCO - VCC1V2   - IO_L14P_T2L_N2_GC_68
set_property IOSTANDARD  POD12    [get_ports "PL_DDR4_DQ7"] ;# Bank  68 VCCO - VCC1V2   - IO_L14P_T2L_N2_GC_68
set_property PACKAGE_PIN G12      [get_ports "SFP0_TX_DISABLE_B"] ;# Bank  68 VCCO - VCC1V2   - IO_L13N_T2L_N1_GC_QBC_68
set_property IOSTANDARD  LVCMOS12 [get_ports "SFP0_TX_DISABLE_B"] ;# Bank  68 VCCO - VCC1V2   - IO_L13N_T2L_N1_GC_QBC_68
set_property PACKAGE_PIN G13      [get_ports "PL_DDR4_DM0_B"] ;# Bank  68 VCCO - VCC1V2   - IO_L13P_T2L_N0_GC_QBC_68
set_property IOSTANDARD  POD12    [get_ports "PL_DDR4_DM0_B"] ;# Bank  68 VCCO - VCC1V2   - IO_L13P_T2L_N0_GC_QBC_68
set_property PACKAGE_PIN H10      [get_ports "PL_DDR4_DQ18"] ;# Bank  68 VCCO - VCC1V2   - IO_L12N_T1U_N11_GC_68
set_property IOSTANDARD  POD12    [get_ports "PL_DDR4_DQ18"] ;# Bank  68 VCCO - VCC1V2   - IO_L12N_T1U_N11_GC_68
set_property PACKAGE_PIN H11      [get_ports "PL_DDR4_DQ23"] ;# Bank  68 VCCO - VCC1V2   - IO_L12P_T1U_N10_GC_68
set_property IOSTANDARD  POD12    [get_ports "PL_DDR4_DQ23"] ;# Bank  68 VCCO - VCC1V2   - IO_L12P_T1U_N10_GC_68
set_property PACKAGE_PIN J10      [get_ports "PL_DDR4_DQ22"] ;# Bank  68 VCCO - VCC1V2   - IO_L11N_T1U_N9_GC_68
set_property IOSTANDARD  POD12    [get_ports "PL_DDR4_DQ22"] ;# Bank  68 VCCO - VCC1V2   - IO_L11N_T1U_N9_GC_68
set_property PACKAGE_PIN J11      [get_ports "PL_DDR4_DQ17"] ;# Bank  68 VCCO - VCC1V2   - IO_L11P_T1U_N8_GC_68
set_property IOSTANDARD  POD12    [get_ports "PL_DDR4_DQ17"] ;# Bank  68 VCCO - VCC1V2   - IO_L11P_T1U_N8_GC_68
set_property PACKAGE_PIN J13      [get_ports "PL_DDR4_DQS2_C"] ;# Bank  68 VCCO - VCC1V2   - IO_L10N_T1U_N7_QBC_AD4N_68
set_property IOSTANDARD  DIFF_POD12    [get_ports "PL_DDR4_DQS2_C"] ;# Bank  68 VCCO - VCC1V2   - IO_L10N_T1U_N7_QBC_AD4N_68
set_property PACKAGE_PIN J14      [get_ports "PL_DDR4_DQS2_T"] ;# Bank  68 VCCO - VCC1V2   - IO_L10P_T1U_N6_QBC_AD4P_68
set_property IOSTANDARD  DIFF_POD12    [get_ports "PL_DDR4_DQS2_T"] ;# Bank  68 VCCO - VCC1V2   - IO_L10P_T1U_N6_QBC_AD4P_68
set_property PACKAGE_PIN F10      [get_ports "PL_DDR4_DQ21"] ;# Bank  68 VCCO - VCC1V2   - IO_L9N_T1L_N5_AD12N_68
set_property IOSTANDARD  POD12    [get_ports "PL_DDR4_DQ21"] ;# Bank  68 VCCO - VCC1V2   - IO_L9N_T1L_N5_AD12N_68
set_property PACKAGE_PIN F11      [get_ports "PL_DDR4_DQ19"] ;# Bank  68 VCCO - VCC1V2   - IO_L9P_T1L_N4_AD12P_68
set_property IOSTANDARD  POD12    [get_ports "PL_DDR4_DQ19"] ;# Bank  68 VCCO - VCC1V2   - IO_L9P_T1L_N4_AD12P_68
set_property PACKAGE_PIN K10      [get_ports "PL_DDR4_DQ20"] ;# Bank  68 VCCO - VCC1V2   - IO_L8N_T1L_N3_AD5N_68
set_property IOSTANDARD  POD12    [get_ports "PL_DDR4_DQ20"] ;# Bank  68 VCCO - VCC1V2   - IO_L8N_T1L_N3_AD5N_68
set_property PACKAGE_PIN K11      [get_ports "PL_DDR4_DQ16"] ;# Bank  68 VCCO - VCC1V2   - IO_L8P_T1L_N2_AD5P_68
set_property IOSTANDARD  POD12    [get_ports "PL_DDR4_DQ16"] ;# Bank  68 VCCO - VCC1V2   - IO_L8P_T1L_N2_AD5P_68
set_property PACKAGE_PIN K12      [get_ports "SFP2_TX_DISABLE_B"] ;# Bank  68 VCCO - VCC1V2   - IO_L7N_T1L_N1_QBC_AD13N_68
set_property IOSTANDARD  LVCMOS12 [get_ports "SFP2_TX_DISABLE_B"] ;# Bank  68 VCCO - VCC1V2   - IO_L7N_T1L_N1_QBC_AD13N_68
set_property PACKAGE_PIN K13      [get_ports "PL_DDR4_DM2_B"] ;# Bank  68 VCCO - VCC1V2   - IO_L7P_T1L_N0_QBC_AD13P_68
set_property IOSTANDARD  POD12    [get_ports "PL_DDR4_DM2_B"] ;# Bank  68 VCCO - VCC1V2   - IO_L7P_T1L_N0_QBC_AD13P_68
set_property PACKAGE_PIN G10      [get_ports "SFP1_TX_DISABLE_B"] ;# Bank  68 VCCO - VCC1V2   - IO_T1U_N12_68
set_property IOSTANDARD  LVCMOS12 [get_ports "SFP1_TX_DISABLE_B"] ;# Bank  68 VCCO - VCC1V2   - IO_T1U_N12_68
set_property PACKAGE_PIN F7       [get_ports "VRP_68"] ;# Bank  68 VCCO - VCC1V2   - IO_T0U_N12_VRP_68
set_property IOSTANDARD  LVCMOSxx [get_ports "VRP_68"] ;# Bank  68 VCCO - VCC1V2   - IO_T0U_N12_VRP_68
set_property PACKAGE_PIN F9       [get_ports "PL_DDR4_DQ26"] ;# Bank  68 VCCO - VCC1V2   - IO_L6N_T0U_N11_AD6N_68
set_property IOSTANDARD  POD12    [get_ports "PL_DDR4_DQ26"] ;# Bank  68 VCCO - VCC1V2   - IO_L6N_T0U_N11_AD6N_68
set_property PACKAGE_PIN G9       [get_ports "PL_DDR4_DQ24"] ;# Bank  68 VCCO - VCC1V2   - IO_L6P_T0U_N10_AD6P_68
set_property IOSTANDARD  POD12    [get_ports "PL_DDR4_DQ24"] ;# Bank  68 VCCO - VCC1V2   - IO_L6P_T0U_N10_AD6P_68
set_property PACKAGE_PIN G6       [get_ports "PL_DDR4_DQ27"] ;# Bank  68 VCCO - VCC1V2   - IO_L5N_T0U_N9_AD14N_68
set_property IOSTANDARD  POD12    [get_ports "PL_DDR4_DQ27"] ;# Bank  68 VCCO - VCC1V2   - IO_L5N_T0U_N9_AD14N_68
set_property PACKAGE_PIN G7       [get_ports "PL_DDR4_DQ25"] ;# Bank  68 VCCO - VCC1V2   - IO_L5P_T0U_N8_AD14P_68
set_property IOSTANDARD  POD12    [get_ports "PL_DDR4_DQ25"] ;# Bank  68 VCCO - VCC1V2   - IO_L5P_T0U_N8_AD14P_68
set_property PACKAGE_PIN G8       [get_ports "PL_DDR4_DQS3_C"] ;# Bank  68 VCCO - VCC1V2   - IO_L4N_T0U_N7_DBC_AD7N_68
set_property IOSTANDARD  DIFF_POD12    [get_ports "PL_DDR4_DQS3_C"] ;# Bank  68 VCCO - VCC1V2   - IO_L4N_T0U_N7_DBC_AD7N_68
set_property PACKAGE_PIN H8       [get_ports "PL_DDR4_DQS3_T"] ;# Bank  68 VCCO - VCC1V2   - IO_L4P_T0U_N6_DBC_AD7P_68
set_property IOSTANDARD  DIFF_POD12    [get_ports "PL_DDR4_DQS3_T"] ;# Bank  68 VCCO - VCC1V2   - IO_L4P_T0U_N6_DBC_AD7P_68
set_property PACKAGE_PIN H6       [get_ports "PL_DDR4_DQ28"] ;# Bank  68 VCCO - VCC1V2   - IO_L3N_T0L_N5_AD15N_68
set_property IOSTANDARD  POD12    [get_ports "PL_DDR4_DQ28"] ;# Bank  68 VCCO - VCC1V2   - IO_L3N_T0L_N5_AD15N_68
set_property PACKAGE_PIN H7       [get_ports "PL_DDR4_DQ29"] ;# Bank  68 VCCO - VCC1V2   - IO_L3P_T0L_N4_AD15P_68
set_property IOSTANDARD  POD12    [get_ports "PL_DDR4_DQ29"] ;# Bank  68 VCCO - VCC1V2   - IO_L3P_T0L_N4_AD15P_68
set_property PACKAGE_PIN J9       [get_ports "PL_DDR4_DQ30"] ;# Bank  68 VCCO - VCC1V2   - IO_L2N_T0L_N3_68
set_property IOSTANDARD  POD12    [get_ports "PL_DDR4_DQ30"] ;# Bank  68 VCCO - VCC1V2   - IO_L2N_T0L_N3_68
set_property PACKAGE_PIN K9       [get_ports "PL_DDR4_DQ31"] ;# Bank  68 VCCO - VCC1V2   - IO_L2P_T0L_N2_68
set_property IOSTANDARD  POD12    [get_ports "PL_DDR4_DQ31"] ;# Bank  68 VCCO - VCC1V2   - IO_L2P_T0L_N2_68
set_property PACKAGE_PIN J7       [get_ports "SFP3_TX_DISABLE_B"] ;# Bank  68 VCCO - VCC1V2   - IO_L1N_T0L_N1_DBC_68
set_property IOSTANDARD  LVCMOS12 [get_ports "SFP3_TX_DISABLE_B"] ;# Bank  68 VCCO - VCC1V2   - IO_L1N_T0L_N1_DBC_68
set_property PACKAGE_PIN J8       [get_ports "PL_DDR4_DM3_B"] ;# Bank  68 VCCO - VCC1V2   - IO_L1P_T0L_N0_DBC_68
set_property IOSTANDARD  POD12    [get_ports "PL_DDR4_DM3_B"] ;# Bank  68 VCCO - VCC1V2   - IO_L1P_T0L_N0_DBC_68
#Other net   PACKAGE_PIN K14      - N21437174                 Bank  68 - VREF_68
set_property PACKAGE_PIN A21      [get_ports "PL_DDR4_DQ34"] ;# Bank  67 VCCO - VCC1V2   - IO_L24N_T3U_N11_67
set_property IOSTANDARD  POD12    [get_ports "PL_DDR4_DQ34"] ;# Bank  67 VCCO - VCC1V2   - IO_L24N_T3U_N11_67
set_property PACKAGE_PIN A20      [get_ports "PL_DDR4_DQ33"] ;# Bank  67 VCCO - VCC1V2   - IO_L24P_T3U_N10_67
set_property IOSTANDARD  POD12    [get_ports "PL_DDR4_DQ33"] ;# Bank  67 VCCO - VCC1V2   - IO_L24P_T3U_N10_67
set_property PACKAGE_PIN B20      [get_ports "PL_DDR4_DQ37"] ;# Bank  67 VCCO - VCC1V2   - IO_L23N_T3U_N9_67
set_property IOSTANDARD  POD12    [get_ports "PL_DDR4_DQ37"] ;# Bank  67 VCCO - VCC1V2   - IO_L23N_T3U_N9_67
set_property PACKAGE_PIN C20      [get_ports "PL_DDR4_DQ39"] ;# Bank  67 VCCO - VCC1V2   - IO_L23P_T3U_N8_67
set_property IOSTANDARD  POD12    [get_ports "PL_DDR4_DQ39"] ;# Bank  67 VCCO - VCC1V2   - IO_L23P_T3U_N8_67
set_property PACKAGE_PIN A22      [get_ports "PL_DDR4_DQS4_C"] ;# Bank  67 VCCO - VCC1V2   - IO_L22N_T3U_N7_DBC_AD0N_67
set_property IOSTANDARD  DIFF_POD12    [get_ports "PL_DDR4_DQS4_C"] ;# Bank  67 VCCO - VCC1V2   - IO_L22N_T3U_N7_DBC_AD0N_67
set_property PACKAGE_PIN B22      [get_ports "PL_DDR4_DQS4_T"] ;# Bank  67 VCCO - VCC1V2   - IO_L22P_T3U_N6_DBC_AD0P_67
set_property IOSTANDARD  DIFF_POD12    [get_ports "PL_DDR4_DQS4_T"] ;# Bank  67 VCCO - VCC1V2   - IO_L22P_T3U_N6_DBC_AD0P_67
set_property PACKAGE_PIN C22      [get_ports "PL_DDR4_DQ32"] ;# Bank  67 VCCO - VCC1V2   - IO_L21N_T3L_N5_AD8N_67
set_property IOSTANDARD  POD12    [get_ports "PL_DDR4_DQ32"] ;# Bank  67 VCCO - VCC1V2   - IO_L21N_T3L_N5_AD8N_67
set_property PACKAGE_PIN C21      [get_ports "PL_DDR4_DQ35"] ;# Bank  67 VCCO - VCC1V2   - IO_L21P_T3L_N4_AD8P_67
set_property IOSTANDARD  POD12    [get_ports "PL_DDR4_DQ35"] ;# Bank  67 VCCO - VCC1V2   - IO_L21P_T3L_N4_AD8P_67
set_property PACKAGE_PIN A24      [get_ports "PL_DDR4_DQ36"] ;# Bank  67 VCCO - VCC1V2   - IO_L20N_T3L_N3_AD1N_67
set_property IOSTANDARD  POD12    [get_ports "PL_DDR4_DQ36"] ;# Bank  67 VCCO - VCC1V2   - IO_L20N_T3L_N3_AD1N_67
set_property PACKAGE_PIN B24      [get_ports "PL_DDR4_DQ38"] ;# Bank  67 VCCO - VCC1V2   - IO_L20P_T3L_N2_AD1P_67
set_property IOSTANDARD  POD12    [get_ports "PL_DDR4_DQ38"] ;# Bank  67 VCCO - VCC1V2   - IO_L20P_T3L_N2_AD1P_67
set_property PACKAGE_PIN B23      [get_ports "No Connect"] ;# Bank  67 VCCO - VCC1V2   - IO_L19N_T3L_N1_DBC_AD9N_67
set_property IOSTANDARD  LVCMOSxx [get_ports "No Connect"] ;# Bank  67 VCCO - VCC1V2   - IO_L19N_T3L_N1_DBC_AD9N_67
set_property PACKAGE_PIN C23      [get_ports "PL_DDR4_DM4_B"] ;# Bank  67 VCCO - VCC1V2   - IO_L19P_T3L_N0_DBC_AD9P_67
set_property IOSTANDARD  POD12    [get_ports "PL_DDR4_DM4_B"] ;# Bank  67 VCCO - VCC1V2   - IO_L19P_T3L_N0_DBC_AD9P_67
set_property PACKAGE_PIN A25      [get_ports "No Connect"] ;# Bank  67 VCCO - VCC1V2   - IO_T3U_N12_67
set_property IOSTANDARD  LVCMOSxx [get_ports "No Connect"] ;# Bank  67 VCCO - VCC1V2   - IO_T3U_N12_67
set_property PACKAGE_PIN D20      [get_ports "No Connect"] ;# Bank  67 VCCO - VCC1V2   - IO_T2U_N12_67
set_property IOSTANDARD  LVCMOSxx [get_ports "No Connect"] ;# Bank  67 VCCO - VCC1V2   - IO_T2U_N12_67
set_property PACKAGE_PIN D21      [get_ports "PL_DDR4_DQ47"] ;# Bank  67 VCCO - VCC1V2   - IO_L18N_T2U_N11_AD2N_67
set_property IOSTANDARD  POD12    [get_ports "PL_DDR4_DQ47"] ;# Bank  67 VCCO - VCC1V2   - IO_L18N_T2U_N11_AD2N_67
set_property PACKAGE_PIN E21      [get_ports "PL_DDR4_DQ45"] ;# Bank  67 VCCO - VCC1V2   - IO_L18P_T2U_N10_AD2P_67
set_property IOSTANDARD  POD12    [get_ports "PL_DDR4_DQ45"] ;# Bank  67 VCCO - VCC1V2   - IO_L18P_T2U_N10_AD2P_67
set_property PACKAGE_PIN E23      [get_ports "PL_DDR4_DQ42"] ;# Bank  67 VCCO - VCC1V2   - IO_L17N_T2U_N9_AD10N_67
set_property IOSTANDARD  POD12    [get_ports "PL_DDR4_DQ42"] ;# Bank  67 VCCO - VCC1V2   - IO_L17N_T2U_N9_AD10N_67
set_property PACKAGE_PIN E22      [get_ports "PL_DDR4_DQ41"] ;# Bank  67 VCCO - VCC1V2   - IO_L17P_T2U_N8_AD10P_67
set_property IOSTANDARD  POD12    [get_ports "PL_DDR4_DQ41"] ;# Bank  67 VCCO - VCC1V2   - IO_L17P_T2U_N8_AD10P_67
set_property PACKAGE_PIN D24      [get_ports "PL_DDR4_DQS5_C"] ;# Bank  67 VCCO - VCC1V2   - IO_L16N_T2U_N7_QBC_AD3N_67
set_property IOSTANDARD  DIFF_POD12    [get_ports "PL_DDR4_DQS5_C"] ;# Bank  67 VCCO - VCC1V2   - IO_L16N_T2U_N7_QBC_AD3N_67
set_property PACKAGE_PIN D23      [get_ports "PL_DDR4_DQS5_T"] ;# Bank  67 VCCO - VCC1V2   - IO_L16P_T2U_N6_QBC_AD3P_67
set_property IOSTANDARD  DIFF_POD12    [get_ports "PL_DDR4_DQS5_T"] ;# Bank  67 VCCO - VCC1V2   - IO_L16P_T2U_N6_QBC_AD3P_67
set_property PACKAGE_PIN E24      [get_ports "PL_DDR4_DQ40"] ;# Bank  67 VCCO - VCC1V2   - IO_L15N_T2L_N5_AD11N_67
set_property IOSTANDARD  POD12    [get_ports "PL_DDR4_DQ40"] ;# Bank  67 VCCO - VCC1V2   - IO_L15N_T2L_N5_AD11N_67
set_property PACKAGE_PIN F24      [get_ports "PL_DDR4_DQ44"] ;# Bank  67 VCCO - VCC1V2   - IO_L15P_T2L_N4_AD11P_67
set_property IOSTANDARD  POD12    [get_ports "PL_DDR4_DQ44"] ;# Bank  67 VCCO - VCC1V2   - IO_L15P_T2L_N4_AD11P_67
set_property PACKAGE_PIN F20      [get_ports "PL_DDR4_DQ46"] ;# Bank  67 VCCO - VCC1V2   - IO_L14N_T2L_N3_GC_67
set_property IOSTANDARD  POD12    [get_ports "PL_DDR4_DQ46"] ;# Bank  67 VCCO - VCC1V2   - IO_L14N_T2L_N3_GC_67
set_property PACKAGE_PIN G20      [get_ports "PL_DDR4_DQ43"] ;# Bank  67 VCCO - VCC1V2   - IO_L14P_T2L_N2_GC_67
set_property IOSTANDARD  POD12    [get_ports "PL_DDR4_DQ43"] ;# Bank  67 VCCO - VCC1V2   - IO_L14P_T2L_N2_GC_67
set_property PACKAGE_PIN F22      [get_ports "No Connect"] ;# Bank  67 VCCO - VCC1V2   - IO_L13N_T2L_N1_GC_QBC_67
set_property IOSTANDARD  LVCMOSxx [get_ports "No Connect"] ;# Bank  67 VCCO - VCC1V2   - IO_L13N_T2L_N1_GC_QBC_67
set_property PACKAGE_PIN F21      [get_ports "PL_DDR4_DM5_B"] ;# Bank  67 VCCO - VCC1V2   - IO_L13P_T2L_N0_GC_QBC_67
set_property IOSTANDARD  POD12    [get_ports "PL_DDR4_DM5_B"] ;# Bank  67 VCCO - VCC1V2   - IO_L13P_T2L_N0_GC_QBC_67
set_property PACKAGE_PIN G23      [get_ports "PL_DDR4_DQ49"] ;# Bank  67 VCCO - VCC1V2   - IO_L12N_T1U_N11_GC_67
set_property IOSTANDARD  POD12    [get_ports "PL_DDR4_DQ49"] ;# Bank  67 VCCO - VCC1V2   - IO_L12N_T1U_N11_GC_67
set_property PACKAGE_PIN H23      [get_ports "PL_DDR4_DQ48"] ;# Bank  67 VCCO - VCC1V2   - IO_L12P_T1U_N10_GC_67
set_property IOSTANDARD  POD12    [get_ports "PL_DDR4_DQ48"] ;# Bank  67 VCCO - VCC1V2   - IO_L12P_T1U_N10_GC_67
set_property PACKAGE_PIN G22      [get_ports "PL_DDR4_DQ51"] ;# Bank  67 VCCO - VCC1V2   - IO_L11N_T1U_N9_GC_67
set_property IOSTANDARD  POD12    [get_ports "PL_DDR4_DQ51"] ;# Bank  67 VCCO - VCC1V2   - IO_L11N_T1U_N9_GC_67
set_property PACKAGE_PIN H22      [get_ports "PL_DDR4_DQ53"] ;# Bank  67 VCCO - VCC1V2   - IO_L11P_T1U_N8_GC_67
set_property IOSTANDARD  POD12    [get_ports "PL_DDR4_DQ53"] ;# Bank  67 VCCO - VCC1V2   - IO_L11P_T1U_N8_GC_67
set_property PACKAGE_PIN H20      [get_ports "PL_DDR4_DQS6_C"] ;# Bank  67 VCCO - VCC1V2   - IO_L10N_T1U_N7_QBC_AD4N_67
set_property IOSTANDARD  DIFF_POD12    [get_ports "PL_DDR4_DQS6_C"] ;# Bank  67 VCCO - VCC1V2   - IO_L10N_T1U_N7_QBC_AD4N_67
set_property PACKAGE_PIN J20      [get_ports "PL_DDR4_DQS6_T"] ;# Bank  67 VCCO - VCC1V2   - IO_L10P_T1U_N6_QBC_AD4P_67
set_property IOSTANDARD  DIFF_POD12    [get_ports "PL_DDR4_DQS6_T"] ;# Bank  67 VCCO - VCC1V2   - IO_L10P_T1U_N6_QBC_AD4P_67
set_property PACKAGE_PIN H21      [get_ports "PL_DDR4_DQ55"] ;# Bank  67 VCCO - VCC1V2   - IO_L9N_T1L_N5_AD12N_67
set_property IOSTANDARD  POD12    [get_ports "PL_DDR4_DQ55"] ;# Bank  67 VCCO - VCC1V2   - IO_L9N_T1L_N5_AD12N_67
set_property PACKAGE_PIN J21      [get_ports "PL_DDR4_DQ52"] ;# Bank  67 VCCO - VCC1V2   - IO_L9P_T1L_N4_AD12P_67
set_property IOSTANDARD  POD12    [get_ports "PL_DDR4_DQ52"] ;# Bank  67 VCCO - VCC1V2   - IO_L9P_T1L_N4_AD12P_67
set_property PACKAGE_PIN K24      [get_ports "PL_DDR4_DQ50"] ;# Bank  67 VCCO - VCC1V2   - IO_L8N_T1L_N3_AD5N_67
set_property IOSTANDARD  POD12    [get_ports "PL_DDR4_DQ50"] ;# Bank  67 VCCO - VCC1V2   - IO_L8N_T1L_N3_AD5N_67
set_property PACKAGE_PIN L24      [get_ports "PL_DDR4_DQ54"] ;# Bank  67 VCCO - VCC1V2   - IO_L8P_T1L_N2_AD5P_67
set_property IOSTANDARD  POD12    [get_ports "PL_DDR4_DQ54"] ;# Bank  67 VCCO - VCC1V2   - IO_L8P_T1L_N2_AD5P_67
set_property PACKAGE_PIN J24      [get_ports "No Connect"] ;# Bank  67 VCCO - VCC1V2   - IO_L7N_T1L_N1_QBC_AD13N_67
set_property IOSTANDARD  LVCMOSxx [get_ports "No Connect"] ;# Bank  67 VCCO - VCC1V2   - IO_L7N_T1L_N1_QBC_AD13N_67
set_property PACKAGE_PIN J23      [get_ports "PL_DDR4_DM6_B"] ;# Bank  67 VCCO - VCC1V2   - IO_L7P_T1L_N0_QBC_AD13P_67
set_property IOSTANDARD  POD12    [get_ports "PL_DDR4_DM6_B"] ;# Bank  67 VCCO - VCC1V2   - IO_L7P_T1L_N0_QBC_AD13P_67
set_property PACKAGE_PIN G24      [get_ports "No Connect"] ;# Bank  67 VCCO - VCC1V2   - IO_T1U_N12_67
set_property IOSTANDARD  LVCMOSxx [get_ports "No Connect"] ;# Bank  67 VCCO - VCC1V2   - IO_T1U_N12_67
set_property PACKAGE_PIN K23      [get_ports "VRP_67"] ;# Bank  67 VCCO - VCC1V2   - IO_T0U_N12_VRP_67
set_property IOSTANDARD  LVCMOSxx [get_ports "VRP_67"] ;# Bank  67 VCCO - VCC1V2   - IO_T0U_N12_VRP_67
set_property PACKAGE_PIN L20      [get_ports "PL_DDR4_DQ57"] ;# Bank  67 VCCO - VCC1V2   - IO_L6N_T0U_N11_AD6N_67
set_property IOSTANDARD  POD12    [get_ports "PL_DDR4_DQ57"] ;# Bank  67 VCCO - VCC1V2   - IO_L6N_T0U_N11_AD6N_67
set_property PACKAGE_PIN L19      [get_ports "PL_DDR4_DQ61"] ;# Bank  67 VCCO - VCC1V2   - IO_L6P_T0U_N10_AD6P_67
set_property IOSTANDARD  POD12    [get_ports "PL_DDR4_DQ61"] ;# Bank  67 VCCO - VCC1V2   - IO_L6P_T0U_N10_AD6P_67
set_property PACKAGE_PIN L23      [get_ports "PL_DDR4_DQ56"] ;# Bank  67 VCCO - VCC1V2   - IO_L5N_T0U_N9_AD14N_67
set_property IOSTANDARD  POD12    [get_ports "PL_DDR4_DQ56"] ;# Bank  67 VCCO - VCC1V2   - IO_L5N_T0U_N9_AD14N_67
set_property PACKAGE_PIN L22      [get_ports "PL_DDR4_DQ58"] ;# Bank  67 VCCO - VCC1V2   - IO_L5P_T0U_N8_AD14P_67
set_property IOSTANDARD  POD12    [get_ports "PL_DDR4_DQ58"] ;# Bank  67 VCCO - VCC1V2   - IO_L5P_T0U_N8_AD14P_67
set_property PACKAGE_PIN K22      [get_ports "PL_DDR4_DQS7_C"] ;# Bank  67 VCCO - VCC1V2   - IO_L4N_T0U_N7_DBC_AD7N_67
set_property IOSTANDARD  DIFF_POD12    [get_ports "PL_DDR4_DQS7_C"] ;# Bank  67 VCCO - VCC1V2   - IO_L4N_T0U_N7_DBC_AD7N_67
set_property PACKAGE_PIN K21      [get_ports "PL_DDR4_DQS7_T"] ;# Bank  67 VCCO - VCC1V2   - IO_L4P_T0U_N6_DBC_AD7P_67
set_property IOSTANDARD  DIFF_POD12    [get_ports "PL_DDR4_DQS7_T"] ;# Bank  67 VCCO - VCC1V2   - IO_L4P_T0U_N6_DBC_AD7P_67
set_property PACKAGE_PIN L21      [get_ports "PL_DDR4_DQ59"] ;# Bank  67 VCCO - VCC1V2   - IO_L3N_T0L_N5_AD15N_67
set_property IOSTANDARD  POD12    [get_ports "PL_DDR4_DQ59"] ;# Bank  67 VCCO - VCC1V2   - IO_L3N_T0L_N5_AD15N_67
set_property PACKAGE_PIN M20      [get_ports "PL_DDR4_DQ60"] ;# Bank  67 VCCO - VCC1V2   - IO_L3P_T0L_N4_AD15P_67
set_property IOSTANDARD  POD12    [get_ports "PL_DDR4_DQ60"] ;# Bank  67 VCCO - VCC1V2   - IO_L3P_T0L_N4_AD15P_67
set_property PACKAGE_PIN M19      [get_ports "PL_DDR4_DQ62"] ;# Bank  67 VCCO - VCC1V2   - IO_L2N_T0L_N3_67
set_property IOSTANDARD  POD12    [get_ports "PL_DDR4_DQ62"] ;# Bank  67 VCCO - VCC1V2   - IO_L2N_T0L_N3_67
set_property PACKAGE_PIN N19      [get_ports "PL_DDR4_DQ63"] ;# Bank  67 VCCO - VCC1V2   - IO_L2P_T0L_N2_67
set_property IOSTANDARD  POD12    [get_ports "PL_DDR4_DQ63"] ;# Bank  67 VCCO - VCC1V2   - IO_L2P_T0L_N2_67
set_property PACKAGE_PIN N21      [get_ports "No Connect"] ;# Bank  67 VCCO - VCC1V2   - IO_L1N_T0L_N1_DBC_67
set_property IOSTANDARD  LVCMOSxx [get_ports "No Connect"] ;# Bank  67 VCCO - VCC1V2   - IO_L1N_T0L_N1_DBC_67
set_property PACKAGE_PIN N20      [get_ports "PL_DDR4_DM7_B"] ;# Bank  67 VCCO - VCC1V2   - IO_L1P_T0L_N0_DBC_67
set_property IOSTANDARD  POD12    [get_ports "PL_DDR4_DM7_B"] ;# Bank  67 VCCO - VCC1V2   - IO_L1P_T0L_N0_DBC_67
#Other net   PACKAGE_PIN N18      - N21436724                 Bank  67 - VREF_67
set_property PACKAGE_PIN AF19     [get_ports "FMCP_HSPC_LA32_N"] ;# Bank  66 VCCO - VADJ_FMC - IO_L24N_T3U_N11_66
set_property IOSTANDARD  LVCMOS18 [get_ports "FMCP_HSPC_LA32_N"] ;# Bank  66 VCCO - VADJ_FMC - IO_L24N_T3U_N11_66
set_property PACKAGE_PIN AF20     [get_ports "FMCP_HSPC_LA32_P"] ;# Bank  66 VCCO - VADJ_FMC - IO_L24P_T3U_N10_66
set_property IOSTANDARD  LVCMOS18 [get_ports "FMCP_HSPC_LA32_P"] ;# Bank  66 VCCO - VADJ_FMC - IO_L24P_T3U_N10_66
set_property PACKAGE_PIN AH18     [get_ports "FMCP_HSPC_LA33_N"] ;# Bank  66 VCCO - VADJ_FMC - IO_L23N_T3U_N9_66
set_property IOSTANDARD  LVCMOS18 [get_ports "FMCP_HSPC_LA33_N"] ;# Bank  66 VCCO - VADJ_FMC - IO_L23N_T3U_N9_66
set_property PACKAGE_PIN AG18     [get_ports "FMCP_HSPC_LA33_P"] ;# Bank  66 VCCO - VADJ_FMC - IO_L23P_T3U_N8_66
set_property IOSTANDARD  LVCMOS18 [get_ports "FMCP_HSPC_LA33_P"] ;# Bank  66 VCCO - VADJ_FMC - IO_L23P_T3U_N8_66
set_property PACKAGE_PIN AH20     [get_ports "FMCP_HSPC_LA30_N"] ;# Bank  66 VCCO - VADJ_FMC - IO_L22N_T3U_N7_DBC_AD0N_66
set_property IOSTANDARD  LVCMOS18 [get_ports "FMCP_HSPC_LA30_N"] ;# Bank  66 VCCO - VADJ_FMC - IO_L22N_T3U_N7_DBC_AD0N_66
set_property PACKAGE_PIN AG20     [get_ports "FMCP_HSPC_LA30_P"] ;# Bank  66 VCCO - VADJ_FMC - IO_L22P_T3U_N6_DBC_AD0P_66
set_property IOSTANDARD  LVCMOS18 [get_ports "FMCP_HSPC_LA30_P"] ;# Bank  66 VCCO - VADJ_FMC - IO_L22P_T3U_N6_DBC_AD0P_66
set_property PACKAGE_PIN AJ19     [get_ports "FMCP_HSPC_LA31_N"] ;# Bank  66 VCCO - VADJ_FMC - IO_L21N_T3L_N5_AD8N_66
set_property IOSTANDARD  LVCMOS18 [get_ports "FMCP_HSPC_LA31_N"] ;# Bank  66 VCCO - VADJ_FMC - IO_L21N_T3L_N5_AD8N_66
set_property PACKAGE_PIN AJ20     [get_ports "FMCP_HSPC_LA31_P"] ;# Bank  66 VCCO - VADJ_FMC - IO_L21P_T3L_N4_AD8P_66
set_property IOSTANDARD  LVCMOS18 [get_ports "FMCP_HSPC_LA31_P"] ;# Bank  66 VCCO - VADJ_FMC - IO_L21P_T3L_N4_AD8P_66
set_property PACKAGE_PIN AK21     [get_ports "FMCP_HSPC_LA29_N"] ;# Bank  66 VCCO - VADJ_FMC - IO_L20N_T3L_N3_AD1N_66
set_property IOSTANDARD  LVCMOS18 [get_ports "FMCP_HSPC_LA29_N"] ;# Bank  66 VCCO - VADJ_FMC - IO_L20N_T3L_N3_AD1N_66
set_property PACKAGE_PIN AK22     [get_ports "FMCP_HSPC_LA29_P"] ;# Bank  66 VCCO - VADJ_FMC - IO_L20P_T3L_N2_AD1P_66
set_property IOSTANDARD  LVCMOS18 [get_ports "FMCP_HSPC_LA29_P"] ;# Bank  66 VCCO - VADJ_FMC - IO_L20P_T3L_N2_AD1P_66
set_property PACKAGE_PIN AK18     [get_ports "FMCP_HSPC_LA28_N"] ;# Bank  66 VCCO - VADJ_FMC - IO_L19N_T3L_N1_DBC_AD9N_66
set_property IOSTANDARD  LVCMOS18 [get_ports "FMCP_HSPC_LA28_N"] ;# Bank  66 VCCO - VADJ_FMC - IO_L19N_T3L_N1_DBC_AD9N_66
set_property PACKAGE_PIN AJ18     [get_ports "FMCP_HSPC_LA28_P"] ;# Bank  66 VCCO - VADJ_FMC - IO_L19P_T3L_N0_DBC_AD9P_66
set_property IOSTANDARD  LVCMOS18 [get_ports "FMCP_HSPC_LA28_P"] ;# Bank  66 VCCO - VADJ_FMC - IO_L19P_T3L_N0_DBC_AD9P_66
set_property PACKAGE_PIN AK19     [get_ports "No Connect"] ;# Bank  66 VCCO - VADJ_FMC - IO_T3U_N12_66
set_property IOSTANDARD  LVCMOSxx [get_ports "No Connect"] ;# Bank  66 VCCO - VADJ_FMC - IO_T3U_N12_66
set_property PACKAGE_PIN AN22     [get_ports "No Connect"] ;# Bank  66 VCCO - VADJ_FMC - IO_T2U_N12_66
set_property IOSTANDARD  LVCMOSxx [get_ports "No Connect"] ;# Bank  66 VCCO - VADJ_FMC - IO_T2U_N12_66
set_property PACKAGE_PIN AL20     [get_ports "FMCP_HSPC_LA25_N"] ;# Bank  66 VCCO - VADJ_FMC - IO_L18N_T2U_N11_AD2N_66
set_property IOSTANDARD  LVCMOS18 [get_ports "FMCP_HSPC_LA25_N"] ;# Bank  66 VCCO - VADJ_FMC - IO_L18N_T2U_N11_AD2N_66
set_property PACKAGE_PIN AL21     [get_ports "FMCP_HSPC_LA25_P"] ;# Bank  66 VCCO - VADJ_FMC - IO_L18P_T2U_N10_AD2P_66
set_property IOSTANDARD  LVCMOS18 [get_ports "FMCP_HSPC_LA25_P"] ;# Bank  66 VCCO - VADJ_FMC - IO_L18P_T2U_N10_AD2P_66
set_property PACKAGE_PIN AM19     [get_ports "FMCP_HSPC_LA21_N"] ;# Bank  66 VCCO - VADJ_FMC - IO_L17N_T2U_N9_AD10N_66
set_property IOSTANDARD  LVCMOS18 [get_ports "FMCP_HSPC_LA21_N"] ;# Bank  66 VCCO - VADJ_FMC - IO_L17N_T2U_N9_AD10N_66
set_property PACKAGE_PIN AL19     [get_ports "FMCP_HSPC_LA21_P"] ;# Bank  66 VCCO - VADJ_FMC - IO_L17P_T2U_N8_AD10P_66
set_property IOSTANDARD  LVCMOS18 [get_ports "FMCP_HSPC_LA21_P"] ;# Bank  66 VCCO - VADJ_FMC - IO_L17P_T2U_N8_AD10P_66
set_property PACKAGE_PIN AM22     [get_ports "FMCP_HSPC_LA24_N"] ;# Bank  66 VCCO - VADJ_FMC - IO_L16N_T2U_N7_QBC_AD3N_66
set_property IOSTANDARD  LVCMOS18 [get_ports "FMCP_HSPC_LA24_N"] ;# Bank  66 VCCO - VADJ_FMC - IO_L16N_T2U_N7_QBC_AD3N_66
set_property PACKAGE_PIN AL22     [get_ports "FMCP_HSPC_LA24_P"] ;# Bank  66 VCCO - VADJ_FMC - IO_L16P_T2U_N6_QBC_AD3P_66
set_property IOSTANDARD  LVCMOS18 [get_ports "FMCP_HSPC_LA24_P"] ;# Bank  66 VCCO - VADJ_FMC - IO_L16P_T2U_N6_QBC_AD3P_66
set_property PACKAGE_PIN AN18     [get_ports "FMCP_HSPC_LA23_N"] ;# Bank  66 VCCO - VADJ_FMC - IO_L15N_T2L_N5_AD11N_66
set_property IOSTANDARD  LVCMOS18 [get_ports "FMCP_HSPC_LA23_N"] ;# Bank  66 VCCO - VADJ_FMC - IO_L15N_T2L_N5_AD11N_66
set_property PACKAGE_PIN AM18     [get_ports "FMCP_HSPC_LA23_P"] ;# Bank  66 VCCO - VADJ_FMC - IO_L15P_T2L_N4_AD11P_66
set_property IOSTANDARD  LVCMOS18 [get_ports "FMCP_HSPC_LA23_P"] ;# Bank  66 VCCO - VADJ_FMC - IO_L15P_T2L_N4_AD11P_66
set_property PACKAGE_PIN AP21     [get_ports "FMCP_HSPC_LA17_CC_N"] ;# Bank  66 VCCO - VADJ_FMC - IO_L14N_T2L_N3_GC_66
set_property IOSTANDARD  LVCMOS18 [get_ports "FMCP_HSPC_LA17_CC_N"] ;# Bank  66 VCCO - VADJ_FMC - IO_L14N_T2L_N3_GC_66
set_property PACKAGE_PIN AN21     [get_ports "FMCP_HSPC_LA17_CC_P"] ;# Bank  66 VCCO - VADJ_FMC - IO_L14P_T2L_N2_GC_66
set_property IOSTANDARD  LVCMOS18 [get_ports "FMCP_HSPC_LA17_CC_P"] ;# Bank  66 VCCO - VADJ_FMC - IO_L14P_T2L_N2_GC_66
set_property PACKAGE_PIN AN20     [get_ports "FMCP_HSPC_LA18_CC_N"] ;# Bank  66 VCCO - VADJ_FMC - IO_L13N_T2L_N1_GC_QBC_66
set_property IOSTANDARD  LVCMOS18 [get_ports "FMCP_HSPC_LA18_CC_N"] ;# Bank  66 VCCO - VADJ_FMC - IO_L13N_T2L_N1_GC_QBC_66
set_property PACKAGE_PIN AM20     [get_ports "FMCP_HSPC_LA18_CC_P"] ;# Bank  66 VCCO - VADJ_FMC - IO_L13P_T2L_N0_GC_QBC_66
set_property IOSTANDARD  LVCMOS18 [get_ports "FMCP_HSPC_LA18_CC_P"] ;# Bank  66 VCCO - VADJ_FMC - IO_L13P_T2L_N0_GC_QBC_66
set_property PACKAGE_PIN AP19     [get_ports "FMCP_HSPC_CLK1_M2C_N"] ;# Bank  66 VCCO - VADJ_FMC - IO_L12N_T1U_N11_GC_66
set_property IOSTANDARD  LVDS     [get_ports "FMCP_HSPC_CLK1_M2C_N"] ;# Bank  66 VCCO - VADJ_FMC - IO_L12N_T1U_N11_GC_66
set_property PACKAGE_PIN AP20     [get_ports "FMCP_HSPC_CLK1_M2C_P"] ;# Bank  66 VCCO - VADJ_FMC - IO_L12P_T1U_N10_GC_66
set_property IOSTANDARD  LVDS     [get_ports "FMCP_HSPC_CLK1_M2C_P"] ;# Bank  66 VCCO - VADJ_FMC - IO_L12P_T1U_N10_GC_66
set_property PACKAGE_PIN AR18     [get_ports "No Connect"] ;# Bank  66 VCCO - VADJ_FMC - IO_L11N_T1U_N9_GC_66
set_property IOSTANDARD  LVCMOSxx [get_ports "No Connect"] ;# Bank  66 VCCO - VADJ_FMC - IO_L11N_T1U_N9_GC_66
set_property PACKAGE_PIN AP18     [get_ports "No Connect"] ;# Bank  66 VCCO - VADJ_FMC - IO_L11P_T1U_N8_GC_66
set_property IOSTANDARD  LVCMOSxx [get_ports "No Connect"] ;# Bank  66 VCCO - VADJ_FMC - IO_L11P_T1U_N8_GC_66
set_property PACKAGE_PIN AT22     [get_ports "FMCP_HSPC_LA26_N"] ;# Bank  66 VCCO - VADJ_FMC - IO_L10N_T1U_N7_QBC_AD4N_66
set_property IOSTANDARD  LVCMOS18 [get_ports "FMCP_HSPC_LA26_N"] ;# Bank  66 VCCO - VADJ_FMC - IO_L10N_T1U_N7_QBC_AD4N_66
set_property PACKAGE_PIN AR22     [get_ports "FMCP_HSPC_LA26_P"] ;# Bank  66 VCCO - VADJ_FMC - IO_L10P_T1U_N6_QBC_AD4P_66
set_property IOSTANDARD  LVCMOS18 [get_ports "FMCP_HSPC_LA26_P"] ;# Bank  66 VCCO - VADJ_FMC - IO_L10P_T1U_N6_QBC_AD4P_66
set_property PACKAGE_PIN AT19     [get_ports "FMCP_HSPC_LA22_N"] ;# Bank  66 VCCO - VADJ_FMC - IO_L9N_T1L_N5_AD12N_66
set_property IOSTANDARD  LVCMOS18 [get_ports "FMCP_HSPC_LA22_N"] ;# Bank  66 VCCO - VADJ_FMC - IO_L9N_T1L_N5_AD12N_66
set_property PACKAGE_PIN AR19     [get_ports "FMCP_HSPC_LA22_P"] ;# Bank  66 VCCO - VADJ_FMC - IO_L9P_T1L_N4_AD12P_66
set_property IOSTANDARD  LVCMOS18 [get_ports "FMCP_HSPC_LA22_P"] ;# Bank  66 VCCO - VADJ_FMC - IO_L9P_T1L_N4_AD12P_66
set_property PACKAGE_PIN AT21     [get_ports "FMCP_HSPC_LA27_N"] ;# Bank  66 VCCO - VADJ_FMC - IO_L8N_T1L_N3_AD5N_66
set_property IOSTANDARD  LVCMOS18 [get_ports "FMCP_HSPC_LA27_N"] ;# Bank  66 VCCO - VADJ_FMC - IO_L8N_T1L_N3_AD5N_66
set_property PACKAGE_PIN AR21     [get_ports "FMCP_HSPC_LA27_P"] ;# Bank  66 VCCO - VADJ_FMC - IO_L8P_T1L_N2_AD5P_66
set_property IOSTANDARD  LVCMOS18 [get_ports "FMCP_HSPC_LA27_P"] ;# Bank  66 VCCO - VADJ_FMC - IO_L8P_T1L_N2_AD5P_66
set_property PACKAGE_PIN AT17     [get_ports "FMCP_HSPC_LA20_N"] ;# Bank  66 VCCO - VADJ_FMC - IO_L7N_T1L_N1_QBC_AD13N_66
set_property IOSTANDARD  LVCMOS18 [get_ports "FMCP_HSPC_LA20_N"] ;# Bank  66 VCCO - VADJ_FMC - IO_L7N_T1L_N1_QBC_AD13N_66
set_property PACKAGE_PIN AR17     [get_ports "FMCP_HSPC_LA20_P"] ;# Bank  66 VCCO - VADJ_FMC - IO_L7P_T1L_N0_QBC_AD13P_66
set_property IOSTANDARD  LVCMOS18 [get_ports "FMCP_HSPC_LA20_P"] ;# Bank  66 VCCO - VADJ_FMC - IO_L7P_T1L_N0_QBC_AD13P_66
set_property PACKAGE_PIN AT20     [get_ports "No Connect"] ;# Bank  66 VCCO - VADJ_FMC - IO_T1U_N12_66
set_property IOSTANDARD  LVCMOSxx [get_ports "No Connect"] ;# Bank  66 VCCO - VADJ_FMC - IO_T1U_N12_66
set_property PACKAGE_PIN AU22     [get_ports "VRP_66"] ;# Bank  66 VCCO - VADJ_FMC - IO_T0U_N12_VRP_66
set_property IOSTANDARD  LVCMOSxx [get_ports "VRP_66"] ;# Bank  66 VCCO - VADJ_FMC - IO_T0U_N12_VRP_66
set_property PACKAGE_PIN AU19     [get_ports "FMCP_HSPC_LA19_N"] ;# Bank  66 VCCO - VADJ_FMC - IO_L6N_T0U_N11_AD6N_66
set_property IOSTANDARD  LVCMOS18 [get_ports "FMCP_HSPC_LA19_N"] ;# Bank  66 VCCO - VADJ_FMC - IO_L6N_T0U_N11_AD6N_66
set_property PACKAGE_PIN AU20     [get_ports "FMCP_HSPC_LA19_P"] ;# Bank  66 VCCO - VADJ_FMC - IO_L6P_T0U_N10_AD6P_66
set_property IOSTANDARD  LVCMOS18 [get_ports "FMCP_HSPC_LA19_P"] ;# Bank  66 VCCO - VADJ_FMC - IO_L6P_T0U_N10_AD6P_66
set_property PACKAGE_PIN AV18     [get_ports "No Connect"] ;# Bank  66 VCCO - VADJ_FMC - IO_L5N_T0U_N9_AD14N_66
set_property IOSTANDARD  LVCMOSxx [get_ports "No Connect"] ;# Bank  66 VCCO - VADJ_FMC - IO_L5N_T0U_N9_AD14N_66
set_property PACKAGE_PIN AU18     [get_ports "No Connect"] ;# Bank  66 VCCO - VADJ_FMC - IO_L5P_T0U_N8_AD14P_66
set_property IOSTANDARD  LVCMOSxx [get_ports "No Connect"] ;# Bank  66 VCCO - VADJ_FMC - IO_L5P_T0U_N8_AD14P_66
set_property PACKAGE_PIN AW21     [get_ports "No Connect"] ;# Bank  66 VCCO - VADJ_FMC - IO_L4N_T0U_N7_DBC_AD7N_66
set_property IOSTANDARD  LVCMOSxx [get_ports "No Connect"] ;# Bank  66 VCCO - VADJ_FMC - IO_L4N_T0U_N7_DBC_AD7N_66
set_property PACKAGE_PIN AV21     [get_ports "No Connect"] ;# Bank  66 VCCO - VADJ_FMC - IO_L4P_T0U_N6_DBC_AD7P_66
set_property IOSTANDARD  LVCMOSxx [get_ports "No Connect"] ;# Bank  66 VCCO - VADJ_FMC - IO_L4P_T0U_N6_DBC_AD7P_66
set_property PACKAGE_PIN AV17     [get_ports "No Connect"] ;# Bank  66 VCCO - VADJ_FMC - IO_L3N_T0L_N5_AD15N_66
set_property IOSTANDARD  LVCMOSxx [get_ports "No Connect"] ;# Bank  66 VCCO - VADJ_FMC - IO_L3N_T0L_N5_AD15N_66
set_property PACKAGE_PIN AU17     [get_ports "No Connect"] ;# Bank  66 VCCO - VADJ_FMC - IO_L3P_T0L_N4_AD15P_66
set_property IOSTANDARD  LVCMOSxx [get_ports "No Connect"] ;# Bank  66 VCCO - VADJ_FMC - IO_L3P_T0L_N4_AD15P_66
set_property PACKAGE_PIN AW20     [get_ports "No Connect"] ;# Bank  66 VCCO - VADJ_FMC - IO_L2N_T0L_N3_66
set_property IOSTANDARD  LVCMOSxx [get_ports "No Connect"] ;# Bank  66 VCCO - VADJ_FMC - IO_L2N_T0L_N3_66
set_property PACKAGE_PIN AV20     [get_ports "No Connect"] ;# Bank  66 VCCO - VADJ_FMC - IO_L2P_T0L_N2_66
set_property IOSTANDARD  LVCMOSxx [get_ports "No Connect"] ;# Bank  66 VCCO - VADJ_FMC - IO_L2P_T0L_N2_66
set_property PACKAGE_PIN AW18     [get_ports "No Connect"] ;# Bank  66 VCCO - VADJ_FMC - IO_L1N_T0L_N1_DBC_66
set_property IOSTANDARD  LVCMOSxx [get_ports "No Connect"] ;# Bank  66 VCCO - VADJ_FMC - IO_L1N_T0L_N1_DBC_66
set_property PACKAGE_PIN AW19     [get_ports "No Connect"] ;# Bank  66 VCCO - VADJ_FMC - IO_L1P_T0L_N0_DBC_66
set_property IOSTANDARD  LVCMOSxx [get_ports "No Connect"] ;# Bank  66 VCCO - VADJ_FMC - IO_L1P_T0L_N0_DBC_66
#Other net   PACKAGE_PIN AG19     - N21116918                 Bank  66 - VREF_66
set_property PACKAGE_PIN AJ13     [get_ports "FMCP_HSPC_LA02_N"] ;# Bank  65 VCCO - VADJ_FMC - IO_L24N_T3U_N11_PERSTN0_65
set_property IOSTANDARD  LVCMOS18 [get_ports "FMCP_HSPC_LA02_N"] ;# Bank  65 VCCO - VADJ_FMC - IO_L24N_T3U_N11_PERSTN0_65
set_property PACKAGE_PIN AH13     [get_ports "FMCP_HSPC_LA02_P"] ;# Bank  65 VCCO - VADJ_FMC - IO_L24P_T3U_N10_PERSTN1_I2C_SDA_65
set_property IOSTANDARD  LVCMOS18 [get_ports "FMCP_HSPC_LA02_P"] ;# Bank  65 VCCO - VADJ_FMC - IO_L24P_T3U_N10_PERSTN1_I2C_SDA_65
set_property PACKAGE_PIN AH12     [get_ports "FMCP_HSPC_LA04_N"] ;# Bank  65 VCCO - VADJ_FMC - IO_L23N_T3U_N9_65
set_property IOSTANDARD  LVCMOS18 [get_ports "FMCP_HSPC_LA04_N"] ;# Bank  65 VCCO - VADJ_FMC - IO_L23N_T3U_N9_65
set_property PACKAGE_PIN AG12     [get_ports "FMCP_HSPC_LA04_P"] ;# Bank  65 VCCO - VADJ_FMC - IO_L23P_T3U_N8_I2C_SCLK_65
set_property IOSTANDARD  LVCMOS18 [get_ports "FMCP_HSPC_LA04_P"] ;# Bank  65 VCCO - VADJ_FMC - IO_L23P_T3U_N8_I2C_SCLK_65
set_property PACKAGE_PIN AK14     [get_ports "FMCP_HSPC_LA15_N"] ;# Bank  65 VCCO - VADJ_FMC - IO_L22N_T3U_N7_DBC_AD0N_65
set_property IOSTANDARD  LVCMOS18 [get_ports "FMCP_HSPC_LA15_N"] ;# Bank  65 VCCO - VADJ_FMC - IO_L22N_T3U_N7_DBC_AD0N_65
set_property PACKAGE_PIN AJ14     [get_ports "FMCP_HSPC_LA15_P"] ;# Bank  65 VCCO - VADJ_FMC - IO_L22P_T3U_N6_DBC_AD0P_65
set_property IOSTANDARD  LVCMOS18 [get_ports "FMCP_HSPC_LA15_P"] ;# Bank  65 VCCO - VADJ_FMC - IO_L22P_T3U_N6_DBC_AD0P_65
set_property PACKAGE_PIN AK12     [get_ports "FMCP_HSPC_LA03_N"] ;# Bank  65 VCCO - VADJ_FMC - IO_L21N_T3L_N5_AD8N_65
set_property IOSTANDARD  LVCMOS18 [get_ports "FMCP_HSPC_LA03_N"] ;# Bank  65 VCCO - VADJ_FMC - IO_L21N_T3L_N5_AD8N_65
set_property PACKAGE_PIN AJ12     [get_ports "FMCP_HSPC_LA03_P"] ;# Bank  65 VCCO - VADJ_FMC - IO_L21P_T3L_N4_AD8P_65
set_property IOSTANDARD  LVCMOS18 [get_ports "FMCP_HSPC_LA03_P"] ;# Bank  65 VCCO - VADJ_FMC - IO_L21P_T3L_N4_AD8P_65
set_property PACKAGE_PIN AM14     [get_ports "FMCP_HSPC_LA14_N"] ;# Bank  65 VCCO - VADJ_FMC - IO_L20N_T3L_N3_AD1N_65
set_property IOSTANDARD  LVCMOS18 [get_ports "FMCP_HSPC_LA14_N"] ;# Bank  65 VCCO - VADJ_FMC - IO_L20N_T3L_N3_AD1N_65
set_property PACKAGE_PIN AL14     [get_ports "FMCP_HSPC_LA14_P"] ;# Bank  65 VCCO - VADJ_FMC - IO_L20P_T3L_N2_AD1P_65
set_property IOSTANDARD  LVCMOS18 [get_ports "FMCP_HSPC_LA14_P"] ;# Bank  65 VCCO - VADJ_FMC - IO_L20P_T3L_N2_AD1P_65
set_property PACKAGE_PIN AL12     [get_ports "FMCP_HSPC_LA07_N"] ;# Bank  65 VCCO - VADJ_FMC - IO_L19N_T3L_N1_DBC_AD9N_65
set_property IOSTANDARD  LVCMOS18 [get_ports "FMCP_HSPC_LA07_N"] ;# Bank  65 VCCO - VADJ_FMC - IO_L19N_T3L_N1_DBC_AD9N_65
set_property PACKAGE_PIN AK13     [get_ports "FMCP_HSPC_LA07_P"] ;# Bank  65 VCCO - VADJ_FMC - IO_L19P_T3L_N0_DBC_AD9P_65
set_property IOSTANDARD  LVCMOS18 [get_ports "FMCP_HSPC_LA07_P"] ;# Bank  65 VCCO - VADJ_FMC - IO_L19P_T3L_N0_DBC_AD9P_65
set_property PACKAGE_PIN AG14     [get_ports "No Connect"] ;# Bank  65 VCCO - VADJ_FMC - IO_T3U_N12_65
set_property IOSTANDARD  LVCMOSxx [get_ports "No Connect"] ;# Bank  65 VCCO - VADJ_FMC - IO_T3U_N12_65
set_property PACKAGE_PIN AT9      [get_ports "No Connect"] ;# Bank  65 VCCO - VADJ_FMC - IO_T2U_N12_65
set_property IOSTANDARD  LVCMOSxx [get_ports "No Connect"] ;# Bank  65 VCCO - VADJ_FMC - IO_T2U_N12_65
set_property PACKAGE_PIN AL7      [get_ports "FMCP_HSPC_LA06_N"] ;# Bank  65 VCCO - VADJ_FMC - IO_L18N_T2U_N11_AD2N_65
set_property IOSTANDARD  LVCMOS18 [get_ports "FMCP_HSPC_LA06_N"] ;# Bank  65 VCCO - VADJ_FMC - IO_L18N_T2U_N11_AD2N_65
set_property PACKAGE_PIN AL8      [get_ports "FMCP_HSPC_LA06_P"] ;# Bank  65 VCCO - VADJ_FMC - IO_L18P_T2U_N10_AD2P_65
set_property IOSTANDARD  LVCMOS18 [get_ports "FMCP_HSPC_LA06_P"] ;# Bank  65 VCCO - VADJ_FMC - IO_L18P_T2U_N10_AD2P_65
set_property PACKAGE_PIN AM9      [get_ports "FMCP_HSPC_LA08_N"] ;# Bank  65 VCCO - VADJ_FMC - IO_L17N_T2U_N9_AD10N_65
set_property IOSTANDARD  LVCMOS18 [get_ports "FMCP_HSPC_LA08_N"] ;# Bank  65 VCCO - VADJ_FMC - IO_L17N_T2U_N9_AD10N_65
set_property PACKAGE_PIN AL9      [get_ports "FMCP_HSPC_LA08_P"] ;# Bank  65 VCCO - VADJ_FMC - IO_L17P_T2U_N8_AD10P_65
set_property IOSTANDARD  LVCMOS18 [get_ports "FMCP_HSPC_LA08_P"] ;# Bank  65 VCCO - VADJ_FMC - IO_L17P_T2U_N8_AD10P_65
set_property PACKAGE_PIN AN7      [get_ports "FMCP_HSPC_LA09_N"] ;# Bank  65 VCCO - VADJ_FMC - IO_L16N_T2U_N7_QBC_AD3N_65
set_property IOSTANDARD  LVCMOS18 [get_ports "FMCP_HSPC_LA09_N"] ;# Bank  65 VCCO - VADJ_FMC - IO_L16N_T2U_N7_QBC_AD3N_65
set_property PACKAGE_PIN AN8      [get_ports "FMCP_HSPC_LA09_P"] ;# Bank  65 VCCO - VADJ_FMC - IO_L16P_T2U_N6_QBC_AD3P_65
set_property IOSTANDARD  LVCMOS18 [get_ports "FMCP_HSPC_LA09_P"] ;# Bank  65 VCCO - VADJ_FMC - IO_L16P_T2U_N6_QBC_AD3P_65
set_property PACKAGE_PIN AM7      [get_ports "FMCP_HSPC_LA05_N"] ;# Bank  65 VCCO - VADJ_FMC - IO_L15N_T2L_N5_AD11N_65
set_property IOSTANDARD  LVCMOS18 [get_ports "FMCP_HSPC_LA05_N"] ;# Bank  65 VCCO - VADJ_FMC - IO_L15N_T2L_N5_AD11N_65
set_property PACKAGE_PIN AM8      [get_ports "FMCP_HSPC_LA05_P"] ;# Bank  65 VCCO - VADJ_FMC - IO_L15P_T2L_N4_AD11P_65
set_property IOSTANDARD  LVCMOS18 [get_ports "FMCP_HSPC_LA05_P"] ;# Bank  65 VCCO - VADJ_FMC - IO_L15P_T2L_N4_AD11P_65
set_property PACKAGE_PIN AR9      [get_ports "FMCP_HSPC_LA00_CC_N"] ;# Bank  65 VCCO - VADJ_FMC - IO_L14N_T2L_N3_GC_65
set_property IOSTANDARD  LVDS     [get_ports "FMCP_HSPC_LA00_CC_N"] ;# Bank  65 VCCO - VADJ_FMC - IO_L14N_T2L_N3_GC_65
set_property PACKAGE_PIN AP9      [get_ports "FMCP_HSPC_LA00_CC_P"] ;# Bank  65 VCCO - VADJ_FMC - IO_L14P_T2L_N2_GC_65
set_property IOSTANDARD  LVDS     [get_ports "FMCP_HSPC_LA00_CC_P"] ;# Bank  65 VCCO - VADJ_FMC - IO_L14P_T2L_N2_GC_65
set_property PACKAGE_PIN AR8      [get_ports "FMCP_HSPC_LA01_CC_N"] ;# Bank  65 VCCO - VADJ_FMC - IO_L13N_T2L_N1_GC_QBC_65
set_property IOSTANDARD  LVDS     [get_ports "FMCP_HSPC_LA01_CC_N"] ;# Bank  65 VCCO - VADJ_FMC - IO_L13N_T2L_N1_GC_QBC_65
set_property PACKAGE_PIN AP8      [get_ports "FMCP_HSPC_LA01_CC_P"] ;# Bank  65 VCCO - VADJ_FMC - IO_L13P_T2L_N0_GC_QBC_65
set_property IOSTANDARD  LVDS     [get_ports "FMCP_HSPC_LA01_CC_P"] ;# Bank  65 VCCO - VADJ_FMC - IO_L13P_T2L_N0_GC_QBC_65
set_property PACKAGE_PIN AP10     [get_ports "FMCP_HSPC_CLK0_M2C_N"] ;# Bank  65 VCCO - VADJ_FMC - IO_L12N_T1U_N11_GC_65
set_property IOSTANDARD  LVDS     [get_ports "FMCP_HSPC_CLK0_M2C_N"] ;# Bank  65 VCCO - VADJ_FMC - IO_L12N_T1U_N11_GC_65
set_property PACKAGE_PIN AN10     [get_ports "FMCP_HSPC_CLK0_M2C_P"] ;# Bank  65 VCCO - VADJ_FMC - IO_L12P_T1U_N10_GC_65
set_property IOSTANDARD  LVDS     [get_ports "FMCP_HSPC_CLK0_M2C_P"] ;# Bank  65 VCCO - VADJ_FMC - IO_L12P_T1U_N10_GC_65
set_property PACKAGE_PIN AP11     [get_ports "FMCP_HSPC_REFCLK_M2C_N"] ;# Bank  65 VCCO - VADJ_FMC - IO_L11N_T1U_N9_GC_65
set_property IOSTANDARD  LVDS     [get_ports "FMCP_HSPC_REFCLK_M2C_N"] ;# Bank  65 VCCO - VADJ_FMC - IO_L11N_T1U_N9_GC_65
set_property PACKAGE_PIN AN11     [get_ports "FMCP_HSPC_REFCLK_M2C_P"] ;# Bank  65 VCCO - VADJ_FMC - IO_L11P_T1U_N8_GC_65
set_property IOSTANDARD  LVDS     [get_ports "FMCP_HSPC_REFCLK_M2C_P"] ;# Bank  65 VCCO - VADJ_FMC - IO_L11P_T1U_N8_GC_65
set_property PACKAGE_PIN AN13     [get_ports "FMCP_HSPC_LA13_N"] ;# Bank  65 VCCO - VADJ_FMC - IO_L10N_T1U_N7_QBC_AD4N_65
set_property IOSTANDARD  LVCMOS18 [get_ports "FMCP_HSPC_LA13_N"] ;# Bank  65 VCCO - VADJ_FMC - IO_L10N_T1U_N7_QBC_AD4N_65
set_property PACKAGE_PIN AM13     [get_ports "FMCP_HSPC_LA13_P"] ;# Bank  65 VCCO - VADJ_FMC - IO_L10P_T1U_N6_QBC_AD4P_65
set_property IOSTANDARD  LVCMOS18 [get_ports "FMCP_HSPC_LA13_P"] ;# Bank  65 VCCO - VADJ_FMC - IO_L10P_T1U_N6_QBC_AD4P_65
set_property PACKAGE_PIN AM10     [get_ports "FMCP_HSPC_LA12_N"] ;# Bank  65 VCCO - VADJ_FMC - IO_L9N_T1L_N5_AD12N_65
set_property IOSTANDARD  LVCMOS18 [get_ports "FMCP_HSPC_LA12_N"] ;# Bank  65 VCCO - VADJ_FMC - IO_L9N_T1L_N5_AD12N_65
set_property PACKAGE_PIN AL10     [get_ports "FMCP_HSPC_LA12_P"] ;# Bank  65 VCCO - VADJ_FMC - IO_L9P_T1L_N4_AD12P_65
set_property IOSTANDARD  LVCMOS18 [get_ports "FMCP_HSPC_LA12_P"] ;# Bank  65 VCCO - VADJ_FMC - IO_L9P_T1L_N4_AD12P_65
set_property PACKAGE_PIN AR11     [get_ports "FMCP_HSPC_LA16_N"] ;# Bank  65 VCCO - VADJ_FMC - IO_L8N_T1L_N3_AD5N_65
set_property IOSTANDARD  LVCMOS18 [get_ports "FMCP_HSPC_LA16_N"] ;# Bank  65 VCCO - VADJ_FMC - IO_L8N_T1L_N3_AD5N_65
set_property PACKAGE_PIN AR12     [get_ports "FMCP_HSPC_LA16_P"] ;# Bank  65 VCCO - VADJ_FMC - IO_L8P_T1L_N2_AD5P_65
set_property IOSTANDARD  LVCMOS18 [get_ports "FMCP_HSPC_LA16_P"] ;# Bank  65 VCCO - VADJ_FMC - IO_L8P_T1L_N2_AD5P_65
set_property PACKAGE_PIN AN12     [get_ports "FMCP_HSPC_LA10_N"] ;# Bank  65 VCCO - VADJ_FMC - IO_L7N_T1L_N1_QBC_AD13N_65
set_property IOSTANDARD  LVCMOS18 [get_ports "FMCP_HSPC_LA10_N"] ;# Bank  65 VCCO - VADJ_FMC - IO_L7N_T1L_N1_QBC_AD13N_65
set_property PACKAGE_PIN AM12     [get_ports "FMCP_HSPC_LA10_P"] ;# Bank  65 VCCO - VADJ_FMC - IO_L7P_T1L_N0_QBC_AD13P_65
set_property IOSTANDARD  LVCMOS18 [get_ports "FMCP_HSPC_LA10_P"] ;# Bank  65 VCCO - VADJ_FMC - IO_L7P_T1L_N0_QBC_AD13P_65
set_property PACKAGE_PIN AL11     [get_ports "No Connect"] ;# Bank  65 VCCO - VADJ_FMC - IO_T1U_N12_65
set_property IOSTANDARD  LVCMOSxx [get_ports "No Connect"] ;# Bank  65 VCCO - VADJ_FMC - IO_T1U_N12_65
set_property PACKAGE_PIN AU9      [get_ports "VRP_65"] ;# Bank  65 VCCO - VADJ_FMC - IO_T0U_N12_VRP_65
set_property IOSTANDARD  LVCMOSxx [get_ports "VRP_65"] ;# Bank  65 VCCO - VADJ_FMC - IO_T0U_N12_VRP_65
set_property PACKAGE_PIN AU10     [get_ports "FMCP_HSPC_LA11_N"] ;# Bank  65 VCCO - VADJ_FMC - IO_L6N_T0U_N11_AD6N_65
set_property IOSTANDARD  LVCMOS18 [get_ports "FMCP_HSPC_LA11_N"] ;# Bank  65 VCCO - VADJ_FMC - IO_L6N_T0U_N11_AD6N_65
set_property PACKAGE_PIN AT10     [get_ports "FMCP_HSPC_LA11_P"] ;# Bank  65 VCCO - VADJ_FMC - IO_L6P_T0U_N10_AD6P_65
set_property IOSTANDARD  LVCMOS18 [get_ports "FMCP_HSPC_LA11_P"] ;# Bank  65 VCCO - VADJ_FMC - IO_L6P_T0U_N10_AD6P_65
set_property PACKAGE_PIN AW8      [get_ports "No Connect"] ;# Bank  65 VCCO - VADJ_FMC - IO_L5N_T0U_N9_AD14N_65
set_property IOSTANDARD  LVCMOSxx [get_ports "No Connect"] ;# Bank  65 VCCO - VADJ_FMC - IO_L5N_T0U_N9_AD14N_65
set_property PACKAGE_PIN AW9      [get_ports "No Connect"] ;# Bank  65 VCCO - VADJ_FMC - IO_L5P_T0U_N8_AD14P_65
set_property IOSTANDARD  LVCMOSxx [get_ports "No Connect"] ;# Bank  65 VCCO - VADJ_FMC - IO_L5P_T0U_N8_AD14P_65
set_property PACKAGE_PIN AT11     [get_ports "FMCP_HSPC_SYNC_C2M_N"] ;# Bank  65 VCCO - VADJ_FMC - IO_L4N_T0U_N7_DBC_AD7N_65
set_property IOSTANDARD  LVDS     [get_ports "FMCP_HSPC_SYNC_C2M_N"] ;# Bank  65 VCCO - VADJ_FMC - IO_L4N_T0U_N7_DBC_AD7N_65
set_property PACKAGE_PIN AT12     [get_ports "FMCP_HSPC_SYNC_C2M_P"] ;# Bank  65 VCCO - VADJ_FMC - IO_L4P_T0U_N6_DBC_AD7P_SMBALERT_65
set_property IOSTANDARD  LVDS     [get_ports "FMCP_HSPC_SYNC_C2M_P"] ;# Bank  65 VCCO - VADJ_FMC - IO_L4P_T0U_N6_DBC_AD7P_SMBALERT_65
set_property PACKAGE_PIN AW11     [get_ports "FMCP_HSPC_REFCLK_C2M_N"] ;# Bank  65 VCCO - VADJ_FMC - IO_L3N_T0L_N5_AD15N_65
set_property IOSTANDARD  LVDS     [get_ports "FMCP_HSPC_REFCLK_C2M_N"] ;# Bank  65 VCCO - VADJ_FMC - IO_L3N_T0L_N5_AD15N_65
set_property PACKAGE_PIN AV11     [get_ports "FMCP_HSPC_REFCLK_C2M_P"] ;# Bank  65 VCCO - VADJ_FMC - IO_L3P_T0L_N4_AD15P_65
set_property IOSTANDARD  LVDS     [get_ports "FMCP_HSPC_REFCLK_C2M_P"] ;# Bank  65 VCCO - VADJ_FMC - IO_L3P_T0L_N4_AD15P_65
set_property PACKAGE_PIN AV12     [get_ports "FMCP_HSPC_SYNC_M2C_N"] ;# Bank  65 VCCO - VADJ_FMC - IO_L2N_T0L_N3_65
set_property IOSTANDARD  LVDS     [get_ports "FMCP_HSPC_SYNC_M2C_N"] ;# Bank  65 VCCO - VADJ_FMC - IO_L2N_T0L_N3_65
set_property PACKAGE_PIN AU12     [get_ports "FMCP_HSPC_SYNC_M2C_P"] ;# Bank  65 VCCO - VADJ_FMC - IO_L2P_T0L_N2_65
set_property IOSTANDARD  LVDS     [get_ports "FMCP_HSPC_SYNC_M2C_P"] ;# Bank  65 VCCO - VADJ_FMC - IO_L2P_T0L_N2_65
set_property PACKAGE_PIN AW10     [get_ports "No Connect"] ;# Bank  65 VCCO - VADJ_FMC - IO_L1N_T0L_N1_DBC_65
set_property IOSTANDARD  LVCMOSxx [get_ports "No Connect"] ;# Bank  65 VCCO - VADJ_FMC - IO_L1N_T0L_N1_DBC_65
set_property PACKAGE_PIN AV10     [get_ports "No Connect"] ;# Bank  65 VCCO - VADJ_FMC - IO_L1P_T0L_N0_DBC_65
set_property IOSTANDARD  LVCMOSxx [get_ports "No Connect"] ;# Bank  65 VCCO - VADJ_FMC - IO_L1P_T0L_N0_DBC_65
#Other net   PACKAGE_PIN AG13     - N19275201                 Bank  65 - VREF_65
set_property PACKAGE_PIN AF15     [get_ports "CPU_RESET"] ;# Bank  64 VCCO - VCC1V8   - IO_T2U_N12_64
set_property IOSTANDARD  LVCMOS18 [get_ports "CPU_RESET"] ;# Bank  64 VCCO - VCC1V8   - IO_T2U_N12_64
set_property PACKAGE_PIN AF16     [get_ports "GPIO_DIP_SW0"] ;# Bank  64 VCCO - VCC1V8   - IO_L18N_T2U_N11_AD2N_64
set_property IOSTANDARD  LVCMOS18 [get_ports "GPIO_DIP_SW0"] ;# Bank  64 VCCO - VCC1V8   - IO_L18N_T2U_N11_AD2N_64
set_property PACKAGE_PIN AF17     [get_ports "GPIO_DIP_SW1"] ;# Bank  64 VCCO - VCC1V8   - IO_L18P_T2U_N10_AD2P_64
set_property IOSTANDARD  LVCMOS18 [get_ports "GPIO_DIP_SW1"] ;# Bank  64 VCCO - VCC1V8   - IO_L18P_T2U_N10_AD2P_64
set_property PACKAGE_PIN AH15     [get_ports "GPIO_DIP_SW2"] ;# Bank  64 VCCO - VCC1V8   - IO_L17N_T2U_N9_AD10N_64
set_property IOSTANDARD  LVCMOS18 [get_ports "GPIO_DIP_SW2"] ;# Bank  64 VCCO - VCC1V8   - IO_L17N_T2U_N9_AD10N_64
set_property PACKAGE_PIN AH16     [get_ports "GPIO_DIP_SW3"] ;# Bank  64 VCCO - VCC1V8   - IO_L17P_T2U_N8_AD10P_64
set_property IOSTANDARD  LVCMOS18 [get_ports "GPIO_DIP_SW3"] ;# Bank  64 VCCO - VCC1V8   - IO_L17P_T2U_N8_AD10P_64
set_property PACKAGE_PIN AH17     [get_ports "GPIO_DIP_SW4"] ;# Bank  64 VCCO - VCC1V8   - IO_L16N_T2U_N7_QBC_AD3N_64
set_property IOSTANDARD  LVCMOS18 [get_ports "GPIO_DIP_SW4"] ;# Bank  64 VCCO - VCC1V8   - IO_L16N_T2U_N7_QBC_AD3N_64
set_property PACKAGE_PIN AG17     [get_ports "GPIO_DIP_SW5"] ;# Bank  64 VCCO - VCC1V8   - IO_L16P_T2U_N6_QBC_AD3P_64
set_property IOSTANDARD  LVCMOS18 [get_ports "GPIO_DIP_SW5"] ;# Bank  64 VCCO - VCC1V8   - IO_L16P_T2U_N6_QBC_AD3P_64
set_property PACKAGE_PIN AJ15     [get_ports "GPIO_DIP_SW6"] ;# Bank  64 VCCO - VCC1V8   - IO_L15N_T2L_N5_AD11N_64
set_property IOSTANDARD  LVCMOS18 [get_ports "GPIO_DIP_SW6"] ;# Bank  64 VCCO - VCC1V8   - IO_L15N_T2L_N5_AD11N_64
set_property PACKAGE_PIN AJ16     [get_ports "GPIO_DIP_SW7"] ;# Bank  64 VCCO - VCC1V8   - IO_L15P_T2L_N4_AD11P_64
set_property IOSTANDARD  LVCMOS18 [get_ports "GPIO_DIP_SW7"] ;# Bank  64 VCCO - VCC1V8   - IO_L15P_T2L_N4_AD11P_64
set_property PACKAGE_PIN AK16     [get_ports "SYSREF_FPGA_C_N"] ;# Bank  64 VCCO - VCC1V8   - IO_L14N_T2L_N3_GC_64
set_property IOSTANDARD  LVDS     [get_ports "SYSREF_FPGA_C_N"] ;# Bank  64 VCCO - VCC1V8   - IO_L14N_T2L_N3_GC_64
set_property PACKAGE_PIN AK17     [get_ports "SYSREF_FPGA_C_P"] ;# Bank  64 VCCO - VCC1V8   - IO_L14P_T2L_N2_GC_64
set_property IOSTANDARD  LVDS     [get_ports "SYSREF_FPGA_C_P"] ;# Bank  64 VCCO - VCC1V8   - IO_L14P_T2L_N2_GC_64
set_property PACKAGE_PIN AL15     [get_ports "FPGA_REFCLK_OUT_C_N"] ;# Bank  64 VCCO - VCC1V8   - IO_L13N_T2L_N1_GC_QBC_64
set_property IOSTANDARD  LVDS     [get_ports "FPGA_REFCLK_OUT_C_N"] ;# Bank  64 VCCO - VCC1V8   - IO_L13N_T2L_N1_GC_QBC_64
set_property PACKAGE_PIN AL16     [get_ports "FPGA_REFCLK_OUT_C_P"] ;# Bank  64 VCCO - VCC1V8   - IO_L13P_T2L_N0_GC_QBC_64
set_property IOSTANDARD  LVDS     [get_ports "FPGA_REFCLK_OUT_C_P"] ;# Bank  64 VCCO - VCC1V8   - IO_L13P_T2L_N0_GC_QBC_64
set_property PACKAGE_PIN AM17     [get_ports "CLK_125_N"] ;# Bank  64 VCCO - VCC1V8   - IO_L12N_T1U_N11_GC_64
set_property IOSTANDARD  LVDS     [get_ports "CLK_125_N"] ;# Bank  64 VCCO - VCC1V8   - IO_L12N_T1U_N11_GC_64
set_property PACKAGE_PIN AL17     [get_ports "CLK_125_P"] ;# Bank  64 VCCO - VCC1V8   - IO_L12P_T1U_N10_GC_64
set_property IOSTANDARD  LVDS     [get_ports "CLK_125_P"] ;# Bank  64 VCCO - VCC1V8   - IO_L12P_T1U_N10_GC_64
set_property PACKAGE_PIN AN15     [get_ports "CLK_100_N"] ;# Bank  64 VCCO - VCC1V8   - IO_L11N_T1U_N9_GC_64
set_property IOSTANDARD  LVDS     [get_ports "CLK_100_N"] ;# Bank  64 VCCO - VCC1V8   - IO_L11N_T1U_N9_GC_64
set_property PACKAGE_PIN AM15     [get_ports "CLK_100_P"] ;# Bank  64 VCCO - VCC1V8   - IO_L11P_T1U_N8_GC_64
set_property IOSTANDARD  LVDS     [get_ports "CLK_100_P"] ;# Bank  64 VCCO - VCC1V8   - IO_L11P_T1U_N8_GC_64
set_property PACKAGE_PIN AN16     [get_ports "GPIO_LED_5_LS"] ;# Bank  64 VCCO - VCC1V8   - IO_L10N_T1U_N7_QBC_AD4N_64
set_property IOSTANDARD  LVCMOS18 [get_ports "GPIO_LED_5_LS"] ;# Bank  64 VCCO - VCC1V8   - IO_L10N_T1U_N7_QBC_AD4N_64
set_property PACKAGE_PIN AN17     [get_ports "GPIO_LED_6_LS"] ;# Bank  64 VCCO - VCC1V8   - IO_L10P_T1U_N6_QBC_AD4P_64
set_property IOSTANDARD  LVCMOS18 [get_ports "GPIO_LED_6_LS"] ;# Bank  64 VCCO - VCC1V8   - IO_L10P_T1U_N6_QBC_AD4P_64
set_property PACKAGE_PIN AR14     [get_ports "No Connect"] ;# Bank  64 VCCO - VCC1V8   - IO_L9N_T1L_N5_AD12N_64
set_property IOSTANDARD  LVCMOSxx [get_ports "No Connect"] ;# Bank  64 VCCO - VCC1V8   - IO_L9N_T1L_N5_AD12N_64
set_property PACKAGE_PIN AP14     [get_ports "AMS_FPGA_REF_CLK"] ;# Bank  64 VCCO - VCC1V8   - IO_L9P_T1L_N4_AD12P_64
set_property IOSTANDARD  LVCMOS18 [get_ports "AMS_FPGA_REF_CLK"] ;# Bank  64 VCCO - VCC1V8   - IO_L9P_T1L_N4_AD12P_64
set_property PACKAGE_PIN AR16     [get_ports "GPIO_LED_2_LS"] ;# Bank  64 VCCO - VCC1V8   - IO_L8N_T1L_N3_AD5N_64
set_property IOSTANDARD  LVCMOS18 [get_ports "GPIO_LED_2_LS"] ;# Bank  64 VCCO - VCC1V8   - IO_L8N_T1L_N3_AD5N_64
set_property PACKAGE_PIN AP16     [get_ports "GPIO_LED_3_LS"] ;# Bank  64 VCCO - VCC1V8   - IO_L8P_T1L_N2_AD5P_64
set_property IOSTANDARD  LVCMOS18 [get_ports "GPIO_LED_3_LS"] ;# Bank  64 VCCO - VCC1V8   - IO_L8P_T1L_N2_AD5P_64
set_property PACKAGE_PIN AR13     [get_ports "GPIO_LED_0_LS"] ;# Bank  64 VCCO - VCC1V8   - IO_L7N_T1L_N1_QBC_AD13N_64
set_property IOSTANDARD  LVCMOS18 [get_ports "GPIO_LED_0_LS"] ;# Bank  64 VCCO - VCC1V8   - IO_L7N_T1L_N1_QBC_AD13N_64
set_property PACKAGE_PIN AP13     [get_ports "GPIO_LED_1_LS"] ;# Bank  64 VCCO - VCC1V8   - IO_L7P_T1L_N0_QBC_AD13P_64
set_property IOSTANDARD  LVCMOS18 [get_ports "GPIO_LED_1_LS"] ;# Bank  64 VCCO - VCC1V8   - IO_L7P_T1L_N0_QBC_AD13P_64
set_property PACKAGE_PIN AP15     [get_ports "GPIO_LED_4_LS"] ;# Bank  64 VCCO - VCC1V8   - IO_T1U_N12_64
set_property IOSTANDARD  LVCMOS18 [get_ports "GPIO_LED_4_LS"] ;# Bank  64 VCCO - VCC1V8   - IO_T1U_N12_64
set_property PACKAGE_PIN AT14     [get_ports "UART2_CTS_B"] ;# Bank  64 VCCO - VCC1V8   - IO_T0U_N12_VRP_64
set_property IOSTANDARD  LVCMOS18 [get_ports "UART2_CTS_B"] ;# Bank  64 VCCO - VCC1V8   - IO_T0U_N12_VRP_64
set_property PACKAGE_PIN AU14     [get_ports "UART2_RTS_B"] ;# Bank  64 VCCO - VCC1V8   - IO_L6N_T0U_N11_AD6N_64
set_property IOSTANDARD  LVCMOS18 [get_ports "UART2_RTS_B"] ;# Bank  64 VCCO - VCC1V8   - IO_L6N_T0U_N11_AD6N_64
set_property PACKAGE_PIN AU15     [get_ports "UART2_RXD_FPGA_TXD"] ;# Bank  64 VCCO - VCC1V8   - IO_L6P_T0U_N10_AD6P_64
set_property IOSTANDARD  LVCMOS18 [get_ports "UART2_RXD_FPGA_TXD"] ;# Bank  64 VCCO - VCC1V8   - IO_L6P_T0U_N10_AD6P_64
set_property PACKAGE_PIN AT15     [get_ports "UART2_TXD_FPGA_RXD"] ;# Bank  64 VCCO - VCC1V8   - IO_L5N_T0U_N9_AD14N_64
set_property IOSTANDARD  LVCMOS18 [get_ports "UART2_TXD_FPGA_RXD"] ;# Bank  64 VCCO - VCC1V8   - IO_L5N_T0U_N9_AD14N_64
set_property PACKAGE_PIN AT16     [get_ports "PL_I2C0_SCL_LS"] ;# Bank  64 VCCO - VCC1V8   - IO_L5P_T0U_N8_AD14P_64
set_property IOSTANDARD  LVCMOS18 [get_ports "PL_I2C0_SCL_LS"] ;# Bank  64 VCCO - VCC1V8   - IO_L5P_T0U_N8_AD14P_64
set_property PACKAGE_PIN AW16     [get_ports "PL_I2C0_SDA_LS"] ;# Bank  64 VCCO - VCC1V8   - IO_L4N_T0U_N7_DBC_AD7N_64
set_property IOSTANDARD  LVCMOS18 [get_ports "PL_I2C0_SDA_LS"] ;# Bank  64 VCCO - VCC1V8   - IO_L4N_T0U_N7_DBC_AD7N_64
set_property PACKAGE_PIN AV16     [get_ports "PL_I2C1_SCL_LS"] ;# Bank  64 VCCO - VCC1V8   - IO_L4P_T0U_N6_DBC_AD7P_64
set_property IOSTANDARD  LVCMOS18 [get_ports "PL_I2C1_SCL_LS"] ;# Bank  64 VCCO - VCC1V8   - IO_L4P_T0U_N6_DBC_AD7P_64
set_property PACKAGE_PIN AV13     [get_ports "PL_I2C1_SDA_LS"] ;# Bank  64 VCCO - VCC1V8   - IO_L3N_T0L_N5_AD15N_64
set_property IOSTANDARD  LVCMOS18 [get_ports "PL_I2C1_SDA_LS"] ;# Bank  64 VCCO - VCC1V8   - IO_L3N_T0L_N5_AD15N_64
set_property PACKAGE_PIN AU13     [get_ports "MSP430_UCA1_TXD_LS"] ;# Bank  64 VCCO - VCC1V8   - IO_L3P_T0L_N4_AD15P_64
set_property IOSTANDARD  LVCMOS18 [get_ports "MSP430_UCA1_TXD_LS"] ;# Bank  64 VCCO - VCC1V8   - IO_L3P_T0L_N4_AD15P_64
set_property PACKAGE_PIN AW15     [get_ports "MSP430_UCA1_RXD_LS"] ;# Bank  64 VCCO - VCC1V8   - IO_L2N_T0L_N3_64
set_property IOSTANDARD  LVCMOS18 [get_ports "MSP430_UCA1_RXD_LS"] ;# Bank  64 VCCO - VCC1V8   - IO_L2N_T0L_N3_64
set_property PACKAGE_PIN AV15     [get_ports "GPIO_LED_7_LS"] ;# Bank  64 VCCO - VCC1V8   - IO_L2P_T0L_N2_64
set_property IOSTANDARD  LVCMOS18 [get_ports "GPIO_LED_7_LS"] ;# Bank  64 VCCO - VCC1V8   - IO_L2P_T0L_N2_64
set_property PACKAGE_PIN AW13     [get_ports "SFP_REC_CLOCK_C_N"] ;# Bank  64 VCCO - VCC1V8   - IO_L1N_T0L_N1_DBC_64
set_property IOSTANDARD  LVDS     [get_ports "SFP_REC_CLOCK_C_N"] ;# Bank  64 VCCO - VCC1V8   - IO_L1N_T0L_N1_DBC_64
set_property PACKAGE_PIN AW14     [get_ports "SFP_REC_CLOCK_C_P"] ;# Bank  64 VCCO - VCC1V8   - IO_L1P_T0L_N0_DBC_64
set_property IOSTANDARD  LVDS     [get_ports "SFP_REC_CLOCK_C_P"] ;# Bank  64 VCCO - VCC1V8   - IO_L1P_T0L_N0_DBC_64
#Other net   PACKAGE_PIN AG15     - No Connect                Bank  64 - VREF_64
set_property PACKAGE_PIN A9       [get_ports "DACIO_00"] ;# Bank  87 VCCO - VCC1V8   - IO_L12N_AD8N_87
set_property IOSTANDARD  LVCMOS18 [get_ports "DACIO_00"] ;# Bank  87 VCCO - VCC1V8   - IO_L12N_AD8N_87
set_property PACKAGE_PIN A10      [get_ports "DACIO_01"] ;# Bank  87 VCCO - VCC1V8   - IO_L12P_AD8P_87
set_property IOSTANDARD  LVCMOS18 [get_ports "DACIO_01"] ;# Bank  87 VCCO - VCC1V8   - IO_L12P_AD8P_87
set_property PACKAGE_PIN A6       [get_ports "DACIO_02"] ;# Bank  87 VCCO - VCC1V8   - IO_L11N_AD9N_87
set_property IOSTANDARD  LVCMOS18 [get_ports "DACIO_02"] ;# Bank  87 VCCO - VCC1V8   - IO_L11N_AD9N_87
set_property PACKAGE_PIN A7       [get_ports "DACIO_03"] ;# Bank  87 VCCO - VCC1V8   - IO_L11P_AD9P_87
set_property IOSTANDARD  LVCMOS18 [get_ports "DACIO_03"] ;# Bank  87 VCCO - VCC1V8   - IO_L11P_AD9P_87
set_property PACKAGE_PIN A5       [get_ports "DACIO_04"] ;# Bank  87 VCCO - VCC1V8   - IO_L10N_AD10N_87
set_property IOSTANDARD  LVCMOS18 [get_ports "DACIO_04"] ;# Bank  87 VCCO - VCC1V8   - IO_L10N_AD10N_87
set_property PACKAGE_PIN B5       [get_ports "DACIO_05"] ;# Bank  87 VCCO - VCC1V8   - IO_L10P_AD10P_87
set_property IOSTANDARD  LVCMOS18 [get_ports "DACIO_05"] ;# Bank  87 VCCO - VCC1V8   - IO_L10P_AD10P_87
set_property PACKAGE_PIN C5       [get_ports "DACIO_06"] ;# Bank  87 VCCO - VCC1V8   - IO_L9N_AD11N_87
set_property IOSTANDARD  LVCMOS18 [get_ports "DACIO_06"] ;# Bank  87 VCCO - VCC1V8   - IO_L9N_AD11N_87
set_property PACKAGE_PIN C6       [get_ports "DACIO_07"] ;# Bank  87 VCCO - VCC1V8   - IO_L9P_AD11P_87
set_property IOSTANDARD  LVCMOS18 [get_ports "DACIO_07"] ;# Bank  87 VCCO - VCC1V8   - IO_L9P_AD11P_87
set_property PACKAGE_PIN B9       [get_ports "DACIO_08"] ;# Bank  87 VCCO - VCC1V8   - IO_L8N_HDGC_87
set_property IOSTANDARD  LVCMOS18 [get_ports "DACIO_08"] ;# Bank  87 VCCO - VCC1V8   - IO_L8N_HDGC_87
set_property PACKAGE_PIN B10      [get_ports "DACIO_09"] ;# Bank  87 VCCO - VCC1V8   - IO_L8P_HDGC_87
set_property IOSTANDARD  LVCMOS18 [get_ports "DACIO_09"] ;# Bank  87 VCCO - VCC1V8   - IO_L8P_HDGC_87
set_property PACKAGE_PIN B7       [get_ports "DACIO_10"] ;# Bank  87 VCCO - VCC1V8   - IO_L7N_HDGC_87
set_property IOSTANDARD  LVCMOS18 [get_ports "DACIO_10"] ;# Bank  87 VCCO - VCC1V8   - IO_L7N_HDGC_87
set_property PACKAGE_PIN B8       [get_ports "DACIO_11"] ;# Bank  87 VCCO - VCC1V8   - IO_L7P_HDGC_87
set_property IOSTANDARD  LVCMOS18 [get_ports "DACIO_11"] ;# Bank  87 VCCO - VCC1V8   - IO_L7P_HDGC_87
set_property PACKAGE_PIN D8       [get_ports "DACIO_12"] ;# Bank  87 VCCO - VCC1V8   - IO_L6N_HDGC_87
set_property IOSTANDARD  LVCMOS18 [get_ports "DACIO_12"] ;# Bank  87 VCCO - VCC1V8   - IO_L6N_HDGC_87
set_property PACKAGE_PIN D9       [get_ports "DACIO_13"] ;# Bank  87 VCCO - VCC1V8   - IO_L6P_HDGC_87
set_property IOSTANDARD  LVCMOS18 [get_ports "DACIO_13"] ;# Bank  87 VCCO - VCC1V8   - IO_L6P_HDGC_87
set_property PACKAGE_PIN C7       [get_ports "DACIO_14"] ;# Bank  87 VCCO - VCC1V8   - IO_L5N_HDGC_87
set_property IOSTANDARD  LVCMOS18 [get_ports "DACIO_14"] ;# Bank  87 VCCO - VCC1V8   - IO_L5N_HDGC_87
set_property PACKAGE_PIN C8       [get_ports "DACIO_15"] ;# Bank  87 VCCO - VCC1V8   - IO_L5P_HDGC_87
set_property IOSTANDARD  LVCMOS18 [get_ports "DACIO_15"] ;# Bank  87 VCCO - VCC1V8   - IO_L5P_HDGC_87
set_property PACKAGE_PIN C10      [get_ports "DACIO_16"] ;# Bank  87 VCCO - VCC1V8   - IO_L4N_AD12N_87
set_property IOSTANDARD  LVCMOS18 [get_ports "DACIO_16"] ;# Bank  87 VCCO - VCC1V8   - IO_L4N_AD12N_87
set_property PACKAGE_PIN D10      [get_ports "DACIO_17"] ;# Bank  87 VCCO - VCC1V8   - IO_L4P_AD12P_87
set_property IOSTANDARD  LVCMOS18 [get_ports "DACIO_17"] ;# Bank  87 VCCO - VCC1V8   - IO_L4P_AD12P_87
set_property PACKAGE_PIN D6       [get_ports "DACIO_18"] ;# Bank  87 VCCO - VCC1V8   - IO_L3N_AD13N_87
set_property IOSTANDARD  LVCMOS18 [get_ports "DACIO_18"] ;# Bank  87 VCCO - VCC1V8   - IO_L3N_AD13N_87
set_property PACKAGE_PIN E7       [get_ports "DACIO_19"] ;# Bank  87 VCCO - VCC1V8   - IO_L3P_AD13P_87
set_property IOSTANDARD  LVCMOS18 [get_ports "DACIO_19"] ;# Bank  87 VCCO - VCC1V8   - IO_L3P_AD13P_87
set_property PACKAGE_PIN E8       [get_ports "GPIO_SW_S"] ;# Bank  87 VCCO - VCC1V8   - IO_L2N_AD14N_87
set_property IOSTANDARD  LVCMOS18 [get_ports "GPIO_SW_S"] ;# Bank  87 VCCO - VCC1V8   - IO_L2N_AD14N_87
set_property PACKAGE_PIN E9       [get_ports "SFP_SI5382_CLK_IN_SEL"] ;# Bank  87 VCCO - VCC1V8   - IO_L2P_AD14P_87
set_property IOSTANDARD  LVCMOS18 [get_ports "SFP_SI5382_CLK_IN_SEL"] ;# Bank  87 VCCO - VCC1V8   - IO_L2P_AD14P_87
set_property PACKAGE_PIN E6       [get_ports "No Connect"] ;# Bank  87 VCCO - VCC1V8   - IO_L1N_AD15N_87
set_property IOSTANDARD  LVCMOSxx [get_ports "No Connect"] ;# Bank  87 VCCO - VCC1V8   - IO_L1N_AD15N_87
set_property PACKAGE_PIN F6       [get_ports "No Connect"] ;# Bank  87 VCCO - VCC1V8   - IO_L1P_AD15P_87
set_property IOSTANDARD  LVCMOSxx [get_ports "No Connect"] ;# Bank  87 VCCO - VCC1V8   - IO_L1P_AD15P_87
set_property PACKAGE_PIN AP5      [get_ports "ADCIO_00"] ;# Bank  84 VCCO - VCC1V8   - IO_L12N_AD0N_84
set_property IOSTANDARD  LVCMOS18 [get_ports "ADCIO_00"] ;# Bank  84 VCCO - VCC1V8   - IO_L12N_AD0N_84
set_property PACKAGE_PIN AP6      [get_ports "ADCIO_01"] ;# Bank  84 VCCO - VCC1V8   - IO_L12P_AD0P_84
set_property IOSTANDARD  LVCMOS18 [get_ports "ADCIO_01"] ;# Bank  84 VCCO - VCC1V8   - IO_L12P_AD0P_84
set_property PACKAGE_PIN AR6      [get_ports "ADCIO_02"] ;# Bank  84 VCCO - VCC1V8   - IO_L11N_AD1N_84
set_property IOSTANDARD  LVCMOS18 [get_ports "ADCIO_02"] ;# Bank  84 VCCO - VCC1V8   - IO_L11N_AD1N_84
set_property PACKAGE_PIN AR7      [get_ports "ADCIO_03"] ;# Bank  84 VCCO - VCC1V8   - IO_L11P_AD1P_84
set_property IOSTANDARD  LVCMOS18 [get_ports "ADCIO_03"] ;# Bank  84 VCCO - VCC1V8   - IO_L11P_AD1P_84
set_property PACKAGE_PIN AV7      [get_ports "ADCIO_04"] ;# Bank  84 VCCO - VCC1V8   - IO_L10N_AD2N_84
set_property IOSTANDARD  LVCMOS18 [get_ports "ADCIO_04"] ;# Bank  84 VCCO - VCC1V8   - IO_L10N_AD2N_84
set_property PACKAGE_PIN AU7      [get_ports "ADCIO_05"] ;# Bank  84 VCCO - VCC1V8   - IO_L10P_AD2P_84
set_property IOSTANDARD  LVCMOS18 [get_ports "ADCIO_05"] ;# Bank  84 VCCO - VCC1V8   - IO_L10P_AD2P_84
set_property PACKAGE_PIN AV8      [get_ports "ADCIO_06"] ;# Bank  84 VCCO - VCC1V8   - IO_L9N_AD3N_84
set_property IOSTANDARD  LVCMOS18 [get_ports "ADCIO_06"] ;# Bank  84 VCCO - VCC1V8   - IO_L9N_AD3N_84
set_property PACKAGE_PIN AU8      [get_ports "ADCIO_07"] ;# Bank  84 VCCO - VCC1V8   - IO_L9P_AD3P_84
set_property IOSTANDARD  LVCMOS18 [get_ports "ADCIO_07"] ;# Bank  84 VCCO - VCC1V8   - IO_L9P_AD3P_84
set_property PACKAGE_PIN AT6      [get_ports "ADCIO_08"] ;# Bank  84 VCCO - VCC1V8   - IO_L8N_HDGC_AD4N_84
set_property IOSTANDARD  LVCMOS18 [get_ports "ADCIO_08"] ;# Bank  84 VCCO - VCC1V8   - IO_L8N_HDGC_AD4N_84
set_property PACKAGE_PIN AT7      [get_ports "ADCIO_09"] ;# Bank  84 VCCO - VCC1V8   - IO_L8P_HDGC_AD4P_84
set_property IOSTANDARD  LVCMOS18 [get_ports "ADCIO_09"] ;# Bank  84 VCCO - VCC1V8   - IO_L8P_HDGC_AD4P_84
set_property PACKAGE_PIN AU5      [get_ports "ADCIO_10"] ;# Bank  84 VCCO - VCC1V8   - IO_L7N_HDGC_AD5N_84
set_property IOSTANDARD  LVCMOS18 [get_ports "ADCIO_10"] ;# Bank  84 VCCO - VCC1V8   - IO_L7N_HDGC_AD5N_84
set_property PACKAGE_PIN AT5      [get_ports "ADCIO_11"] ;# Bank  84 VCCO - VCC1V8   - IO_L7P_HDGC_AD5P_84
set_property IOSTANDARD  LVCMOS18 [get_ports "ADCIO_11"] ;# Bank  84 VCCO - VCC1V8   - IO_L7P_HDGC_AD5P_84
set_property PACKAGE_PIN AU3      [get_ports "ADCIO_12"] ;# Bank  84 VCCO - VCC1V8   - IO_L6N_HDGC_AD6N_84
set_property IOSTANDARD  LVCMOS18 [get_ports "ADCIO_12"] ;# Bank  84 VCCO - VCC1V8   - IO_L6N_HDGC_AD6N_84
set_property PACKAGE_PIN AU4      [get_ports "ADCIO_13"] ;# Bank  84 VCCO - VCC1V8   - IO_L6P_HDGC_AD6P_84
set_property IOSTANDARD  LVCMOS18 [get_ports "ADCIO_13"] ;# Bank  84 VCCO - VCC1V8   - IO_L6P_HDGC_AD6P_84
set_property PACKAGE_PIN AV5      [get_ports "ADCIO_14"] ;# Bank  84 VCCO - VCC1V8   - IO_L5N_HDGC_AD7N_84
set_property IOSTANDARD  LVCMOS18 [get_ports "ADCIO_14"] ;# Bank  84 VCCO - VCC1V8   - IO_L5N_HDGC_AD7N_84
set_property PACKAGE_PIN AV6      [get_ports "ADCIO_15"] ;# Bank  84 VCCO - VCC1V8   - IO_L5P_HDGC_AD7P_84
set_property IOSTANDARD  LVCMOS18 [get_ports "ADCIO_15"] ;# Bank  84 VCCO - VCC1V8   - IO_L5P_HDGC_AD7P_84
set_property PACKAGE_PIN AU1      [get_ports "ADCIO_16"] ;# Bank  84 VCCO - VCC1V8   - IO_L4N_AD8N_84
set_property IOSTANDARD  LVCMOS18 [get_ports "ADCIO_16"] ;# Bank  84 VCCO - VCC1V8   - IO_L4N_AD8N_84
set_property PACKAGE_PIN AU2      [get_ports "ADCIO_17"] ;# Bank  84 VCCO - VCC1V8   - IO_L4P_AD8P_84
set_property IOSTANDARD  LVCMOS18 [get_ports "ADCIO_17"] ;# Bank  84 VCCO - VCC1V8   - IO_L4P_AD8P_84
set_property PACKAGE_PIN AV2      [get_ports "ADCIO_18"] ;# Bank  84 VCCO - VCC1V8   - IO_L3N_AD9N_84
set_property IOSTANDARD  LVCMOS18 [get_ports "ADCIO_18"] ;# Bank  84 VCCO - VCC1V8   - IO_L3N_AD9N_84
set_property PACKAGE_PIN AV3      [get_ports "ADCIO_19"] ;# Bank  84 VCCO - VCC1V8   - IO_L3P_AD9P_84
set_property IOSTANDARD  LVCMOS18 [get_ports "ADCIO_19"] ;# Bank  84 VCCO - VCC1V8   - IO_L3P_AD9P_84
set_property PACKAGE_PIN AW3      [get_ports "GPIO_SW_N"] ;# Bank  84 VCCO - VCC1V8   - IO_L2N_AD10N_84
set_property IOSTANDARD  LVCMOS18 [get_ports "GPIO_SW_N"] ;# Bank  84 VCCO - VCC1V8   - IO_L2N_AD10N_84
set_property PACKAGE_PIN AW4      [get_ports "GPIO_SW_E"] ;# Bank  84 VCCO - VCC1V8   - IO_L2P_AD10P_84
set_property IOSTANDARD  LVCMOS18 [get_ports "GPIO_SW_E"] ;# Bank  84 VCCO - VCC1V8   - IO_L2P_AD10P_84
set_property PACKAGE_PIN AW5      [get_ports "GPIO_SW_C"] ;# Bank  84 VCCO - VCC1V8   - IO_L1N_AD11N_84
set_property IOSTANDARD  LVCMOS18 [get_ports "GPIO_SW_C"] ;# Bank  84 VCCO - VCC1V8   - IO_L1N_AD11N_84
set_property PACKAGE_PIN AW6      [get_ports "GPIO_SW_W"] ;# Bank  84 VCCO - VCC1V8   - IO_L1P_AD11P_84
set_property IOSTANDARD  LVCMOS18 [get_ports "GPIO_SW_W"] ;# Bank  84 VCCO - VCC1V8   - IO_L1P_AD11P_84
set_property PACKAGE_PIN AA34     [get_ports "SFP_SI5382_IN1_C_N"] ;# Bank 128 - MGTREFCLK0N_128
set_property PACKAGE_PIN AA33     [get_ports "SFP_SI5382_IN1_C_P"] ;# Bank 128 - MGTREFCLK0P_128
set_property PACKAGE_PIN Y32      [get_ports "SFP_SI5382_OUT_C_N"] ;# Bank 128 - MGTREFCLK1N_128
set_property PACKAGE_PIN Y31      [get_ports "SFP_SI5382_OUT_C_P"] ;# Bank 128 - MGTREFCLK1P_128
set_property PACKAGE_PIN AA39     [get_ports "SFP0_RX_N"] ;# Bank 128 - MGTYRXN0_128
set_property PACKAGE_PIN W39      [get_ports "SFP1_RX_N"] ;# Bank 128 - MGTYRXN1_128
set_property PACKAGE_PIN U39      [get_ports "SFP2_RX_N"] ;# Bank 128 - MGTYRXN2_128
set_property PACKAGE_PIN R39      [get_ports "SFP3_RX_N"] ;# Bank 128 - MGTYRXN3_128
set_property PACKAGE_PIN AA38     [get_ports "SFP0_RX_P"] ;# Bank 128 - MGTYRXP0_128
set_property PACKAGE_PIN W38      [get_ports "SFP1_RX_P"] ;# Bank 128 - MGTYRXP1_128
set_property PACKAGE_PIN U38      [get_ports "SFP2_RX_P"] ;# Bank 128 - MGTYRXP2_128
set_property PACKAGE_PIN R38      [get_ports "SFP3_RX_P"] ;# Bank 128 - MGTYRXP3_128
set_property PACKAGE_PIN Y36      [get_ports "SFP0_TX_N"] ;# Bank 128 - MGTYTXN0_128
set_property PACKAGE_PIN V36      [get_ports "SFP1_TX_N"] ;# Bank 128 - MGTYTXN1_128
set_property PACKAGE_PIN T36      [get_ports "SFP2_TX_N"] ;# Bank 128 - MGTYTXN2_128
set_property PACKAGE_PIN R34      [get_ports "SFP3_TX_N"] ;# Bank 128 - MGTYTXN3_128
set_property PACKAGE_PIN Y35      [get_ports "SFP0_TX_P"] ;# Bank 128 - MGTYTXP0_128
set_property PACKAGE_PIN V35      [get_ports "SFP1_TX_P"] ;# Bank 128 - MGTYTXP1_128
set_property PACKAGE_PIN T35      [get_ports "SFP2_TX_P"] ;# Bank 128 - MGTYTXP2_128
set_property PACKAGE_PIN R33      [get_ports "SFP3_TX_P"] ;# Bank 128 - MGTYTXP3_128
#Other net   PACKAGE_PIN K32      - MGT1V2                    Bank 129 - MGTAVTTRCAL_L
set_property PACKAGE_PIN W34      [get_ports "FMCP_HSPC_GBTCLK0_M2C_C_N"] ;# Bank 129 - MGTREFCLK0N_129
set_property PACKAGE_PIN W33      [get_ports "FMCP_HSPC_GBTCLK0_M2C_C_P"] ;# Bank 129 - MGTREFCLK0P_129
set_property PACKAGE_PIN V32      [get_ports "USER_MGT_SI570_CLOCK_C_N"] ;# Bank 129 - MGTREFCLK1N_129
set_property PACKAGE_PIN V31      [get_ports "USER_MGT_SI570_CLOCK_C_P"] ;# Bank 129 - MGTREFCLK1P_129
set_property PACKAGE_PIN K31      [get_ports "N17453038"] ;# Bank 129 - MGTRREF_L
set_property PACKAGE_PIN N39      [get_ports "FMCP_HSPC_DP0_M2C_N"] ;# Bank 129 - MGTYRXN0_129
set_property PACKAGE_PIN M37      [get_ports "FMCP_HSPC_DP1_M2C_N"] ;# Bank 129 - MGTYRXN1_129
set_property PACKAGE_PIN L39      [get_ports "FMCP_HSPC_DP2_M2C_N"] ;# Bank 129 - MGTYRXN2_129
set_property PACKAGE_PIN K37      [get_ports "FMCP_HSPC_DP3_M2C_N"] ;# Bank 129 - MGTYRXN3_129
set_property PACKAGE_PIN N38      [get_ports "FMCP_HSPC_DP0_M2C_P"] ;# Bank 129 - MGTYRXP0_129
set_property PACKAGE_PIN M36      [get_ports "FMCP_HSPC_DP1_M2C_P"] ;# Bank 129 - MGTYRXP1_129
set_property PACKAGE_PIN L38      [get_ports "FMCP_HSPC_DP2_M2C_P"] ;# Bank 129 - MGTYRXP2_129
set_property PACKAGE_PIN K36      [get_ports "FMCP_HSPC_DP3_M2C_P"] ;# Bank 129 - MGTYRXP3_129
set_property PACKAGE_PIN P36      [get_ports "FMCP_HSPC_DP0_C2M_N"] ;# Bank 129 - MGTYTXN0_129
set_property PACKAGE_PIN N34      [get_ports "FMCP_HSPC_DP1_C2M_N"] ;# Bank 129 - MGTYTXN1_129
set_property PACKAGE_PIN L34      [get_ports "FMCP_HSPC_DP2_C2M_N"] ;# Bank 129 - MGTYTXN2_129
set_property PACKAGE_PIN J34      [get_ports "FMCP_HSPC_DP3_C2M_N"] ;# Bank 129 - MGTYTXN3_129
set_property PACKAGE_PIN P35      [get_ports "FMCP_HSPC_DP0_C2M_P"] ;# Bank 129 - MGTYTXP0_129
set_property PACKAGE_PIN N33      [get_ports "FMCP_HSPC_DP1_C2M_P"] ;# Bank 129 - MGTYTXP1_129
set_property PACKAGE_PIN L33      [get_ports "FMCP_HSPC_DP2_C2M_P"] ;# Bank 129 - MGTYTXP2_129
set_property PACKAGE_PIN J33      [get_ports "FMCP_HSPC_DP3_C2M_P"] ;# Bank 129 - MGTYTXP3_129
set_property PACKAGE_PIN U34      [get_ports "FMCP_HSPC_GBTCLK1_M2C_C_N"] ;# Bank 130 - MGTREFCLK0N_130
set_property PACKAGE_PIN U33      [get_ports "FMCP_HSPC_GBTCLK1_M2C_C_P"] ;# Bank 130 - MGTREFCLK0P_130
set_property PACKAGE_PIN T32      [get_ports "USER_SMA_MGT_CLOCK_C_N"] ;# Bank 130 - MGTREFCLK1N_130
set_property PACKAGE_PIN T31      [get_ports "USER_SMA_MGT_CLOCK_C_P"] ;# Bank 130 - MGTREFCLK1P_130
set_property PACKAGE_PIN J39      [get_ports "FMCP_HSPC_DP4_M2C_N"] ;# Bank 130 - MGTYRXN0_130
set_property PACKAGE_PIN H37      [get_ports "FMCP_HSPC_DP5_M2C_N"] ;# Bank 130 - MGTYRXN1_130
set_property PACKAGE_PIN G39      [get_ports "FMCP_HSPC_DP6_M2C_N"] ;# Bank 130 - MGTYRXN2_130
set_property PACKAGE_PIN F37      [get_ports "FMCP_HSPC_DP7_M2C_N"] ;# Bank 130 - MGTYRXN3_130
set_property PACKAGE_PIN J38      [get_ports "FMCP_HSPC_DP4_M2C_P"] ;# Bank 130 - MGTYRXP0_130
set_property PACKAGE_PIN H36      [get_ports "FMCP_HSPC_DP5_M2C_P"] ;# Bank 130 - MGTYRXP1_130
set_property PACKAGE_PIN G38      [get_ports "FMCP_HSPC_DP6_M2C_P"] ;# Bank 130 - MGTYRXP2_130
set_property PACKAGE_PIN F36      [get_ports "FMCP_HSPC_DP7_M2C_P"] ;# Bank 130 - MGTYRXP3_130
set_property PACKAGE_PIN H32      [get_ports "FMCP_HSPC_DP4_C2M_N"] ;# Bank 130 - MGTYTXN0_130
set_property PACKAGE_PIN G34      [get_ports "FMCP_HSPC_DP5_C2M_N"] ;# Bank 130 - MGTYTXN1_130
set_property PACKAGE_PIN F32      [get_ports "FMCP_HSPC_DP6_C2M_N"] ;# Bank 130 - MGTYTXN2_130
set_property PACKAGE_PIN E34      [get_ports "FMCP_HSPC_DP7_C2M_N"] ;# Bank 130 - MGTYTXN3_130
set_property PACKAGE_PIN H31      [get_ports "FMCP_HSPC_DP4_C2M_P"] ;# Bank 130 - MGTYTXP0_130
set_property PACKAGE_PIN G33      [get_ports "FMCP_HSPC_DP5_C2M_P"] ;# Bank 130 - MGTYTXP1_130
set_property PACKAGE_PIN F31      [get_ports "FMCP_HSPC_DP6_C2M_P"] ;# Bank 130 - MGTYTXP2_130
set_property PACKAGE_PIN E33      [get_ports "FMCP_HSPC_DP7_C2M_P"] ;# Bank 130 - MGTYTXP3_130
set_property PACKAGE_PIN P32      [get_ports "FMCP_HSPC_GBTCLK2_M2C_C_N"] ;# Bank 131 - MGTREFCLK0N_131
set_property PACKAGE_PIN P31      [get_ports "FMCP_HSPC_GBTCLK2_M2C_C_P"] ;# Bank 131 - MGTREFCLK0P_131
set_property PACKAGE_PIN M32      [get_ports "No Connect"] ;# Bank 131 - MGTREFCLK1N_131
set_property PACKAGE_PIN M31      [get_ports "No Connect"] ;# Bank 131 - MGTREFCLK1P_131
set_property PACKAGE_PIN E39      [get_ports "FMCP_HSPC_DP8_M2C_N"] ;# Bank 131 - MGTYRXN0_131
set_property PACKAGE_PIN D37      [get_ports "FMCP_HSPC_DP9_M2C_N"] ;# Bank 131 - MGTYRXN1_131
set_property PACKAGE_PIN C39      [get_ports "FMCP_HSPC_DP10_M2C_N"] ;# Bank 131 - MGTYRXN2_131
set_property PACKAGE_PIN B37      [get_ports "FMCP_HSPC_DP11_M2C_N"] ;# Bank 131 - MGTYRXN3_131
set_property PACKAGE_PIN E38      [get_ports "FMCP_HSPC_DP8_M2C_P"] ;# Bank 131 - MGTYRXP0_131
set_property PACKAGE_PIN D36      [get_ports "FMCP_HSPC_DP9_M2C_P"] ;# Bank 131 - MGTYRXP1_131
set_property PACKAGE_PIN C38      [get_ports "FMCP_HSPC_DP10_M2C_P"] ;# Bank 131 - MGTYRXP2_131
set_property PACKAGE_PIN B36      [get_ports "FMCP_HSPC_DP11_M2C_P"] ;# Bank 131 - MGTYRXP3_131
set_property PACKAGE_PIN D32      [get_ports "FMCP_HSPC_DP8_C2M_N"] ;# Bank 131 - MGTYTXN0_131
set_property PACKAGE_PIN C34      [get_ports "FMCP_HSPC_DP9_C2M_N"] ;# Bank 131 - MGTYTXN1_131
set_property PACKAGE_PIN B32      [get_ports "FMCP_HSPC_DP10_C2M_N"] ;# Bank 131 - MGTYTXN2_131
set_property PACKAGE_PIN A34      [get_ports "FMCP_HSPC_DP11_C2M_N"] ;# Bank 131 - MGTYTXN3_131
set_property PACKAGE_PIN D31      [get_ports "FMCP_HSPC_DP8_C2M_P"] ;# Bank 131 - MGTYTXP0_131
set_property PACKAGE_PIN C33      [get_ports "FMCP_HSPC_DP9_C2M_P"] ;# Bank 131 - MGTYTXP1_131
set_property PACKAGE_PIN B31      [get_ports "FMCP_HSPC_DP10_C2M_P"] ;# Bank 131 - MGTYTXP2_131
set_property PACKAGE_PIN A33      [get_ports "FMCP_HSPC_DP11_C2M_P"] ;# Bank 131 - MGTYTXP3_131
#Other net   PACKAGE_PIN AF4      - RF1_CLKO_A_C_N           Bank 224 - ADC_CLK_N_224
#Other net   PACKAGE_PIN AF5      - RF1_CLKO_A_C_P           Bank 224 - ADC_CLK_P_224
#Other net   PACKAGE_PIN AB8      - N20236576                 Bank 224 - ADC_REXT_224
#Other net   PACKAGE_PIN AL5      - VCM01_224                 Bank 224 - VCM01_224
#Other net   PACKAGE_PIN AL4      - VCM23_224                 Bank 224 - VCM23_224
#Other net   PACKAGE_PIN AP1      - RFMC_ADC_00_N             Bank 224 - ADC_VIN_I01_N_224
#Other net   PACKAGE_PIN AP2      - RFMC_ADC_00_P             Bank 224 - ADC_VIN_I01_P_224
#Other net   PACKAGE_PIN AM1      - RFMC_ADC_01_N             Bank 224 - ADC_VIN_I23_N_224
#Other net   PACKAGE_PIN AM2      - RFMC_ADC_01_P             Bank 224 - ADC_VIN_I23_P_224
#Other net   PACKAGE_PIN AD4      - RF1_CLKO_B_C_N            Bank 225 - ADC_CLK_N_225
#Other net   PACKAGE_PIN AD5      - RF1_CLKO_B_C_P            Bank 225 - ADC_CLK_P_225
#Other net   PACKAGE_PIN AK5      - VCM01_225                 Bank 225 - VCM01_225
#Other net   PACKAGE_PIN AK4      - VCM23_225                 Bank 225 - VCM23_225
#Other net   PACKAGE_PIN AK1      - RFMC_ADC_02_N             Bank 225 - ADC_VIN_I01_N_225
#Other net   PACKAGE_PIN AK2      - RFMC_ADC_02_P             Bank 225 - ADC_VIN_I01_P_225
#Other net   PACKAGE_PIN AH1      - RFMC_ADC_03_N             Bank 225 - ADC_VIN_I23_N_225
#Other net   PACKAGE_PIN AH2      - RFMC_ADC_03_P             Bank 225 - ADC_VIN_I23_P_225
#Other net   PACKAGE_PIN AB4      - RF2_CLKO_B_C_N            Bank 226 - ADC_CLK_N_226
#Other net   PACKAGE_PIN AB5      - RF2_CLKO_B_C_P            Bank 226 - ADC_CLK_P_226
#Other net   PACKAGE_PIN AJ5      - VCM01_226                 Bank 226 - VCM01_226
#Other net   PACKAGE_PIN AJ4      - VCM23_226                 Bank 226 - VCM23_226
#Other net   PACKAGE_PIN AF1      - RFMC_ADC_04_N             Bank 226 - ADC_VIN_I01_N_226
#Other net   PACKAGE_PIN AF2      - RFMC_ADC_04_P             Bank 226 - ADC_VIN_I01_P_226
#Other net   PACKAGE_PIN AD1      - RFMC_ADC_05_N             Bank 226 - ADC_VIN_I23_N_226
#Other net   PACKAGE_PIN AD2      - RFMC_ADC_05_P             Bank 226 - ADC_VIN_I23_P_226
#Other net   PACKAGE_PIN Y4       - RF2_CLKO_A_C_N            Bank 227 - ADC_CLK_N_227
#Other net   PACKAGE_PIN Y5       - RF2_CLKO_A_C_P            Bank 227 - ADC_CLK_P_227
#Other net   PACKAGE_PIN AH5      - VCM01_227                 Bank 227 - VCM01_227
#Other net   PACKAGE_PIN AH4      - VCM23_227                 Bank 227 - VCM23_227
#Other net   PACKAGE_PIN AB1      - RFMC_ADC_06_N             Bank 227 - ADC_VIN_I01_N_227
#Other net   PACKAGE_PIN AB2      - RFMC_ADC_06_P             Bank 227 - ADC_VIN_I01_P_227
#Other net   PACKAGE_PIN Y1       - RFMC_ADC_07_N             Bank 227 - ADC_VIN_I23_N_227
#Other net   PACKAGE_PIN Y2       - RFMC_ADC_07_P             Bank 227 - ADC_VIN_I23_P_227
#Other net   PACKAGE_PIN R4       - RF3_CLKO_A_C_N           Bank 228 - DAC_CLK_N_228
#Other net   PACKAGE_PIN R5       - RF3_CLKO_A_C_P           Bank 228 - DAC_CLK_P_228
#Other net   PACKAGE_PIN W8       - N20238874                 Bank 228 - DAC_REXT_228
#Other net   PACKAGE_PIN U4       - SYSREF_RFSOC_C_N          Bank 228 - SYSREF_N_228
#Other net   PACKAGE_PIN U5       - SYSREF_RFSOC_C_P          Bank 228 - SYSREF_P_228
#Other net   PACKAGE_PIN U1       - RFMC_DAC_00_N             Bank 228 - DAC_VOUT0_N_228
#Other net   PACKAGE_PIN U2       - RFMC_DAC_00_P             Bank 228 - DAC_VOUT0_P_228
#Other net   PACKAGE_PIN R1       - RFMC_DAC_01_N             Bank 228 - DAC_VOUT1_N_228
#Other net   PACKAGE_PIN R2       - RFMC_DAC_01_P             Bank 228 - DAC_VOUT1_P_228
#Other net   PACKAGE_PIN N1       - RFMC_DAC_02_N             Bank 228 - DAC_VOUT2_N_228
#Other net   PACKAGE_PIN N2       - RFMC_DAC_02_P             Bank 228 - DAC_VOUT2_P_228
#Other net   PACKAGE_PIN L1       - RFMC_DAC_03_N             Bank 228 - DAC_VOUT3_N_228
#Other net   PACKAGE_PIN L2       - RFMC_DAC_03_P             Bank 228 - DAC_VOUT3_P_228
#Other net   PACKAGE_PIN N4       - RF3_CLKO_A_C_N            Bank 229 - DAC_CLK_N_229
#Other net   PACKAGE_PIN N5       - RF3_CLKO_A_C_P            Bank 229 - DAC_CLK_P_229
#Other net   PACKAGE_PIN J1       - RFMC_DAC_04_N             Bank 229 - DAC_VOUT0_N_229
#Other net   PACKAGE_PIN J2       - RFMC_DAC_04_P             Bank 229 - DAC_VOUT0_P_229
#Other net   PACKAGE_PIN G1       - RFMC_DAC_05_N             Bank 229 - DAC_VOUT1_N_229
#Other net   PACKAGE_PIN G2       - RFMC_DAC_05_P             Bank 229 - DAC_VOUT1_P_229
#Other net   PACKAGE_PIN E1       - RFMC_DAC_06_N             Bank 229 - DAC_VOUT2_N_229
#Other net   PACKAGE_PIN E2       - RFMC_DAC_06_P             Bank 229 - DAC_VOUT2_P_229
#Other net   PACKAGE_PIN C1       - RFMC_DAC_07_N             Bank 229 - DAC_VOUT3_N_229
#Other net   PACKAGE_PIN C2       - RFMC_DAC_07_P             Bank 229 - DAC_VOUT3_P_229
#Other net   PACKAGE_PIN R26      - MIO0_QSPI_LWR_CLK         Bank 500 - PS_MIO0
#Other net   PACKAGE_PIN P26      - MIO1_QSPI_LWR_DQ1         Bank 500 - PS_MIO1
#Other net   PACKAGE_PIN V27      - MIO10_QSPI_UPR_DQ2        Bank 500 - PS_MIO10
#Other net   PACKAGE_PIN P28      - MIO11_QSPI_UPR_DQ3        Bank 500 - PS_MIO11
#Other net   PACKAGE_PIN N29      - MIO12_QSPI_UPR_CLK        Bank 500 - PS_MIO12
#Other net   PACKAGE_PIN R28      - MIO13_PS_GPIO2            Bank 500 - PS_MIO13
#Other net   PACKAGE_PIN P29      - MIO14_I2C0_SCL            Bank 500 - PS_MIO14
#Other net   PACKAGE_PIN U28      - MIO15_I2C0_SDA            Bank 500 - PS_MIO15
#Other net   PACKAGE_PIN R29      - MIO16_I2C1_SCL            Bank 500 - PS_MIO16
#Other net   PACKAGE_PIN T29      - MIO17_I2C1_SDA            Bank 500 - PS_MIO17
#Other net   PACKAGE_PIN Y27      - UART0_TXD_MIO18_RXD       Bank 500 - PS_MIO18
#Other net   PACKAGE_PIN W28      - UART0_RXD_MIO19_TXD       Bank 500 - PS_MIO19
#Other net   PACKAGE_PIN Y26      - MIO2_QSPI_LWR_DQ2         Bank 500 - PS_MIO2
#Other net   PACKAGE_PIN V28      - No Connect                Bank 500 - PS_MIO20
#Other net   PACKAGE_PIN V29      - No Connect                Bank 500 - PS_MIO21
#Other net   PACKAGE_PIN Y28      - MIO22_BUTTON              Bank 500 - PS_MIO22
#Other net   PACKAGE_PIN U29      - MIO23_LED                 Bank 500 - PS_MIO23
#Other net   PACKAGE_PIN Y29      - No Connect                Bank 500 - PS_MIO24
#Other net   PACKAGE_PIN W29      - No Connect                Bank 500 - PS_MIO25
#Other net   PACKAGE_PIN T27      - MIO3_QSPI_LWR_DQ3         Bank 500 - PS_MIO3
#Other net   PACKAGE_PIN V26      - MIO4_QSPI_LWR_DQ0         Bank 500 - PS_MIO4
#Other net   PACKAGE_PIN AA26     - MIO5_QSPI_LWR_CS_B        Bank 500 - PS_MIO5
#Other net   PACKAGE_PIN W26      - No Connect                Bank 500 - PS_MIO6
#Other net   PACKAGE_PIN T26      - MIO7_QSPI_UPR_CS_B        Bank 500 - PS_MIO7
#Other net   PACKAGE_PIN U27      - MIO8_QSPI_UPR_DQ0         Bank 500 - PS_MIO8
#Other net   PACKAGE_PIN R27      - MIO9_QSPI_UPR_DQ1         Bank 500 - PS_MIO9
#Other net   PACKAGE_PIN G25      - MIO26_PMU_INPUT           Bank 501 - PS_MIO26
#Other net   PACKAGE_PIN C25      - MIO27_DP_AUX_OUT          Bank 501 - PS_MIO27
#Other net   PACKAGE_PIN F25      - MIO28_DP_HPD              Bank 501 - PS_MIO28
#Other net   PACKAGE_PIN B25      - MIO29_DP_OE               Bank 501 - PS_MIO29
#Other net   PACKAGE_PIN D25      - MIO30_DP_AUX_IN           Bank 501 - PS_MIO30
#Other net   PACKAGE_PIN C26      - No Connect                Bank 501 - PS_MIO31
#Other net   PACKAGE_PIN A26      - MIO32_PMU_GPO0            Bank 501 - PS_MIO32
#Other net   PACKAGE_PIN A27      - MIO33_PMU_GPO1            Bank 501 - PS_MIO33
#Other net   PACKAGE_PIN B27      - MIO34_PMU_GPO2            Bank 501 - PS_MIO34
#Other net   PACKAGE_PIN E26      - MIO35_PMU_GPO3            Bank 501 - PS_MIO35
#Other net   PACKAGE_PIN C27      - MIO36_PMU_GPO4            Bank 501 - PS_MIO36
#Other net   PACKAGE_PIN F26      - MIO37_PMU_GPO5            Bank 501 - PS_MIO37
#Other net   PACKAGE_PIN E27      - MIO38_PS_GPIO1            Bank 501 - PS_MIO38
#Other net   PACKAGE_PIN B28      - MIO39_SDIO_SEL            Bank 501 - PS_MIO39
#Other net   PACKAGE_PIN D26      - MIO40_SDIO_DIR_CMD        Bank 501 - PS_MIO40
#Other net   PACKAGE_PIN C28      - MIO41_SDIO_DIR_DAT0       Bank 501 - PS_MIO41
#Other net   PACKAGE_PIN E28      - MIO42_SDIO_DIR_DAT1       Bank 501 - PS_MIO42
#Other net   PACKAGE_PIN D28      - No Connect                Bank 501 - PS_MIO43
#Other net   PACKAGE_PIN F27      - No Connect                Bank 501 - PS_MIO44
#Other net   PACKAGE_PIN G27      - MIO45_SDIO_DETECT         Bank 501 - PS_MIO45
#Other net   PACKAGE_PIN A29      - MIO46_SDIO_DAT0_R         Bank 501 - PS_MIO46
#Other net   PACKAGE_PIN C29      - MIO47_SDIO_DAT1_R         Bank 501 - PS_MIO47
#Other net   PACKAGE_PIN D29      - MIO48_SDIO_DAT2_R         Bank 501 - PS_MIO48
#Other net   PACKAGE_PIN B29      - MIO49_SDIO_DAT3_R         Bank 501 - PS_MIO49
#Other net   PACKAGE_PIN F29      - MIO50_SDIO_CMD_R          Bank 501 - PS_MIO50
#Other net   PACKAGE_PIN E29      - MIO51_SDIO_CLK_R          Bank 501 - PS_MIO51
#Other net   PACKAGE_PIN N26      - MIO52_USB_CLK             Bank 502 - PS_MIO52
#Other net   PACKAGE_PIN L25      - MIO53_USB_DIR             Bank 502 - PS_MIO53
#Other net   PACKAGE_PIN M26      - MIO54_USB_DATA2           Bank 502 - PS_MIO54
#Other net   PACKAGE_PIN J25      - MIO55_USB_NXT             Bank 502 - PS_MIO55
#Other net   PACKAGE_PIN L26      - MIO56_USB_DATA0_R         Bank 502 - PS_MIO56
#Other net   PACKAGE_PIN H25      - MIO57_USB_DATA1_R         Bank 502 - PS_MIO57
#Other net   PACKAGE_PIN H26      - MIO58_USB_STP_R           Bank 502 - PS_MIO58
#Other net   PACKAGE_PIN H27      - MIO59_USB_DATA3_R         Bank 502 - PS_MIO59
#Other net   PACKAGE_PIN J26      - MIO60_USB_DATA4_R         Bank 502 - PS_MIO60
#Other net   PACKAGE_PIN G28      - MIO61_USB_DATA5_R         Bank 502 - PS_MIO61
#Other net   PACKAGE_PIN K26      - MIO62_USB_DATA6_R         Bank 502 - PS_MIO62
#Other net   PACKAGE_PIN G29      - MIO63_USB_DATA7_R         Bank 502 - PS_MIO63
#Other net   PACKAGE_PIN K27      - MIO64_ENET_TX_CLK         Bank 502 - PS_MIO64
#Other net   PACKAGE_PIN L27      - MIO65_ENET_TX_D0          Bank 502 - PS_MIO65
#Other net   PACKAGE_PIN N27      - MIO66_ENET_TX_D1          Bank 502 - PS_MIO66
#Other net   PACKAGE_PIN J28      - MIO67_ENET_TX_D2          Bank 502 - PS_MIO67
#Other net   PACKAGE_PIN H29      - MIO68_ENET_TX_D3          Bank 502 - PS_MIO68
#Other net   PACKAGE_PIN M27      - MIO69_ENET_TX_CTRL        Bank 502 - PS_MIO69
#Other net   PACKAGE_PIN K28      - MIO70_ENET_RX_CLK         Bank 502 - PS_MIO70
#Other net   PACKAGE_PIN H28      - MIO71_ENET_RX_D0          Bank 502 - PS_MIO71
#Other net   PACKAGE_PIN J29      - MIO72_ENET_RX_D1          Bank 502 - PS_MIO72
#Other net   PACKAGE_PIN K29      - MIO73_ENET_RX_D2          Bank 502 - PS_MIO73
#Other net   PACKAGE_PIN M28      - MIO74_ENET_RX_D3          Bank 502 - PS_MIO74
#Other net   PACKAGE_PIN N28      - MIO75_ENET_RX_CTRL        Bank 502 - PS_MIO75
#Other net   PACKAGE_PIN M29      - MIO76_ENET_MDC            Bank 502 - PS_MIO76
#Other net   PACKAGE_PIN L29      - MIO77_ENET_MDIO           Bank 502 - PS_MIO77
#Other net   PACKAGE_PIN AB27     - PS_DONE                   Bank 503 - PS_DONE
#Other net   PACKAGE_PIN AF30     - PS_ERR_OUT                Bank 503 - PS_ERROR_OUT
#Other net   PACKAGE_PIN AC32     - PS_ERR_STATUS             Bank 503 - PS_ERROR_STATUS
#Other net   PACKAGE_PIN AB26     - PS_INIT_B                 Bank 503 - PS_INIT_B
#Other net   PACKAGE_PIN AA29     - FPGA_TCK                  Bank 503 - PS_JTAG_TCK
#Other net   PACKAGE_PIN AD31     - FPGA_TDI                  Bank 503 - PS_JTAG_TDI
#Other net   PACKAGE_PIN AD32     - FPGA_TDO_FMC_TDI          Bank 503 - PS_JTAG_TDO
#Other net   PACKAGE_PIN AD30     - FPGA_TMS                  Bank 503 - PS_JTAG_TMS
#Other net   PACKAGE_PIN AC31     - PS_MODE0                  Bank 503 - PS_MODE0
#Other net   PACKAGE_PIN AB30     - PS_MODE1                  Bank 503 - PS_MODE1
#Other net   PACKAGE_PIN AB31     - PS_MODE2                  Bank 503 - PS_MODE2
#Other net   PACKAGE_PIN AE32     - PS_MODE3                  Bank 503 - PS_MODE3
#Other net   PACKAGE_PIN AE31     - PS_PADI                   Bank 503 - PS_PADI
#Other net   PACKAGE_PIN AF31     - PS_PADO                   Bank 503 - PS_PADO
#Other net   PACKAGE_PIN AB29     - PS_POR_B                  Bank 503 - PS_POR_B
#Other net   PACKAGE_PIN AA27     - PS_PROG_B                 Bank 503 - PS_PROG_B
#Other net   PACKAGE_PIN AC30     - PS_REF_CLK                Bank 503 - PS_REF_CLK
#Other net   PACKAGE_PIN AB28     - PS_SRST_B                 Bank 503 - PS_SRST_B
#Other net   PACKAGE_PIN AV31     - PS_DDR4_SODIMM_A0         Bank 504 - PS_DDR_A0
#Other net   PACKAGE_PIN AW28     - PS_DDR4_SODIMM_A1         Bank 504 - PS_DDR_A1
#Other net   PACKAGE_PIN AT31     - PS_DDR4_SODIMM_A10        Bank 504 - PS_DDR_A10
#Other net   PACKAGE_PIN AT32     - PS_DDR4_SODIMM_A11        Bank 504 - PS_DDR_A11
#Other net   PACKAGE_PIN AT30     - PS_DDR4_SODIMM_A12        Bank 504 - PS_DDR_A12
#Other net   PACKAGE_PIN AU32     - PS_DDR4_SODIMM_A13        Bank 504 - PS_DDR_A13
#Other net   PACKAGE_PIN AR28     - PS_DDR4_SODIMM_WE_B       Bank 504 - PS_DDR_A14
#Other net   PACKAGE_PIN AP30     - PS_DDR4_SODIMM_CAS_B      Bank 504 - PS_DDR_A15
#Other net   PACKAGE_PIN AP28     - PS_DDR4_SODIMM_RAS_B      Bank 504 - PS_DDR_A16
#Other net   PACKAGE_PIN AK29     - No Connect                Bank 504 - PS_DDR_A17
#Other net   PACKAGE_PIN AV28     - PS_DDR4_SODIMM_A2         Bank 504 - PS_DDR_A2
#Other net   PACKAGE_PIN AU29     - PS_DDR4_SODIMM_A3         Bank 504 - PS_DDR_A3
#Other net   PACKAGE_PIN AW31     - PS_DDR4_SODIMM_A4         Bank 504 - PS_DDR_A4
#Other net   PACKAGE_PIN AU28     - PS_DDR4_SODIMM_A5         Bank 504 - PS_DDR_A5
#Other net   PACKAGE_PIN AL29     - PS_DDR4_SODIMM_A6         Bank 504 - PS_DDR_A6
#Other net   PACKAGE_PIN AM30     - PS_DDR4_SODIMM_A7         Bank 504 - PS_DDR_A7
#Other net   PACKAGE_PIN AM29     - PS_DDR4_SODIMM_A8         Bank 504 - PS_DDR_A8
#Other net   PACKAGE_PIN AP29     - PS_DDR4_SODIMM_A9         Bank 504 - PS_DDR_A9
#Other net   PACKAGE_PIN AL30     - PS_DDR4_SODIMM_ACT_B      Bank 504 - PS_DDR_ACT_N
#Other net   PACKAGE_PIN AL32     - PS_DDR4_SODIMM_ALERT_B    Bank 504 - PS_DDR_ALERT_N
#Other net   PACKAGE_PIN AN30     - PS_DDR4_SODIMM_BA0        Bank 504 - PS_DDR_BA0
#Other net   PACKAGE_PIN AM32     - PS_DDR4_SODIMM_BA1        Bank 504 - PS_DDR_BA1
#Other net   PACKAGE_PIN AN32     - PS_DDR4_SODIMM_BG0        Bank 504 - PS_DDR_BG0
#Other net   PACKAGE_PIN AL31     - PS_DDR4_SODIMM_BG1        Bank 504 - PS_DDR_BG1
#Other net   PACKAGE_PIN AU30     - PS_DDR4_SODIMM_CK0_T      Bank 504 - PS_DDR_CK0
#Other net   PACKAGE_PIN AR29     - PS_DDR4_SODIMM_CK1_T      Bank 504 - PS_DDR_CK1
#Other net   PACKAGE_PIN AW30     - PS_DDR4_SODIMM_CKE0       Bank 504 - PS_DDR_CKE0
#Other net   PACKAGE_PIN AR32     - PS_DDR4_SODIMM_CKE1       Bank 504 - PS_DDR_CKE1
#Other net   PACKAGE_PIN AV30     - PS_DDR4_SODIMM_CK0_C      Bank 504 - PS_DDR_CK_N0
#Other net   PACKAGE_PIN AT29     - PS_DDR4_SODIMM_CK1_C      Bank 504 - PS_DDR_CK_N1
#Other net   PACKAGE_PIN AW29     - PS_DDR4_SODIMM_CS0_B      Bank 504 - PS_DDR_CS_N0
#Other net   PACKAGE_PIN AR31     - PS_DDR4_SODIMM_CS1_B      Bank 504 - PS_DDR_CS_N1
#Other net   PACKAGE_PIN AU23     - PS_DDR4_SODIMM_DM0_B      Bank 504 - PS_DDR_DM0
#Other net   PACKAGE_PIN AT27     - PS_DDR4_SODIMM_DM1_B      Bank 504 - PS_DDR_DM1
#Other net   PACKAGE_PIN AL24     - PS_DDR4_SODIMM_DM2_B      Bank 504 - PS_DDR_DM2
#Other net   PACKAGE_PIN AM27     - PS_DDR4_SODIMM_DM3_B      Bank 504 - PS_DDR_DM3
#Other net   PACKAGE_PIN AV36     - PS_DDR4_SODIMM_DM4_B      Bank 504 - PS_DDR_DM4
#Other net   PACKAGE_PIN AT35     - PS_DDR4_SODIMM_DM5_B      Bank 504 - PS_DDR_DM5
#Other net   PACKAGE_PIN AM36     - PS_DDR4_SODIMM_DM6_B      Bank 504 - PS_DDR_DM6
#Other net   PACKAGE_PIN AJ32     - PS_DDR4_SODIMM_DM7_B      Bank 504 - PS_DDR_DM7
#Other net   PACKAGE_PIN AR38     - No Connect                Bank 504 - PS_DDR_DM8
#Other net   PACKAGE_PIN AW25     - PS_DDR4_SODIMM_DQ0        Bank 504 - PS_DDR_DQ0
#Other net   PACKAGE_PIN AW24     - PS_DDR4_SODIMM_DQ1        Bank 504 - PS_DDR_DQ1
#Other net   PACKAGE_PIN AU25     - PS_DDR4_SODIMM_DQ10       Bank 504 - PS_DDR_DQ10
#Other net   PACKAGE_PIN AR27     - PS_DDR4_SODIMM_DQ11       Bank 504 - PS_DDR_DQ11
#Other net   PACKAGE_PIN AU27     - PS_DDR4_SODIMM_DQ12       Bank 504 - PS_DDR_DQ12
#Other net   PACKAGE_PIN AV26     - PS_DDR4_SODIMM_DQ13       Bank 504 - PS_DDR_DQ13
#Other net   PACKAGE_PIN AV27     - PS_DDR4_SODIMM_DQ14       Bank 504 - PS_DDR_DQ14
#Other net   PACKAGE_PIN AW26     - PS_DDR4_SODIMM_DQ15       Bank 504 - PS_DDR_DQ15
#Other net   PACKAGE_PIN AP25     - PS_DDR4_SODIMM_DQ16       Bank 504 - PS_DDR_DQ16
#Other net   PACKAGE_PIN AP24     - PS_DDR4_SODIMM_DQ17       Bank 504 - PS_DDR_DQ17
#Other net   PACKAGE_PIN AP23     - PS_DDR4_SODIMM_DQ18       Bank 504 - PS_DDR_DQ18
#Other net   PACKAGE_PIN AN25     - PS_DDR4_SODIMM_DQ19       Bank 504 - PS_DDR_DQ19
#Other net   PACKAGE_PIN AV25     - PS_DDR4_SODIMM_DQ2        Bank 504 - PS_DDR_DQ2
#Other net   PACKAGE_PIN AM25     - PS_DDR4_SODIMM_DQ20       Bank 504 - PS_DDR_DQ20
#Other net   PACKAGE_PIN AK24     - PS_DDR4_SODIMM_DQ21       Bank 504 - PS_DDR_DQ21
#Other net   PACKAGE_PIN AN23     - PS_DDR4_SODIMM_DQ22       Bank 504 - PS_DDR_DQ22
#Other net   PACKAGE_PIN AK23     - PS_DDR4_SODIMM_DQ23       Bank 504 - PS_DDR_DQ23
#Other net   PACKAGE_PIN AK26     - PS_DDR4_SODIMM_DQ24       Bank 504 - PS_DDR_DQ24
#Other net   PACKAGE_PIN AL25     - PS_DDR4_SODIMM_DQ25       Bank 504 - PS_DDR_DQ25
#Other net   PACKAGE_PIN AK28     - PS_DDR4_SODIMM_DQ26       Bank 504 - PS_DDR_DQ26
#Other net   PACKAGE_PIN AK27     - PS_DDR4_SODIMM_DQ27       Bank 504 - PS_DDR_DQ27
#Other net   PACKAGE_PIN AN27     - PS_DDR4_SODIMM_DQ28       Bank 504 - PS_DDR_DQ28
#Other net   PACKAGE_PIN AN26     - PS_DDR4_SODIMM_DQ29       Bank 504 - PS_DDR_DQ29
#Other net   PACKAGE_PIN AW23     - PS_DDR4_SODIMM_DQ3        Bank 504 - PS_DDR_DQ3
#Other net   PACKAGE_PIN AN28     - PS_DDR4_SODIMM_DQ30       Bank 504 - PS_DDR_DQ30
#Other net   PACKAGE_PIN AM28     - PS_DDR4_SODIMM_DQ31       Bank 504 - PS_DDR_DQ31
#Other net   PACKAGE_PIN AU39     - PS_DDR4_SODIMM_DQ32       Bank 504 - PS_DDR_DQ32
#Other net   PACKAGE_PIN AU38     - PS_DDR4_SODIMM_DQ33       Bank 504 - PS_DDR_DQ33
#Other net   PACKAGE_PIN AU37     - PS_DDR4_SODIMM_DQ34       Bank 504 - PS_DDR_DQ34
#Other net   PACKAGE_PIN AU35     - PS_DDR4_SODIMM_DQ35       Bank 504 - PS_DDR_DQ35
#Other net   PACKAGE_PIN AV38     - PS_DDR4_SODIMM_DQ36       Bank 504 - PS_DDR_DQ36
#Other net   PACKAGE_PIN AW36     - PS_DDR4_SODIMM_DQ37       Bank 504 - PS_DDR_DQ37
#Other net   PACKAGE_PIN AV35     - PS_DDR4_SODIMM_DQ38       Bank 504 - PS_DDR_DQ38
#Other net   PACKAGE_PIN AW35     - PS_DDR4_SODIMM_DQ39       Bank 504 - PS_DDR_DQ39
#Other net   PACKAGE_PIN AV23     - PS_DDR4_SODIMM_DQ4        Bank 504 - PS_DDR_DQ4
#Other net   PACKAGE_PIN AU33     - PS_DDR4_SODIMM_DQ40       Bank 504 - PS_DDR_DQ40
#Other net   PACKAGE_PIN AV33     - PS_DDR4_SODIMM_DQ41       Bank 504 - PS_DDR_DQ41
#Other net   PACKAGE_PIN AW34     - PS_DDR4_SODIMM_DQ42       Bank 504 - PS_DDR_DQ42
#Other net   PACKAGE_PIN AW33     - PS_DDR4_SODIMM_DQ43       Bank 504 - PS_DDR_DQ43
#Other net   PACKAGE_PIN AR34     - PS_DDR4_SODIMM_DQ44       Bank 504 - PS_DDR_DQ44
#Other net   PACKAGE_PIN AR33     - PS_DDR4_SODIMM_DQ45       Bank 504 - PS_DDR_DQ45
#Other net   PACKAGE_PIN AP33     - PS_DDR4_SODIMM_DQ46       Bank 504 - PS_DDR_DQ46
#Other net   PACKAGE_PIN AP34     - PS_DDR4_SODIMM_DQ47       Bank 504 - PS_DDR_DQ47
#Other net   PACKAGE_PIN AL39     - PS_DDR4_SODIMM_DQ48       Bank 504 - PS_DDR_DQ48
#Other net   PACKAGE_PIN AM38     - PS_DDR4_SODIMM_DQ49       Bank 504 - PS_DDR_DQ49
#Other net   PACKAGE_PIN AV22     - PS_DDR4_SODIMM_DQ5        Bank 504 - PS_DDR_DQ5
#Other net   PACKAGE_PIN AM39     - PS_DDR4_SODIMM_DQ50       Bank 504 - PS_DDR_DQ50
#Other net   PACKAGE_PIN AN38     - PS_DDR4_SODIMM_DQ51       Bank 504 - PS_DDR_DQ51
#Other net   PACKAGE_PIN AM35     - PS_DDR4_SODIMM_DQ52       Bank 504 - PS_DDR_DQ52
#Other net   PACKAGE_PIN AM34     - PS_DDR4_SODIMM_DQ53       Bank 504 - PS_DDR_DQ53
#Other net   PACKAGE_PIN AN36     - PS_DDR4_SODIMM_DQ54       Bank 504 - PS_DDR_DQ54
#Other net   PACKAGE_PIN AN35     - PS_DDR4_SODIMM_DQ55       Bank 504 - PS_DDR_DQ55
#Other net   PACKAGE_PIN AK32     - PS_DDR4_SODIMM_DQ56       Bank 504 - PS_DDR_DQ56
#Other net   PACKAGE_PIN AK31     - PS_DDR4_SODIMM_DQ57       Bank 504 - PS_DDR_DQ57
#Other net   PACKAGE_PIN AJ31     - PS_DDR4_SODIMM_DQ58       Bank 504 - PS_DDR_DQ58
#Other net   PACKAGE_PIN AJ30     - PS_DDR4_SODIMM_DQ59       Bank 504 - PS_DDR_DQ59
#Other net   PACKAGE_PIN AR24     - PS_DDR4_SODIMM_DQ6        Bank 504 - PS_DDR_DQ6
#Other net   PACKAGE_PIN AH30     - PS_DDR4_SODIMM_DQ60       Bank 504 - PS_DDR_DQ60
#Other net   PACKAGE_PIN AG32     - PS_DDR4_SODIMM_DQ61       Bank 504 - PS_DDR_DQ61
#Other net   PACKAGE_PIN AF32     - PS_DDR4_SODIMM_DQ62       Bank 504 - PS_DDR_DQ62
#Other net   PACKAGE_PIN AG30     - PS_DDR4_SODIMM_DQ63       Bank 504 - PS_DDR_DQ63
#Other net   PACKAGE_PIN AT36     - No Connect                Bank 504 - PS_DDR_DQ64
#Other net   PACKAGE_PIN AR36     - No Connect                Bank 504 - PS_DDR_DQ65
#Other net   PACKAGE_PIN AT39     - No Connect                Bank 504 - PS_DDR_DQ66
#Other net   PACKAGE_PIN AP35     - No Connect                Bank 504 - PS_DDR_DQ67
#Other net   PACKAGE_PIN AR39     - No Connect                Bank 504 - PS_DDR_DQ68
#Other net   PACKAGE_PIN AP38     - No Connect                Bank 504 - PS_DDR_DQ69
#Other net   PACKAGE_PIN AR23     - PS_DDR4_SODIMM_DQ7        Bank 504 - PS_DDR_DQ7
#Other net   PACKAGE_PIN AP36     - No Connect                Bank 504 - PS_DDR_DQ70
#Other net   PACKAGE_PIN AP39     - No Connect                Bank 504 - PS_DDR_DQ71
#Other net   PACKAGE_PIN AT25     - PS_DDR4_SODIMM_DQ8        Bank 504 - PS_DDR_DQ8
#Other net   PACKAGE_PIN AP26     - PS_DDR4_SODIMM_DQ9        Bank 504 - PS_DDR_DQ9
#Other net   PACKAGE_PIN AU24     - PS_DDR4_SODIMM_DQS0_C     Bank 504 - PS_DDR_DQS_N0
#Other net   PACKAGE_PIN AT26     - PS_DDR4_SODIMM_DQS1_C     Bank 504 - PS_DDR_DQS_N1
#Other net   PACKAGE_PIN AM24     - PS_DDR4_SODIMM_DQS2_C     Bank 504 - PS_DDR_DQS_N2
#Other net   PACKAGE_PIN AL27     - PS_DDR4_SODIMM_DQS3_C     Bank 504 - PS_DDR_DQS_N3
#Other net   PACKAGE_PIN AW37     - PS_DDR4_SODIMM_DQS4_C     Bank 504 - PS_DDR_DQS_N4
#Other net   PACKAGE_PIN AU34     - PS_DDR4_SODIMM_DQS5_C     Bank 504 - PS_DDR_DQS_N5
#Other net   PACKAGE_PIN AN37     - PS_DDR4_SODIMM_DQS6_C     Bank 504 - PS_DDR_DQS_N6
#Other net   PACKAGE_PIN AH32     - PS_DDR4_SODIMM_DQS7_C     Bank 504 - PS_DDR_DQS_N7
#Other net   PACKAGE_PIN AT37     - No Connect                Bank 504 - PS_DDR_DQS_N8
#Other net   PACKAGE_PIN AT24     - PS_DDR4_SODIMM_DQS0_T     Bank 504 - PS_DDR_DQS_P0
#Other net   PACKAGE_PIN AR26     - PS_DDR4_SODIMM_DQS1_T     Bank 504 - PS_DDR_DQS_P1
#Other net   PACKAGE_PIN AM23     - PS_DDR4_SODIMM_DQS2_T     Bank 504 - PS_DDR_DQS_P2
#Other net   PACKAGE_PIN AL26     - PS_DDR4_SODIMM_DQS3_T     Bank 504 - PS_DDR_DQS_P3
#Other net   PACKAGE_PIN AV37     - PS_DDR4_SODIMM_DQS4_T     Bank 504 - PS_DDR_DQS_P4
#Other net   PACKAGE_PIN AT34     - PS_DDR4_SODIMM_DQS5_T     Bank 504 - PS_DDR_DQS_P5
#Other net   PACKAGE_PIN AM37     - PS_DDR4_SODIMM_DQS6_T     Bank 504 - PS_DDR_DQS_P6
#Other net   PACKAGE_PIN AH31     - PS_DDR4_SODIMM_DQS7_T     Bank 504 - PS_DDR_DQS_P7
#Other net   PACKAGE_PIN AR37     - No Connect                Bank 504 - PS_DDR_DQS_P8
#Other net   PACKAGE_PIN AV32     - PS_DDR4_SODIMM_ODT0       Bank 504 - PS_DDR_ODT0
#Other net   PACKAGE_PIN AP31     - PS_DDR4_SODIMM_ODT1       Bank 504 - PS_DDR_ODT1
#Other net   PACKAGE_PIN AN31     - PS_DDR4_SODIMM_PARITY     Bank 504 - PS_DDR_PARITY
#Other net   PACKAGE_PIN AM33     - ZYNQ_DDR4_SODIMM_RESET_B  Bank 504 - PS_DDR_RAM_RST_N
#Other net   PACKAGE_PIN AN33     - N17010067                 Bank 504 - PS_DDR_ZQ
#Other net   PACKAGE_PIN AJ39     - No Connect                Bank 505 - PS_MGTRRXN0_505
#Other net   PACKAGE_PIN AG39     - No Connect                Bank 505 - PS_MGTRRXN1_505
#Other net   PACKAGE_PIN AE39     - GT2_USB0_RX_N             Bank 505 - PS_MGTRRXN2_505
#Other net   PACKAGE_PIN AC39     - GT3_SATA1_RX_N            Bank 505 - PS_MGTRRXN3_505
#Other net   PACKAGE_PIN AJ38     - No Connect                Bank 505 - PS_MGTRRXP0_505
#Other net   PACKAGE_PIN AG38     - No Connect                Bank 505 - PS_MGTRRXP1_505
#Other net   PACKAGE_PIN AE38     - GT2_USB0_RX_P             Bank 505 - PS_MGTRRXP2_505
#Other net   PACKAGE_PIN AC38     - GT3_SATA1_RX_P            Bank 505 - PS_MGTRRXP3_505
#Other net   PACKAGE_PIN AK37     - DP_TX0_N                  Bank 505 - PS_MGTRTXN0_505
#Other net   PACKAGE_PIN AH37     - DP_TX1_N                  Bank 505 - PS_MGTRTXN1_505
#Other net   PACKAGE_PIN AF37     - GT2_USB0_TX_N             Bank 505 - PS_MGTRTXN2_505
#Other net   PACKAGE_PIN AD37     - GT3_SATA1_TX_N            Bank 505 - PS_MGTRTXN3_505
#Other net   PACKAGE_PIN AK36     - DP_TX0_P                  Bank 505 - PS_MGTRTXP0_505
#Other net   PACKAGE_PIN AH36     - DP_TX1_P                  Bank 505 - PS_MGTRTXP1_505
#Other net   PACKAGE_PIN AF36     - GT2_USB0_TX_P             Bank 505 - PS_MGTRTXP2_505
#Other net   PACKAGE_PIN AD36     - GT3_SATA1_TX_P            Bank 505 - PS_MGTRTXP3_505
#Other net   PACKAGE_PIN AJ35     - No Connect                Bank 505 - PS_MGTREFCLK0N_505
#Other net   PACKAGE_PIN AJ34     - No Connect                Bank 505 - PS_MGTREFCLK0P_505
#Other net   PACKAGE_PIN AG35     - GTR_REF_CLK_DP_C_N        Bank 505 - PS_MGTREFCLK1N_505
#Other net   PACKAGE_PIN AG34     - GTR_REF_CLK_DP_C_P        Bank 505 - PS_MGTREFCLK1P_505
#Other net   PACKAGE_PIN AE35     - GTR_REF_CLK_USB3_C_N      Bank 505 - PS_MGTREFCLK2N_505
#Other net   PACKAGE_PIN AE34     - GTR_REF_CLK_USB3_C_P      Bank 505 - PS_MGTREFCLK2P_505
#Other net   PACKAGE_PIN AC35     - GTR_REF_CLK_SATA_C_N      Bank 505 - PS_MGTREFCLK3N_505
#Other net   PACKAGE_PIN AC34     - GTR_REF_CLK_SATA_C_P      Bank 505 - PS_MGTREFCLK3P_505
#Other net   PACKAGE_PIN AK34     - N17536638                 Bank 505 - PS_MGTRREF_505
#Other net   PACKAGE_PIN AE29     - PS_SYSMON_AVCC            Bank 999 - VCC_PSADC
#Other net   PACKAGE_PIN AF29     - PS_SYSMON_AGND            Bank 999 - GND_PSADC
#Other net   PACKAGE_PIN AJ17     - VCC1V8                    Bank  64 - VCCO_64
#Other net   PACKAGE_PIN AM16     - VCC1V8                    Bank  64 - VCCO_64
#Other net   PACKAGE_PIN AR15     - VCC1V8                    Bank  64 - VCCO_64
#Other net   PACKAGE_PIN AH14     - VADJ_FMC                  Bank  65 - VCCO_65
#Other net   PACKAGE_PIN AM11     - VADJ_FMC                  Bank  65 - VCCO_65
#Other net   PACKAGE_PIN AR10     - VADJ_FMC                  Bank  65 - VCCO_65
#Other net   PACKAGE_PIN AK20     - VADJ_FMC                  Bank  66 - VCCO_66
#Other net   PACKAGE_PIN AN19     - VADJ_FMC                  Bank  66 - VCCO_66
#Other net   PACKAGE_PIN AT18     - VADJ_FMC                  Bank  66 - VCCO_66
#Other net   PACKAGE_PIN D22      - VCC1V2                    Bank  67 - VCCO_67
#Other net   PACKAGE_PIN G21      - VCC1V2                    Bank  67 - VCCO_67
#Other net   PACKAGE_PIN K20      - VCC1V2                    Bank  67 - VCCO_67
#Other net   PACKAGE_PIN D12      - VCC1V2                    Bank  68 - VCCO_68
#Other net   PACKAGE_PIN G11      - VCC1V2                    Bank  68 - VCCO_68
#Other net   PACKAGE_PIN H14      - VCC1V2                    Bank  68 - VCCO_68
#Other net   PACKAGE_PIN E15      - VCC1V2                    Bank  69 - VCCO_69
#Other net   PACKAGE_PIN F18      - VCC1V2                    Bank  69 - VCCO_69
#Other net   PACKAGE_PIN J17      - VCC1V2                    Bank  69 - VCCO_69
#Other net   PACKAGE_PIN AP7      - VCC1V8                    Bank  84 - VCCO_84
#Other net   PACKAGE_PIN AU6      - VCC1V8                    Bank  84 - VCCO_84
#Other net   PACKAGE_PIN B6       - VCC1V8                    Bank  87 - VCCO_87
#Other net   PACKAGE_PIN C9       - VCC1V8                    Bank  87 - VCCO_87
#Other net   PACKAGE_PIN U26      - VCC1V8                    Bank 500 - VCCO_PSIO0_500
#Other net   PACKAGE_PIN W27      - VCC1V8                    Bank 500 - VCCO_PSIO0_500
#Other net   PACKAGE_PIN B26      - VCC1V8                    Bank 501 - VCCO_PSIO1_501
#Other net   PACKAGE_PIN E25      - VCC1V8                    Bank 501 - VCCO_PSIO1_501
#Other net   PACKAGE_PIN H24      - VCC1V8                    Bank 502 - VCCO_PSIO2_502
#Other net   PACKAGE_PIN J27      - VCC1V8                    Bank 502 - VCCO_PSIO2_502
#Other net   PACKAGE_PIN AC29     - VCC1V8                    Bank 503 - VCCO_PSIO3_503
#Other net   PACKAGE_PIN AE30     - VCC1V8                    Bank 503 - VCCO_PSIO3_503
#Other net   PACKAGE_PIN AM31     - VCC1V2                    Bank 504 - VCCO_PSDDR_504
#Other net   PACKAGE_PIN AP27     - VCC1V2                    Bank 504 - VCCO_PSDDR_504
#Other net   PACKAGE_PIN AP32     - VCC1V2                    Bank 504 - VCCO_PSDDR_504
#Other net   PACKAGE_PIN AT28     - VCC1V2                    Bank 504 - VCCO_PSDDR_504
#Other net   PACKAGE_PIN AT33     - VCC1V2                    Bank 504 - VCCO_PSDDR_504
#Other net   PACKAGE_PIN AV29     - VCC1V2                    Bank 504 - VCCO_PSDDR_504
#Other net   PACKAGE_PIN AV34     - VCC1V2                    Bank 504 - VCCO_PSDDR_504
#Other net   PACKAGE_PIN AB9      - ADC_AVCC                  Bank 999 - ADC_AVCC
#Other net   PACKAGE_PIN AC9      - ADC_AVCC                  Bank 999 - ADC_AVCC
#Other net   PACKAGE_PIN AD9      - ADC_AVCC                  Bank 999 - ADC_AVCC
#Other net   PACKAGE_PIN AE9      - ADC_AVCC                  Bank 999 - ADC_AVCC
#Other net   PACKAGE_PIN AF9      - ADC_AVCC                  Bank 999 - ADC_AVCC
#Other net   PACKAGE_PIN AG9      - ADC_AVCC                  Bank 999 - ADC_AVCC
#Other net   PACKAGE_PIN AH9      - ADC_AVCC                  Bank 999 - ADC_AVCC
#Other net   PACKAGE_PIN AJ9      - ADC_AVCC                  Bank 999 - ADC_AVCC
#Other net   PACKAGE_PIN AB7      - ADC_AVCCAUX               Bank 999 - ADC_AVCCAUX
#Other net   PACKAGE_PIN AC7      - ADC_AVCCAUX               Bank 999 - ADC_AVCCAUX
#Other net   PACKAGE_PIN AD7      - ADC_AVCCAUX               Bank 999 - ADC_AVCCAUX
#Other net   PACKAGE_PIN AE7      - ADC_AVCCAUX               Bank 999 - ADC_AVCCAUX
#Other net   PACKAGE_PIN AF7      - ADC_AVCCAUX               Bank 999 - ADC_AVCCAUX
#Other net   PACKAGE_PIN AG7      - ADC_AVCCAUX               Bank 999 - ADC_AVCCAUX
#Other net   PACKAGE_PIN AH7      - ADC_AVCCAUX               Bank 999 - ADC_AVCCAUX
#Other net   PACKAGE_PIN AJ7      - ADC_AVCCAUX               Bank 999 - ADC_AVCCAUX
#Other net   PACKAGE_PIN AA1      - GND                       Bank 999 - ADC_GND
#Other net   PACKAGE_PIN AA10     - GND                       Bank 999 - ADC_GND
#Other net   PACKAGE_PIN AA2      - GND                       Bank 999 - ADC_GND
#Other net   PACKAGE_PIN AA3      - GND                       Bank 999 - ADC_GND
#Other net   PACKAGE_PIN AA4      - GND                       Bank 999 - ADC_GND
#Other net   PACKAGE_PIN AA5      - GND                       Bank 999 - ADC_GND
#Other net   PACKAGE_PIN AA6      - GND                       Bank 999 - ADC_GND
#Other net   PACKAGE_PIN AA7      - GND                       Bank 999 - ADC_GND
#Other net   PACKAGE_PIN AA8      - GND                       Bank 999 - ADC_GND
#Other net   PACKAGE_PIN AA9      - GND                       Bank 999 - ADC_GND
#Other net   PACKAGE_PIN AB10     - GND                       Bank 999 - ADC_GND
#Other net   PACKAGE_PIN AB3      - GND                       Bank 999 - ADC_GND
#Other net   PACKAGE_PIN AB6      - GND                       Bank 999 - ADC_GND
#Other net   PACKAGE_PIN AC1      - GND                       Bank 999 - ADC_GND
#Other net   PACKAGE_PIN AC10     - GND                       Bank 999 - ADC_GND
#Other net   PACKAGE_PIN AC2      - GND                       Bank 999 - ADC_GND
#Other net   PACKAGE_PIN AC3      - GND                       Bank 999 - ADC_GND
#Other net   PACKAGE_PIN AC4      - GND                       Bank 999 - ADC_GND
#Other net   PACKAGE_PIN AC5      - GND                       Bank 999 - ADC_GND
#Other net   PACKAGE_PIN AC6      - GND                       Bank 999 - ADC_GND
#Other net   PACKAGE_PIN AC8      - GND                       Bank 999 - ADC_GND
#Other net   PACKAGE_PIN AD10     - GND                       Bank 999 - ADC_GND
#Other net   PACKAGE_PIN AD3      - GND                       Bank 999 - ADC_GND
#Other net   PACKAGE_PIN AD6      - GND                       Bank 999 - ADC_GND
#Other net   PACKAGE_PIN AD8      - GND                       Bank 999 - ADC_GND
#Other net   PACKAGE_PIN AE1      - GND                       Bank 999 - ADC_GND
#Other net   PACKAGE_PIN AE2      - GND                       Bank 999 - ADC_GND
#Other net   PACKAGE_PIN AE3      - GND                       Bank 999 - ADC_GND
#Other net   PACKAGE_PIN AE4      - GND                       Bank 999 - ADC_GND
#Other net   PACKAGE_PIN AE5      - GND                       Bank 999 - ADC_GND
#Other net   PACKAGE_PIN AE6      - GND                       Bank 999 - ADC_GND
#Other net   PACKAGE_PIN AE8      - GND                       Bank 999 - ADC_GND
#Other net   PACKAGE_PIN AF10     - GND                       Bank 999 - ADC_GND
#Other net   PACKAGE_PIN AF3      - GND                       Bank 999 - ADC_GND
#Other net   PACKAGE_PIN AF6      - GND                       Bank 999 - ADC_GND
#Other net   PACKAGE_PIN AF8      - GND                       Bank 999 - ADC_GND
#Other net   PACKAGE_PIN AG1      - GND                       Bank 999 - ADC_GND
#Other net   PACKAGE_PIN AG10     - GND                       Bank 999 - ADC_GND
#Other net   PACKAGE_PIN AG2      - GND                       Bank 999 - ADC_GND
#Other net   PACKAGE_PIN AG3      - GND                       Bank 999 - ADC_GND
#Other net   PACKAGE_PIN AG4      - GND                       Bank 999 - ADC_GND
#Other net   PACKAGE_PIN AG5      - GND                       Bank 999 - ADC_GND
#Other net   PACKAGE_PIN AG6      - GND                       Bank 999 - ADC_GND
#Other net   PACKAGE_PIN AG8      - GND                       Bank 999 - ADC_GND
#Other net   PACKAGE_PIN AH10     - GND                       Bank 999 - ADC_GND
#Other net   PACKAGE_PIN AH3      - GND                       Bank 999 - ADC_GND
#Other net   PACKAGE_PIN AH6      - GND                       Bank 999 - ADC_GND
#Other net   PACKAGE_PIN AH8      - GND                       Bank 999 - ADC_GND
#Other net   PACKAGE_PIN AJ1      - GND                       Bank 999 - ADC_GND
#Other net   PACKAGE_PIN AJ10     - GND                       Bank 999 - ADC_GND
#Other net   PACKAGE_PIN AJ2      - GND                       Bank 999 - ADC_GND
#Other net   PACKAGE_PIN AJ3      - GND                       Bank 999 - ADC_GND
#Other net   PACKAGE_PIN AJ6      - GND                       Bank 999 - ADC_GND
#Other net   PACKAGE_PIN AJ8      - GND                       Bank 999 - ADC_GND
#Other net   PACKAGE_PIN AK3      - GND                       Bank 999 - ADC_GND
#Other net   PACKAGE_PIN AL1      - GND                       Bank 999 - ADC_GND
#Other net   PACKAGE_PIN AL2      - GND                       Bank 999 - ADC_GND
#Other net   PACKAGE_PIN AL3      - GND                       Bank 999 - ADC_GND
#Other net   PACKAGE_PIN AM3      - GND                       Bank 999 - ADC_GND
#Other net   PACKAGE_PIN AM4      - GND                       Bank 999 - ADC_GND
#Other net   PACKAGE_PIN AM5      - GND                       Bank 999 - ADC_GND
#Other net   PACKAGE_PIN AN1      - GND                       Bank 999 - ADC_GND
#Other net   PACKAGE_PIN AN2      - GND                       Bank 999 - ADC_GND
#Other net   PACKAGE_PIN AN3      - GND                       Bank 999 - ADC_GND
#Other net   PACKAGE_PIN AP3      - GND                       Bank 999 - ADC_GND
#Other net   PACKAGE_PIN AR1      - GND                       Bank 999 - ADC_GND
#Other net   PACKAGE_PIN AR2      - GND                       Bank 999 - ADC_GND
#Other net   PACKAGE_PIN AR3      - GND                       Bank 999 - ADC_GND
#Other net   PACKAGE_PIN W1       - GND                       Bank 999 - ADC_GND
#Other net   PACKAGE_PIN W2       - GND                       Bank 999 - ADC_GND
#Other net   PACKAGE_PIN W3       - GND                       Bank 999 - ADC_GND
#Other net   PACKAGE_PIN W4       - GND                       Bank 999 - ADC_GND
#Other net   PACKAGE_PIN W5       - GND                       Bank 999 - ADC_GND
#Other net   PACKAGE_PIN W6       - GND                       Bank 999 - ADC_GND
#Other net   PACKAGE_PIN Y3       - GND                       Bank 999 - ADC_GND
#Other net   PACKAGE_PIN Y6       - GND                       Bank 999 - ADC_GND
#Other net   PACKAGE_PIN AE10     - GND                       Bank 999 - ADC_SUB_GND
#Other net   PACKAGE_PIN M9       - DAC_AVCC                  Bank 999 - DAC_AVCC
#Other net   PACKAGE_PIN N9       - DAC_AVCC                  Bank 999 - DAC_AVCC
#Other net   PACKAGE_PIN P9       - DAC_AVCC                  Bank 999 - DAC_AVCC
#Other net   PACKAGE_PIN R9       - DAC_AVCC                  Bank 999 - DAC_AVCC
#Other net   PACKAGE_PIN T9       - DAC_AVCC                  Bank 999 - DAC_AVCC
#Other net   PACKAGE_PIN U9       - DAC_AVCC                  Bank 999 - DAC_AVCC
#Other net   PACKAGE_PIN V9       - DAC_AVCC                  Bank 999 - DAC_AVCC
#Other net   PACKAGE_PIN W9       - DAC_AVCC                  Bank 999 - DAC_AVCC
#Other net   PACKAGE_PIN L6       - DAC_AVCCAUX               Bank 999 - DAC_AVCCAUX
#Other net   PACKAGE_PIN L7       - DAC_AVCCAUX               Bank 999 - DAC_AVCCAUX
#Other net   PACKAGE_PIN M7       - DAC_AVCCAUX               Bank 999 - DAC_AVCCAUX
#Other net   PACKAGE_PIN N7       - DAC_AVCCAUX               Bank 999 - DAC_AVCCAUX
#Other net   PACKAGE_PIN P7       - DAC_AVCCAUX               Bank 999 - DAC_AVCCAUX
#Other net   PACKAGE_PIN R7       - DAC_AVCCAUX               Bank 999 - DAC_AVCCAUX
#Other net   PACKAGE_PIN T7       - DAC_AVCCAUX               Bank 999 - DAC_AVCCAUX
#Other net   PACKAGE_PIN U7       - DAC_AVCCAUX               Bank 999 - DAC_AVCCAUX
#Other net   PACKAGE_PIN H4       - DAC_AVTT                  Bank 999 - DAC_AVTT
#Other net   PACKAGE_PIN J4       - DAC_AVTT                  Bank 999 - DAC_AVTT
#Other net   PACKAGE_PIN K4       - DAC_AVTT                  Bank 999 - DAC_AVTT
#Other net   PACKAGE_PIN L4       - DAC_AVTT                  Bank 999 - DAC_AVTT
#Other net   PACKAGE_PIN A2       - GND                       Bank 999 - DAC_GND
#Other net   PACKAGE_PIN B1       - GND                       Bank 999 - DAC_GND
#Other net   PACKAGE_PIN B2       - GND                       Bank 999 - DAC_GND
#Other net   PACKAGE_PIN B3       - GND                       Bank 999 - DAC_GND
#Other net   PACKAGE_PIN C3       - GND                       Bank 999 - DAC_GND
#Other net   PACKAGE_PIN D1       - GND                       Bank 999 - DAC_GND
#Other net   PACKAGE_PIN D2       - GND                       Bank 999 - DAC_GND
#Other net   PACKAGE_PIN D3       - GND                       Bank 999 - DAC_GND
#Other net   PACKAGE_PIN E3       - GND                       Bank 999 - DAC_GND
#Other net   PACKAGE_PIN E4       - GND                       Bank 999 - DAC_GND
#Other net   PACKAGE_PIN F1       - GND                       Bank 999 - DAC_GND
#Other net   PACKAGE_PIN F2       - GND                       Bank 999 - DAC_GND
#Other net   PACKAGE_PIN F3       - GND                       Bank 999 - DAC_GND
#Other net   PACKAGE_PIN F4       - GND                       Bank 999 - DAC_GND
#Other net   PACKAGE_PIN G3       - GND                       Bank 999 - DAC_GND
#Other net   PACKAGE_PIN G4       - GND                       Bank 999 - DAC_GND
#Other net   PACKAGE_PIN H1       - GND                       Bank 999 - DAC_GND
#Other net   PACKAGE_PIN H2       - GND                       Bank 999 - DAC_GND
#Other net   PACKAGE_PIN H3       - GND                       Bank 999 - DAC_GND
#Other net   PACKAGE_PIN J3       - GND                       Bank 999 - DAC_GND
#Other net   PACKAGE_PIN K1       - GND                       Bank 999 - DAC_GND
#Other net   PACKAGE_PIN K2       - GND                       Bank 999 - DAC_GND
#Other net   PACKAGE_PIN K3       - GND                       Bank 999 - DAC_GND
#Other net   PACKAGE_PIN K5       - GND                       Bank 999 - DAC_GND
#Other net   PACKAGE_PIN L3       - GND                       Bank 999 - DAC_GND
#Other net   PACKAGE_PIN L5       - GND                       Bank 999 - DAC_GND
#Other net   PACKAGE_PIN M1       - GND                       Bank 999 - DAC_GND
#Other net   PACKAGE_PIN M10      - GND                       Bank 999 - DAC_GND
#Other net   PACKAGE_PIN M2       - GND                       Bank 999 - DAC_GND
#Other net   PACKAGE_PIN M3       - GND                       Bank 999 - DAC_GND
#Other net   PACKAGE_PIN M4       - GND                       Bank 999 - DAC_GND
#Other net   PACKAGE_PIN M5       - GND                       Bank 999 - DAC_GND
#Other net   PACKAGE_PIN M6       - GND                       Bank 999 - DAC_GND
#Other net   PACKAGE_PIN M8       - GND                       Bank 999 - DAC_GND
#Other net   PACKAGE_PIN N10      - GND                       Bank 999 - DAC_GND
#Other net   PACKAGE_PIN N3       - GND                       Bank 999 - DAC_GND
#Other net   PACKAGE_PIN N6       - GND                       Bank 999 - DAC_GND
#Other net   PACKAGE_PIN N8       - GND                       Bank 999 - DAC_GND
#Other net   PACKAGE_PIN P1       - GND                       Bank 999 - DAC_GND
#Other net   PACKAGE_PIN P2       - GND                       Bank 999 - DAC_GND
#Other net   PACKAGE_PIN P3       - GND                       Bank 999 - DAC_GND
#Other net   PACKAGE_PIN P4       - GND                       Bank 999 - DAC_GND
#Other net   PACKAGE_PIN P5       - GND                       Bank 999 - DAC_GND
#Other net   PACKAGE_PIN P6       - GND                       Bank 999 - DAC_GND
#Other net   PACKAGE_PIN P8       - GND                       Bank 999 - DAC_GND
#Other net   PACKAGE_PIN R10      - GND                       Bank 999 - DAC_GND
#Other net   PACKAGE_PIN R3       - GND                       Bank 999 - DAC_GND
#Other net   PACKAGE_PIN R6       - GND                       Bank 999 - DAC_GND
#Other net   PACKAGE_PIN R8       - GND                       Bank 999 - DAC_GND
#Other net   PACKAGE_PIN T1       - GND                       Bank 999 - DAC_GND
#Other net   PACKAGE_PIN T10      - GND                       Bank 999 - DAC_GND
#Other net   PACKAGE_PIN T2       - GND                       Bank 999 - DAC_GND
#Other net   PACKAGE_PIN T3       - GND                       Bank 999 - DAC_GND
#Other net   PACKAGE_PIN T4       - GND                       Bank 999 - DAC_GND
#Other net   PACKAGE_PIN T5       - GND                       Bank 999 - DAC_GND
#Other net   PACKAGE_PIN T6       - GND                       Bank 999 - DAC_GND
#Other net   PACKAGE_PIN T8       - GND                       Bank 999 - DAC_GND
#Other net   PACKAGE_PIN U10      - GND                       Bank 999 - DAC_GND
#Other net   PACKAGE_PIN U3       - GND                       Bank 999 - DAC_GND
#Other net   PACKAGE_PIN U6       - GND                       Bank 999 - DAC_GND
#Other net   PACKAGE_PIN U8       - GND                       Bank 999 - DAC_GND
#Other net   PACKAGE_PIN V1       - GND                       Bank 999 - DAC_GND
#Other net   PACKAGE_PIN V10      - GND                       Bank 999 - DAC_GND
#Other net   PACKAGE_PIN V2       - GND                       Bank 999 - DAC_GND
#Other net   PACKAGE_PIN V3       - GND                       Bank 999 - DAC_GND
#Other net   PACKAGE_PIN V4       - GND                       Bank 999 - DAC_GND
#Other net   PACKAGE_PIN V5       - GND                       Bank 999 - DAC_GND
#Other net   PACKAGE_PIN V6       - GND                       Bank 999 - DAC_GND
#Other net   PACKAGE_PIN V7       - GND                       Bank 999 - DAC_GND
#Other net   PACKAGE_PIN V8       - GND                       Bank 999 - DAC_GND
#Other net   PACKAGE_PIN W10      - GND                       Bank 999 - DAC_GND
#Other net   PACKAGE_PIN W7       - GND                       Bank 999 - DAC_GND
#Other net   PACKAGE_PIN Y10      - GND                       Bank 999 - DAC_GND
#Other net   PACKAGE_PIN Y7       - GND                       Bank 999 - DAC_GND
#Other net   PACKAGE_PIN Y8       - GND                       Bank 999 - DAC_GND
#Other net   PACKAGE_PIN Y9       - GND                       Bank 999 - DAC_GND
#Other net   PACKAGE_PIN P10      - GND                       Bank 999 - DAC_SUB_GND
#Other net   PACKAGE_PIN A13      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN A18      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN A23      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN A28      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN A3       - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN A30      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN A32      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN A35      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN A36      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN A37      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN A38      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN A4       - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN A8       - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AA12     - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AA14     - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AA18     - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AA21     - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AA23     - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AA24     - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AA28     - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AA30     - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AA31     - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AA32     - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AA36     - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AA37     - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AB11     - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AB13     - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AB15     - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AB17     - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AB19     - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AB21     - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AB23     - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AB25     - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AB32     - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AB33     - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AB34     - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AB35     - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AB36     - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AB37     - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AB38     - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AB39     - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AC12     - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AC14     - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AC16     - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AC18     - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AC20     - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AC21     - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AC23     - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AC24     - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AC33     - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AC37     - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AD11     - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AD13     - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AD15     - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AD17     - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AD19     - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AD21     - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AD23     - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AD27     - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AD33     - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AD35     - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AD38     - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AD39     - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AE12     - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AE14     - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AE16     - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AE18     - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AE21     - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AE23     - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AE24     - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AE25     - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AE33     - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AE37     - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AF11     - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AF13     - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AF18     - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AF21     - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AF23     - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AF28     - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AF33     - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AF35     - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AF38     - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AF39     - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AG16     - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AG22     - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AG24     - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AG26     - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AG31     - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AG33     - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AG37     - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AH11     - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AH19     - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AH21     - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AH23     - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AH29     - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AH33     - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AH35     - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AH38     - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AH39     - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AJ22     - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AJ24     - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AJ27     - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AJ33     - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AJ37     - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AK10     - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AK11     - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AK15     - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AK25     - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AK30     - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AK33     - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AK35     - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AK38     - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AK39     - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AK6      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AK7      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AK8      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AK9      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AL13     - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AL18     - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AL23     - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AL28     - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AL33     - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AL34     - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AL35     - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AL36     - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AL37     - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AL38     - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AL6      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AM21     - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AM26     - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AM6      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AN14     - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AN24     - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AN29     - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AN34     - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AN39     - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AN4      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AN5      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AN6      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AN9      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AP12     - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AP17     - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AP22     - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AP37     - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AP4      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AR20     - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AR25     - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AR30     - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AR35     - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AR4      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AR5      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AT1      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AT13     - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AT2      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AT23     - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AT3      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AT38     - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AT4      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AT8      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AU11     - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AU16     - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AU21     - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AU26     - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AU31     - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AU36     - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AV1      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AV14     - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AV19     - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AV24     - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AV39     - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AV4      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AV9      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AW12     - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AW17     - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AW2      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AW22     - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AW27     - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AW32     - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AW38     - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AW7      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN B11      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN B16      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN B21      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN B30      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN B34      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN B35      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN B38      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN B39      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN B4       - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN C14      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN C19      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN C24      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN C30      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN C32      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN C35      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN C36      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN C37      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN C4       - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN D17      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN D27      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN D30      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN D34      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN D35      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN D38      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN D39      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN D4       - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN D5       - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN D7       - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN E10      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN E20      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN E30      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN E32      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN E35      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN E36      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN E37      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN E5       - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN F13      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN F23      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN F28      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN F30      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN F34      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN F35      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN F38      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN F39      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN F5       - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN F8       - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN G16      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN G26      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN G30      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN G32      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN G35      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN G36      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN G37      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN G5       - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN H19      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN H30      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN H34      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN H35      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN H38      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN H39      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN H5       - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN H9       - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN J12      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN J22      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN J30      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN J32      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN J35      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN J36      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN J37      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN J5       - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN J6       - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN K15      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN K25      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN K30      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN K33      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN K35      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN K38      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN K39      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN K6       - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN K7       - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN K8       - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN L10      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN L11      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN L13      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN L18      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN L28      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN L30      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN L32      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN L35      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN L36      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN L37      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN L8       - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN L9       - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN M11      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN M16      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN M21      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN M23      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN M25      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN M30      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN M34      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN M35      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN M38      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN M39      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN N12      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN N23      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN N24      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN N30      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN N32      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN N35      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN N36      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN N37      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN P11      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN P13      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN P15      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN P17      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN P19      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN P21      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN P23      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN P25      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN P27      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN P30      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN P34      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN P37      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN P38      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN P39      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN R12      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN R14      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN R16      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN R18      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN R21      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN R23      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN R24      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN R30      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN R32      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN R36      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN R37      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN T11      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN T13      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN T15      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN T17      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN T19      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN T21      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN T23      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN T25      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN T28      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN T30      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN T34      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN T37      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN T38      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN T39      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN U12      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN U14      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN U16      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN U18      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN U21      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN U23      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN U24      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN U30      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN U32      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN U36      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN U37      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN V11      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN V13      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN V15      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN V19      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN V21      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN V23      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN V25      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN V30      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN V34      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN V37      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN V38      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN V39      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN W12      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN W14      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN W18      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN W21      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN W23      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN W24      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN W30      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN W32      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN W36      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN W37      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN Y11      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN Y13      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN Y15      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN Y19      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN Y21      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN Y23      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN Y25      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN Y30      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN Y34      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN Y37      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN Y38      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN Y39      - GND                       Bank 999 - GND
#Other net   PACKAGE_PIN AD34     - MGTRAVCC                  Bank 999 - PS_MGTRAVCC
#Other net   PACKAGE_PIN AF34     - MGTRAVCC                  Bank 999 - PS_MGTRAVCC
#Other net   PACKAGE_PIN AH34     - MGTRAVCC                  Bank 999 - PS_MGTRAVCC
#Other net   PACKAGE_PIN AC36     - MGT1V8                    Bank 999 - PS_MGTRAVTT
#Other net   PACKAGE_PIN AE36     - MGT1V8                    Bank 999 - PS_MGTRAVTT
#Other net   PACKAGE_PIN AG36     - MGT1V8                    Bank 999 - PS_MGTRAVTT
#Other net   PACKAGE_PIN AJ36     - MGT1V8                    Bank 999 - PS_MGTRAVTT
#Other net   PACKAGE_PIN P12      - VCC1V8                    Bank 999 - VCCAUX
#Other net   PACKAGE_PIN T12      - VCC1V8                    Bank 999 - VCCAUX
#Other net   PACKAGE_PIN V12      - VCC1V8                    Bank 999 - VCCAUX
#Other net   PACKAGE_PIN AB12     - VCC1V8                    Bank 999 - VCCAUX_IO
#Other net   PACKAGE_PIN AD12     - VCC1V8                    Bank 999 - VCCAUX_IO
#Other net   PACKAGE_PIN Y12      - VCC1V8                    Bank 999 - VCCAUX_IO
#Other net   PACKAGE_PIN AD20     - VCCINT_IO_BRAM_PS_SDFEC   Bank 999 - VCCBRAM
#Other net   PACKAGE_PIN AE19     - VCCINT_IO_BRAM_PS_SDFEC   Bank 999 - VCCBRAM
#Other net   PACKAGE_PIN AE20     - VCCINT_IO_BRAM_PS_SDFEC   Bank 999 - VCCBRAM
#Other net   PACKAGE_PIN AA13     - VCCINT                    Bank 999 - VCCINT
#Other net   PACKAGE_PIN AA15     - VCCINT                    Bank 999 - VCCINT
#Other net   PACKAGE_PIN AA19     - VCCINT                    Bank 999 - VCCINT
#Other net   PACKAGE_PIN AA20     - VCCINT                    Bank 999 - VCCINT
#Other net   PACKAGE_PIN AA25     - VCCINT                    Bank 999 - VCCINT
#Other net   PACKAGE_PIN AB14     - VCCINT                    Bank 999 - VCCINT
#Other net   PACKAGE_PIN AB16     - VCCINT                    Bank 999 - VCCINT
#Other net   PACKAGE_PIN AB18     - VCCINT                    Bank 999 - VCCINT
#Other net   PACKAGE_PIN AB20     - VCCINT                    Bank 999 - VCCINT
#Other net   PACKAGE_PIN AB24     - VCCINT                    Bank 999 - VCCINT
#Other net   PACKAGE_PIN AC13     - VCCINT                    Bank 999 - VCCINT
#Other net   PACKAGE_PIN AC15     - VCCINT                    Bank 999 - VCCINT
#Other net   PACKAGE_PIN AC17     - VCCINT                    Bank 999 - VCCINT
#Other net   PACKAGE_PIN AC19     - VCCINT                    Bank 999 - VCCINT
#Other net   PACKAGE_PIN AD14     - VCCINT                    Bank 999 - VCCINT
#Other net   PACKAGE_PIN AD16     - VCCINT                    Bank 999 - VCCINT
#Other net   PACKAGE_PIN AD18     - VCCINT                    Bank 999 - VCCINT
#Other net   PACKAGE_PIN AD24     - VCCINT                    Bank 999 - VCCINT
#Other net   PACKAGE_PIN AE13     - VCCINT                    Bank 999 - VCCINT
#Other net   PACKAGE_PIN AE15     - VCCINT                    Bank 999 - VCCINT
#Other net   PACKAGE_PIN AE17     - VCCINT                    Bank 999 - VCCINT
#Other net   PACKAGE_PIN AF24     - VCCINT                    Bank 999 - VCCINT
#Other net   PACKAGE_PIN AH24     - VCCINT                    Bank 999 - VCCINT
#Other net   PACKAGE_PIN M24      - VCCINT                    Bank 999 - VCCINT
#Other net   PACKAGE_PIN N25      - VCCINT                    Bank 999 - VCCINT
#Other net   PACKAGE_PIN P14      - VCCINT                    Bank 999 - VCCINT
#Other net   PACKAGE_PIN P16      - VCCINT                    Bank 999 - VCCINT
#Other net   PACKAGE_PIN P18      - VCCINT                    Bank 999 - VCCINT
#Other net   PACKAGE_PIN P20      - VCCINT                    Bank 999 - VCCINT
#Other net   PACKAGE_PIN P24      - VCCINT                    Bank 999 - VCCINT
#Other net   PACKAGE_PIN R13      - VCCINT                    Bank 999 - VCCINT
#Other net   PACKAGE_PIN R15      - VCCINT                    Bank 999 - VCCINT
#Other net   PACKAGE_PIN R17      - VCCINT                    Bank 999 - VCCINT
#Other net   PACKAGE_PIN R19      - VCCINT                    Bank 999 - VCCINT
#Other net   PACKAGE_PIN R20      - VCCINT                    Bank 999 - VCCINT
#Other net   PACKAGE_PIN R25      - VCCINT                    Bank 999 - VCCINT
#Other net   PACKAGE_PIN T14      - VCCINT                    Bank 999 - VCCINT
#Other net   PACKAGE_PIN T16      - VCCINT                    Bank 999 - VCCINT
#Other net   PACKAGE_PIN T18      - VCCINT                    Bank 999 - VCCINT
#Other net   PACKAGE_PIN T20      - VCCINT                    Bank 999 - VCCINT
#Other net   PACKAGE_PIN T24      - VCCINT                    Bank 999 - VCCINT
#Other net   PACKAGE_PIN U13      - VCCINT                    Bank 999 - VCCINT
#Other net   PACKAGE_PIN U15      - VCCINT                    Bank 999 - VCCINT
#Other net   PACKAGE_PIN U17      - VCCINT                    Bank 999 - VCCINT
#Other net   PACKAGE_PIN U19      - VCCINT                    Bank 999 - VCCINT
#Other net   PACKAGE_PIN U20      - VCCINT                    Bank 999 - VCCINT
#Other net   PACKAGE_PIN U25      - VCCINT                    Bank 999 - VCCINT
#Other net   PACKAGE_PIN V14      - VCCINT                    Bank 999 - VCCINT
#Other net   PACKAGE_PIN V18      - VCCINT                    Bank 999 - VCCINT
#Other net   PACKAGE_PIN V20      - VCCINT                    Bank 999 - VCCINT
#Other net   PACKAGE_PIN V24      - VCCINT                    Bank 999 - VCCINT
#Other net   PACKAGE_PIN W13      - VCCINT                    Bank 999 - VCCINT
#Other net   PACKAGE_PIN W15      - VCCINT                    Bank 999 - VCCINT
#Other net   PACKAGE_PIN W19      - VCCINT                    Bank 999 - VCCINT
#Other net   PACKAGE_PIN W20      - VCCINT                    Bank 999 - VCCINT
#Other net   PACKAGE_PIN W25      - VCCINT                    Bank 999 - VCCINT
#Other net   PACKAGE_PIN Y14      - VCCINT                    Bank 999 - VCCINT
#Other net   PACKAGE_PIN Y18      - VCCINT                    Bank 999 - VCCINT
#Other net   PACKAGE_PIN Y20      - VCCINT                    Bank 999 - VCCINT
#Other net   PACKAGE_PIN Y24      - VCCINT                    Bank 999 - VCCINT
#Other net   PACKAGE_PIN AA11     - VCCINT_AMS                Bank 999 - VCCINT_AMS
#Other net   PACKAGE_PIN AC11     - VCCINT_AMS                Bank 999 - VCCINT_AMS
#Other net   PACKAGE_PIN AE11     - VCCINT_AMS                Bank 999 - VCCINT_AMS
#Other net   PACKAGE_PIN AG11     - VCCINT_AMS                Bank 999 - VCCINT_AMS
#Other net   PACKAGE_PIN AJ11     - VCCINT_AMS                Bank 999 - VCCINT_AMS
#Other net   PACKAGE_PIN N11      - VCCINT_AMS                Bank 999 - VCCINT_AMS
#Other net   PACKAGE_PIN R11      - VCCINT_AMS                Bank 999 - VCCINT_AMS
#Other net   PACKAGE_PIN U11      - VCCINT_AMS                Bank 999 - VCCINT_AMS
#Other net   PACKAGE_PIN W11      - VCCINT_AMS                Bank 999 - VCCINT_AMS
#Other net   PACKAGE_PIN AG21     - VCCINT_IO_BRAM_PS_SDFEC   Bank 999 - VCCINT_IO
#Other net   PACKAGE_PIN AJ21     - VCCINT_IO_BRAM_PS_SDFEC   Bank 999 - VCCINT_IO
#Other net   PACKAGE_PIN AA22     - VCCINT_IO_BRAM_PS_SDFEC   Bank 999 - VCCSDFEC
#Other net   PACKAGE_PIN AB22     - VCCINT_IO_BRAM_PS_SDFEC   Bank 999 - VCCSDFEC
#Other net   PACKAGE_PIN AC22     - VCCINT_IO_BRAM_PS_SDFEC   Bank 999 - VCCSDFEC
#Other net   PACKAGE_PIN AD22     - VCCINT_IO_BRAM_PS_SDFEC   Bank 999 - VCCSDFEC
#Other net   PACKAGE_PIN AE22     - VCCINT_IO_BRAM_PS_SDFEC   Bank 999 - VCCSDFEC
#Other net   PACKAGE_PIN AF22     - VCCINT_IO_BRAM_PS_SDFEC   Bank 999 - VCCSDFEC
#Other net   PACKAGE_PIN AG23     - VCCINT_IO_BRAM_PS_SDFEC   Bank 999 - VCCSDFEC
#Other net   PACKAGE_PIN AH22     - VCCINT_IO_BRAM_PS_SDFEC   Bank 999 - VCCSDFEC
#Other net   PACKAGE_PIN AJ23     - VCCINT_IO_BRAM_PS_SDFEC   Bank 999 - VCCSDFEC
#Other net   PACKAGE_PIN M22      - VCCINT_IO_BRAM_PS_SDFEC   Bank 999 - VCCSDFEC
#Other net   PACKAGE_PIN N22      - VCCINT_IO_BRAM_PS_SDFEC   Bank 999 - VCCSDFEC
#Other net   PACKAGE_PIN P22      - VCCINT_IO_BRAM_PS_SDFEC   Bank 999 - VCCSDFEC
#Other net   PACKAGE_PIN R22      - VCCINT_IO_BRAM_PS_SDFEC   Bank 999 - VCCSDFEC
#Other net   PACKAGE_PIN T22      - VCCINT_IO_BRAM_PS_SDFEC   Bank 999 - VCCSDFEC
#Other net   PACKAGE_PIN U22      - VCCINT_IO_BRAM_PS_SDFEC   Bank 999 - VCCSDFEC
#Other net   PACKAGE_PIN V22      - VCCINT_IO_BRAM_PS_SDFEC   Bank 999 - VCCSDFEC
#Other net   PACKAGE_PIN W22      - VCCINT_IO_BRAM_PS_SDFEC   Bank 999 - VCCSDFEC
#Other net   PACKAGE_PIN Y22      - VCCINT_IO_BRAM_PS_SDFEC   Bank 999 - VCCSDFEC
#Other net   PACKAGE_PIN AC26     - VCC1V8                    Bank 999 - VCC_PSAUX
#Other net   PACKAGE_PIN AC27     - VCC1V8                    Bank 999 - VCC_PSAUX
#Other net   PACKAGE_PIN AC28     - VCC1V8                    Bank 999 - VCC_PSAUX
#Other net   PACKAGE_PIN AD29     - VCC1V8                    Bank 999 - VCC_PSAUX
#Other net   PACKAGE_PIN AC25     - VCC_PSBATT                Bank 999 - VCC_PSBATT
#Other net   PACKAGE_PIN AD25     - VCC_PSDDR_PLL             Bank 999 - VCC_PSDDR_PLL
#Other net   PACKAGE_PIN AD26     - VCC_PSDDR_PLL             Bank 999 - VCC_PSDDR_PLL
#Other net   PACKAGE_PIN AF26     - VCC_PSDDR_PLL             Bank 999 - VCC_PSDDR_PLL
#Other net   PACKAGE_PIN AF25     - VCCINT_IO_BRAM_PS_SDFEC   Bank 999 - VCC_PSINTFP
#Other net   PACKAGE_PIN AG25     - VCCINT_IO_BRAM_PS_SDFEC   Bank 999 - VCC_PSINTFP
#Other net   PACKAGE_PIN AH25     - VCCINT_IO_BRAM_PS_SDFEC   Bank 999 - VCC_PSINTFP
#Other net   PACKAGE_PIN AH26     - VCCINT_IO_BRAM_PS_SDFEC   Bank 999 - VCC_PSINTFP
#Other net   PACKAGE_PIN AH27     - VCCINT_IO_BRAM_PS_SDFEC   Bank 999 - VCC_PSINTFP
#Other net   PACKAGE_PIN AJ25     - VCCINT_IO_BRAM_PS_SDFEC   Bank 999 - VCC_PSINTFP
#Other net   PACKAGE_PIN AJ26     - VCCINT_IO_BRAM_PS_SDFEC   Bank 999 - VCC_PSINTFP
#Other net   PACKAGE_PIN AG29     - VCCINT_IO_BRAM_PS_SDFEC   Bank 999 - VCC_PSINTFP_DDR
#Other net   PACKAGE_PIN AH28     - VCCINT_IO_BRAM_PS_SDFEC   Bank 999 - VCC_PSINTFP_DDR
#Other net   PACKAGE_PIN AJ28     - VCCINT_IO_BRAM_PS_SDFEC   Bank 999 - VCC_PSINTFP_DDR
#Other net   PACKAGE_PIN AJ29     - VCCINT_IO_BRAM_PS_SDFEC   Bank 999 - VCC_PSINTFP_DDR
#Other net   PACKAGE_PIN AE26     - VCCINT_IO_BRAM_PS_SDFEC   Bank 999 - VCC_PSINTLP
#Other net   PACKAGE_PIN AF27     - VCCINT_IO_BRAM_PS_SDFEC   Bank 999 - VCC_PSINTLP
#Other net   PACKAGE_PIN AG27     - VCCINT_IO_BRAM_PS_SDFEC   Bank 999 - VCC_PSINTLP
#Other net   PACKAGE_PIN AG28     - VCCINT_IO_BRAM_PS_SDFEC   Bank 999 - VCC_PSINTLP
#Other net   PACKAGE_PIN AD28     - MGT1V2                    Bank 999 - VCC_PSPLL
#Other net   PACKAGE_PIN AE27     - MGT1V2                    Bank 999 - VCC_PSPLL
#Other net   PACKAGE_PIN AE28     - MGT1V2                    Bank 999 - VCC_PSPLL
#Other net   PACKAGE_PIN A31      - MGT1V2                    Bank 999 - MGTAVTT_L
#Other net   PACKAGE_PIN AA35     - MGT1V2                    Bank 999 - MGTAVTT_L
#Other net   PACKAGE_PIN B33      - MGT1V2                    Bank 999 - MGTAVTT_L
#Other net   PACKAGE_PIN C31      - MGT1V2                    Bank 999 - MGTAVTT_L
#Other net   PACKAGE_PIN D33      - MGT1V2                    Bank 999 - MGTAVTT_L
#Other net   PACKAGE_PIN E31      - MGT1V2                    Bank 999 - MGTAVTT_L
#Other net   PACKAGE_PIN F33      - MGT1V2                    Bank 999 - MGTAVTT_L
#Other net   PACKAGE_PIN G31      - MGT1V2                    Bank 999 - MGTAVTT_L
#Other net   PACKAGE_PIN H33      - MGT1V2                    Bank 999 - MGTAVTT_L
#Other net   PACKAGE_PIN J31      - MGT1V2                    Bank 999 - MGTAVTT_L
#Other net   PACKAGE_PIN K34      - MGT1V2                    Bank 999 - MGTAVTT_L
#Other net   PACKAGE_PIN P33      - MGT1V2                    Bank 999 - MGTAVTT_L
#Other net   PACKAGE_PIN R35      - MGT1V2                    Bank 999 - MGTAVTT_L
#Other net   PACKAGE_PIN T33      - MGT1V2                    Bank 999 - MGTAVTT_L
#Other net   PACKAGE_PIN U35      - MGT1V2                    Bank 999 - MGTAVTT_L
#Other net   PACKAGE_PIN W35      - MGT1V2                    Bank 999 - MGTAVTT_L
#Other net   PACKAGE_PIN L31      - MGTAVCC                   Bank 999 - MGTAVCC_L
#Other net   PACKAGE_PIN M33      - MGTAVCC                   Bank 999 - MGTAVCC_L
#Other net   PACKAGE_PIN N31      - MGTAVCC                   Bank 999 - MGTAVCC_L
#Other net   PACKAGE_PIN V33      - MGTAVCC                   Bank 999 - MGTAVCC_L
#Other net   PACKAGE_PIN W31      - MGTAVCC                   Bank 999 - MGTAVCC_L
#Other net   PACKAGE_PIN Y33      - MGTAVCC                   Bank 999 - MGTAVCC_L
#Other net   PACKAGE_PIN R31      - MGT1V8                    Bank 999 - MGTVCCAUX_L
#Other net   PACKAGE_PIN U31      - MGT1V8                    Bank 999 - MGTVCCAUX_L