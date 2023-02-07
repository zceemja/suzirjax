// Some

`timescale 1ns / 1ps

module root(
    // Clocks
    input logic CLK_100_P, CLK_100_N,
    input logic CLK_125_P, CLK_125_N,
    input logic reset,
    // UART
    //input wire UART0_RX, UART0_CTS,
    //output wire UART0_TX, UART0_RTS,
    
    // GPIO
    output wire [7:0] GPIO_LED,
    input wire [4:0] GPIO_SW,
    input logic [7:0] GPIO_DIP_SW
);
       

    wire clk0;
    wire locked;

    clk_wiz_1 clk1_mod(
        // Clock out ports
        .clk_out1(clk0),
        // Status and control signals
        .reset(reset),
        // Clock in ports
        .clk_in1_p(CLK_125_P),
        .clk_in1_n(CLK_125_N),
        .locked(locked)
    );


    assign GPIO_LED[7:3] = GPIO_SW;
    assign GPIO_LED[0] = GPIO_DIP_SW[0];
    assign GPIO_LED[1] = locked;
    assign GPIO_LED[2] = 1;

endmodule
