`include "suzirjax.srcs/sources_1/new/root.sv"

module root_tb;
    logic CLK_100_P, CLK_100_N;
    logic CLK_125_P, CLK_125_N;

    initial forever #5 CLK_100_P = ~CLK_100_P;
    initial forever #4 CLK_125_P = ~CLK_125_P;

    assign CLK_100_N = ~CLK_100_P;
    assign CLK_125_N = ~CLK_125_P;

    logic [7:0] GPIO_LED;
    logic [4:0] GPIO_SW;
    logic [7:0] GPIO_DIP_SW;
    logic reset;

    root u_root(
        .CLK_100_P(CLK_100_P),
        .CLK_100_N(CLK_100_N),
        .CLK_125_P(CLK_125_P),
        .CLK_125_N(CLK_125_N),
        .reset(reset),
        .GPIO_LED(GPIO_LED),
        .GPIO_SW(GPIO_SW),
        .GPIO_DIP_SW(GPIO_DIP_SW)
    );

    // Clock setup
    initial begin
        $display("Running root_tb");
        $dumpfile("root_tb.vcd");
        $dumpvars(4, root_tb);
        // $dumpvars(1, CLK_100_P);/*
        // $dumpvars(1, CLK_100_N);
        // $dumpvars(1, reset);

        CLK_100_P = 0;
        CLK_125_P = 0;
        reset = 0;
        #10
        reset = 1;
        #100
        reset = 0;
        #100
        $finish;
    end

endmodule
