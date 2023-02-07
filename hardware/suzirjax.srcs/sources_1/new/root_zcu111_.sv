module root_zcu111(
    // Clocks
    input logic CLK_100_P, CLK_100_N,
    input logic CLK_125_P, CLK_125_N,
    // input logic reset,
    // UART
    //input wire UART0_RX, UART0_CTS,
    //output wire UART0_TX, UART0_RTS,

    // Other
    input wire CPU_RESET,

    // LEDS
    output wire
        GPIO_LED_0_LS,
        GPIO_LED_1_LS,
        GPIO_LED_2_LS,
        GPIO_LED_3_LS,
        GPIO_LED_4_LS,
        GPIO_LED_5_LS,
        GPIO_LED_6_LS,
        GPIO_LED_7_LS,

    // Switches
    input wire
        GPIO_SW_N,
        GPIO_SW_S,
        GPIO_SW_E,
        GPIO_SW_W,
        GPIO_SW_C,

    // DIP Switches
    output wire
        GPIO_DIP_SW0,
        GPIO_DIP_SW1,
        GPIO_DIP_SW2,
        GPIO_DIP_SW3,
        GPIO_DIP_SW4,
        GPIO_DIP_SW5,
        GPIO_DIP_SW6,
        GPIO_DIP_SW7
);


    assign wire [0:7] GPIO_LED = {
        GPIO_LED_0_LS,
        GPIO_LED_1_LS,
        GPIO_LED_2_LS,
        GPIO_LED_3_LS,
        GPIO_LED_4_LS,
        GPIO_LED_5_LS,
        GPIO_LED_6_LS,
        GPIO_LED_7_LS
    };

    assign wire [0:7] GPIO_DIP_SW = {
        GPIO_DIP_SW0,
        GPIO_DIP_SW1,
        GPIO_DIP_SW2,
        GPIO_DIP_SW3,
        GPIO_DIP_SW4,
        GPIO_DIP_SW5,
        GPIO_DIP_SW6,
        GPIO_DIP_SW7
    };

    assign wire [0:4] GPIO_SW = {
        GPIO_SW_N,
        GPIO_SW_S,
        GPIO_SW_E,
        GPIO_SW_W,
        GPIO_SW_C
    };

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

endmodule
