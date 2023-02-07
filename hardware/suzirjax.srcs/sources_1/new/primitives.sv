`include "suzirjax.srcs/sources_1/new/types.sv"

module p_abs(
    input fp fp_in,
    output fp fp_out
);

    always_comb fp_out = {0, fp_in[$size(fp_in)-2:0]};
endmodule
