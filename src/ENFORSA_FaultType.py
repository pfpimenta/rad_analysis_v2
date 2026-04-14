from enum import IntEnum

"""
The signal identifiers are
enum {
    IDX_io_in_a,   // 0 - Gemmini input A (flows left->right)
    IDX_io_in_b,   // 1 - Gemmini input B (flows top->bottom)
    IDX_io_in_d,   // 2 - Gemmini input D (flows top->bottom)
    IDX_io_out_b,  // 3 - OS: input b passes through during stream and preloads. WS: partial sums during stream and preloads
    IDX_io_out_c,  // 4 - OS: in_d in preloads, MAC out_d during stream. WS: in_d during preloads (actuall input B), MAC out_d during stream
    IDX_propagate, // 5 - propagate ctrl signals to assert in_b and in_d pins (flows top->bottom)
    IDX_valid,     // 6 - valid signals to assert in_b and in_d pins (flows top->bottom)
    IDX_io_out_a   // 7
    IDX_c1,        // 8 - the c1/c2 registers holding preloaded values - check PE.scala to understand this better
    IDX_c2         // 9
};
"""


class ENFORSA_FaultType(IntEnum):
    # Input group
    IN_A = 0
    IN_B = 1
    IN_D = 2

    # Output group
    OUT_A = 7
    OUT_B = 3
    OUT_C = 4

    # Control group
    C1 = 8
    C2 = 9

    # Signal group
    SIG_PROPAG = 5
    SIG_VALID = 6
