#include "test.hpp"

int main()
{
    test_mlp_abstraction();
    test_cnn_abstraction();
    test_mlp();
    cnn_test();
    multi_in_single_out_channel_cnn_test();
    multi_in_multi_out_channel_cnn_test();
    return 0;
}
