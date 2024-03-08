#define WARP_SIZE 32
#define MATRIX_MUL_TILE 32

void inverse_permute_impl(float *input, float *output, int64_t *index, int64_t indice_num, int64_t feature_size);
void permute_impl(float *input, float *output, int64_t *index, int64_t indice_num, int64_t feature_size);
void fill_bias_impl(float *output_permuted, float *bias, int64_t *permuted_offest, int64_t *cluster_perm,
                    int64_t *batch_height, int64_t total_cluster, int64_t out_channels);
void bias_backward_impl(float *grad_output, float *grad_bias, int64_t *permuted_offest, int64_t *cluster_perm,
                        int64_t *batch_height, int64_t total_cluster, int64_t out_channels);