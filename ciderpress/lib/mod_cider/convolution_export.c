// CiderPress: GPL-3.0-or-later; see convolutions.h for the project license.
#include "convolutions.h"
#include <stdint.h>
#include <stddef.h>
#include <string.h>

/*
 * Owned snapshot ABI v1 for a live version-j convolution_collection.
 *
 * int cider_export_atc_v1(const convolution_collection *ccl,
 *     const atc_basis_set *expected_in, const atc_basis_set *expected_out,
 *     size_t *meta, size_t meta_count, int *blocks, size_t block_capacity,
 *     double *data, size_t data_capacity);
 *
 * expected_in/out must be the caller's basis handles; identity is checked
 * without exporting internal pointers. All capacities count elements, not bytes. meta needs 9 size_t elements:
 * [ABI version=1, natm, nalpha, nbeta, shell pairs, block count,
 *  data count, input nao, output nao]. Each block has 11 ints:
 * [atom, l, input shell start/stop, input AO start/stop,
 *  output shell start/stop, output AO start/stop, shell-pair offset].
 * Blocks are in atom/l order. data is [alphas, alpha_norms, ovlp_mats];
 * the last segment has shell_pairs * nbeta * nalpha doubles, in native
 * (output shell, input shell, beta, alpha) order within each block.
 *
 * Query: blocks=data=NULL and both capacities=0 writes only meta.
 * Copy: both output pointers must be non-NULL and capacities sufficient.
 * Return 0 on success, 1 for NULL/invalid arguments, 2 for short capacity,
 * 3 for unsupported features or inconsistent native dimensions/indexes,
 * 4 for size_t overflow. On a short output capacity only meta is written;
 * otherwise errors leave all outputs unchanged. Buffers must not overlap
 * each other or ccl storage. The function allocates nothing, retains no
 * pointers, and neither modifies nor frees ccl. The caller owns all copies.
 *
 * ccl must remain alive and unmodified for query and copy. The caller must
 * have generated integrals and solved projection coefficients first: the
 * native object has no solved flag, so this function cannot certify that.
 */
int cider_export_atc_v1(const convolution_collection *ccl,
                        const atc_basis_set *expected_in,
                        const atc_basis_set *expected_out, size_t *meta,
                        size_t meta_count, int *blocks, size_t block_capacity,
                        double *data, size_t data_capacity) {
    size_t count = 0, pairs = 0, stride, values;
    int ia, l;
    if (!ccl || !meta || !expected_in || !expected_out) return 1;
    if (meta_count < 9) return 2;
    if (ccl->atco_inp != expected_in || ccl->atco_out != expected_out) return 3;
    if (ccl->natm <= 0 || ccl->nalpha <= 0 || ccl->tot_shl_pairs <= 0 ||
        ccl->nfeat_i != 0 || ccl->has_vj != 1 ||
        ccl->nbeta != ccl->nalpha || !ccl->atco_inp || !ccl->atco_out ||
        !ccl->pair_loc || !ccl->alphas || !ccl->alpha_norms || !ccl->ovlp_mats)
        return 3;
    const atc_basis_set *in = ccl->atco_inp, *out = ccl->atco_out;
    if (in->natm != ccl->natm || out->natm != ccl->natm ||
        in->nao <= 0 || out->nao <= 0 || !in->atc_convs || !out->atc_convs ||
        !in->ao_loc || !out->ao_loc) return 3;
    for (ia = 0; ia < ccl->natm; ++ia) {
        const atc_atom *a = in->atc_convs + ia, *b = out->atc_convs + ia;
        if (a->lmax < 0 || a->lmax != b->lmax || !a->global_l_loc ||
            !b->global_l_loc || !ccl->pair_loc[ia]) return 3;
        for (l = 0; l <= a->lmax; ++l) {
            int i0 = a->global_l_loc[l], i1 = a->global_l_loc[l + 1];
            int j0 = b->global_l_loc[l], j1 = b->global_l_loc[l + 1];
            if (i0 < 0 || i1 < i0 || i1 > in->nbas || j0 < 0 ||
                j1 < j0 || j1 > out->nbas || ccl->pair_loc[ia][l] < 0 ||
                (size_t)ccl->pair_loc[ia][l] != pairs) return 3;
            if (i1 == i0 && j1 == j0) continue;
            if (i1 == i0 || j1 == j0) return 3;
            size_t ni = (size_t)(i1 - i0), nj = (size_t)(j1 - j0);
            if (ni > SIZE_MAX / nj || pairs > SIZE_MAX - ni * nj) return 4;
            pairs += ni * nj;
            if (pairs > (size_t)ccl->tot_shl_pairs ||
                in->ao_loc[i0] < 0 || in->ao_loc[i1] > in->nao ||
                out->ao_loc[j0] < 0 || out->ao_loc[j1] > out->nao ||
                (size_t)(in->ao_loc[i1] - in->ao_loc[i0]) != ni * (2*(size_t)l+1) ||
                (size_t)(out->ao_loc[j1] - out->ao_loc[j0]) != nj * (2*(size_t)l+1))
                return 3;
            ++count;
        }
    }
    if (pairs != (size_t)ccl->tot_shl_pairs) return 3;
    if ((size_t)ccl->nalpha > SIZE_MAX / (size_t)ccl->nbeta) return 4;
    stride = (size_t)ccl->nalpha * (size_t)ccl->nbeta;
    if (pairs > SIZE_MAX / stride) return 4;
    values = pairs * stride;
    if ((size_t)ccl->nalpha > (SIZE_MAX - values) / 2) return 4;
    values += 2 * (size_t)ccl->nalpha;
    if (values > SIZE_MAX / sizeof(double) || count > SIZE_MAX / 11) return 4;
    int query = !blocks && !data && !block_capacity && !data_capacity;
    if (!query && (!blocks || !data)) return 1;
    size_t shape[9] = {1, (size_t)ccl->natm, (size_t)ccl->nalpha,
        (size_t)ccl->nbeta, pairs, count, values, (size_t)in->nao, (size_t)out->nao};
    memcpy(meta, shape, sizeof(shape));
    if (query) return 0;
    if (block_capacity < 11 * count || data_capacity < values) return 2;
    size_t index = 0;
    for (ia = 0; ia < ccl->natm; ++ia) {
        const atc_atom *a = in->atc_convs + ia, *b = out->atc_convs + ia;
        for (l = 0; l <= a->lmax; ++l) {
            int i0 = a->global_l_loc[l], i1 = a->global_l_loc[l + 1];
            int j0 = b->global_l_loc[l], j1 = b->global_l_loc[l + 1];
            if (i0 == i1) continue;
            int row[11] = {ia, l, i0, i1, in->ao_loc[i0], in->ao_loc[i1],
                j0, j1, out->ao_loc[j0], out->ao_loc[j1], ccl->pair_loc[ia][l]};
            memcpy(blocks + 11 * index++, row, sizeof(row));
        }
    }
    size_t n = (size_t)ccl->nalpha;
    memcpy(data, ccl->alphas, n * sizeof(double));
    memcpy(data + n, ccl->alpha_norms, n * sizeof(double));
    memcpy(data + 2*n, ccl->ovlp_mats, (values - 2*n) * sizeof(double));
    return 0;
}
