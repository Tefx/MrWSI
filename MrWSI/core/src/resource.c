//
// Created by zhaomengz on 11/5/17.
//

#include "../include/resource.h"

bool mr_richcmp(res_t* r0, res_t* r1, int op, int dim) {
    bool res;
    switch (op) {
        case 0:
            res = mr_le(r0, r1, dim) && !mr_eq(r0, r1, dim);
            break;
        case 1:
            res = mr_le(r0, r1, dim);
            break;
        case 2:
            res = mr_eq(r0, r1, dim);
            break;
        case 3:
            res = !mr_eq(r0, r1, dim);
            break;
        case 4:
            res = mr_le(r1, r0, dim);
            break;
        case 5:
            res = mr_le(r1, r0, dim) && ! mr_eq(r0, r1, dim);
            break;
        default:
            res = false;
    }
    return res;
}
