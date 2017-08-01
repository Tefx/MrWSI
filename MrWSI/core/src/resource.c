//
// Created by zhaomengz on 11/5/17.
//

#include "../include/resource.h"

bool mr_richcmp(res_t* r0, res_t* r1, int op) {
    bool res;
    switch (op) {
        case 0:
            res = r_le(r0[0], r1[0]) && r_le(r0[1], r1[1]) &&
                  (r_ne(r0[0], r1[0]) || r_ne(r1[1], r1[1]));
            break;
        case 1:
            res = r_le(r0[0], r1[0]) && r_le(r0[1], r1[1]);
            break;
        case 2:
            res = r_eq(r0[0], r1[0]) && r_eq(r0[1], r1[1]);
            break;
        case 3:
            res = r_ne(r0[0], r1[0]) || r_ne(r0[1], r1[1]);
            break;
        case 4:
            res = r_ge(r0[0], r1[0]) && r_ge(r0[1], r1[1]) &&
                  (r_ne(r0[0], r1[0]) || r_ne(r0[1], r1[1]));
            break;
        case 5:
            res = r_ge(r0[0], r1[0]) && r_ge(r0[1], r1[1]);
            break;
        default:
            res = false;
    }
    return res;
}
