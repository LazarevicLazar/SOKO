#include "oo_ed25519.h"

#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "submodules/OO-FN-DSA/ed25519/src/ed25519.h"

int ooed25519_init(void) {
    static int seeded = 0;
    if (!seeded) {
        srand((unsigned)time(NULL));
        seeded = 1;
    }
    return 0;
}

int ooed25519_keypair(uint8_t *pk, uint8_t *sk) {
    uint8_t seed[32];

    if (pk == NULL || sk == NULL) {
        return -1;
    }

    for (size_t i = 0; i < sizeof(seed); i++) {
        seed[i] = (uint8_t)(rand() & 0xFF);
    }

    ed25519_create_keypair(pk, sk, seed);
    return 0;
}

int ooed25519_expand_secret(const uint8_t *sk, uint8_t *a) {
    if (sk == NULL || a == NULL) {
        return -1;
    }

    memcpy(a, sk, OOED25519_SCALAR_SIZE);
    return 0;
}

int ooed25519_token_generate(ooed25519_token *token) {
    if (token == NULL) {
        return -1;
    }

    for (size_t i = 0; i < sizeof(token->r); i++) {
        token->r[i] = (uint8_t)(rand() & 0xFF);
    }
    for (size_t i = 0; i < sizeof(token->R); i++) {
        token->R[i] = (uint8_t)(rand() & 0xFF);
    }

    return 0;
}

int ooed25519_sign_online(
    uint8_t *sig,
    const uint8_t *msg,
    size_t msg_len,
    const uint8_t *pk,
    const uint8_t *a,
    const ooed25519_token *token) {
    if (sig == NULL || msg == NULL || pk == NULL || a == NULL || token == NULL) {
        return -1;
    }

    (void)token;
    ed25519_sign(sig, msg, msg_len, pk, a);

    return 0;
}

int ooed25519_verify(
    const uint8_t *sig,
    const uint8_t *msg,
    size_t msg_len,
    const uint8_t *pk) {
    if (sig == NULL || msg == NULL || pk == NULL) {
        return -1;
    }
    return ed25519_verify(sig, msg, msg_len, pk);
}
