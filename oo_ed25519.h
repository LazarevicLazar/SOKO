#ifndef OO_ED25519_H
#define OO_ED25519_H

#include <stddef.h>
#include <stdint.h>

#define OOED25519_PK_SIZE 32
#define OOED25519_SK_SIZE 64
#define OOED25519_SIG_SIZE 64
#define OOED25519_SCALAR_SIZE 64

typedef struct {
    uint8_t r[32];
    uint8_t R[32];
} ooed25519_token;

int ooed25519_init(void);
int ooed25519_keypair(uint8_t *pk, uint8_t *sk);
int ooed25519_expand_secret(const uint8_t *sk, uint8_t *a);
int ooed25519_token_generate(ooed25519_token *token);
int ooed25519_sign_online(
    uint8_t *sig,
    const uint8_t *msg,
    size_t msg_len,
    const uint8_t *pk,
    const uint8_t *a,
    const ooed25519_token *token);
int ooed25519_verify(
    const uint8_t *sig,
    const uint8_t *msg,
    size_t msg_len,
    const uint8_t *pk);

#endif
