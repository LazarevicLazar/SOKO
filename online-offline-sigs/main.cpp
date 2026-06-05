#include <string>

#include <stdlib.h>
#include "mbed.h"
#include "stm32f7xx_hal.h"
#include <malloc.h>

extern "C" {
#include "falcon-lazy/falcon.h"
#include "falcon-20201020/falcon.h"
#include "dilithium-pqm4/api.h"
#include "dilithium-pqm4/config.h"
#include "dilithium-pqm4/keccakf1600.h"
#include "dilithium-pqm4/ntt.h"
#include "dilithium-pqm4/packing.h"
#include "dilithium-pqm4/params.h"
#include "dilithium-pqm4/pointwise_mont.h"
#include "dilithium-pqm4/poly.h"
#include "dilithium-pqm4/polyvec.h"
#include "dilithium-pqm4/randombytes.h"
#include "dilithium-pqm4/reduce.h"
#include "dilithium-pqm4/rounding.h"
#include "dilithium-pqm4/sign.h"
#include "dilithium-pqm4/symmetric.h"
#include "dilithium-pqm4/vector.h"
#include "ed25519/src/ed25519.h"
#include "ed25519/src/ge.h"
#include "ed25519/src/sc.h"

}

//------------------------------------
// Hyperterminal configuration
// 115200 bauds, 8-bit data, no parity
//------------------------------------

Serial pc(SERIAL_TX, SERIAL_RX, 115200);
DigitalOut myled(LED1);
Timer timer;

static void *
xmalloc(size_t len)
{
	void *buf;

	if (len == 0) {
		return NULL;
	}
	buf = malloc(len);
	if (buf == NULL) {
		fprintf(stderr, "memory allocation error\n");
		exit(EXIT_FAILURE);
	}
	return buf;
}

int randombytes(uint8_t *obuf, size_t len)
{
	static uint32_t fibo_a = 0xDEADBEEF, fibo_b = 0x01234567;
	size_t i;
	for (i = 0; i < len; i++) {
		fibo_a += fibo_b;
		fibo_b += fibo_a;
		obuf[i] = (fibo_a >> 24) ^ (fibo_b >> 16);
	}
	return 0;
}

static void
display_mallinfo(void)
{
	struct mallinfo mi;

	mi = mallinfo();

	printf("Total non-mmapped bytes (arena):       %zu\n\r", mi.arena);
	//   printf("# of free chunks (ordblks):            %zu\n\r", mi.ordblks);
	//   printf("# of free fastbin blocks (smblks):     %zu\n\r", mi.smblks);
	//   printf("# of mapped regions (hblks):           %zu\n\r", mi.hblks);
	//   printf("Bytes in mapped regions (hblkhd):      %zu\n\r", mi.hblkhd);
	//   printf("Max. total allocated space (usmblks):  %zu\n\r", mi.usmblks);
	//   printf("Free bytes held in fastbins (fsmblks): %zu\n\r", mi.fsmblks);
	printf("Total allocated space (uordblks):      %zu\n\r", mi.uordblks);
	// printf("Total free space (fordblks):           %zu\n\r", mi.fordblks);
	//printf("Topmost releasable block (keepcost):   %zu\n\r", mi.keepcost);
}

int main()
{

	// Defs for timing calculations
	#define BENCHMARK_ROUND 1000
	uint64_t start, stop, delta, min, max;
	int us, cnt;
	long double average_us, average_clk, avclk_old, var, std_err, ddelta;

	#define MIN(a,b) (((a)<(b))?(a):(b))
	#define MAX(a,b) (((a)>(b))?(a):(b))

	#define CALC_RESET {		 \
		start = stop = 0;        \
		delta       = 0;         \
		ddelta      = 0;         \
		var         = 0;         \
		average_clk = 0;	 \
		average_us  = 0;	 \
		avclk_old   = 0;         \
		min         = 9999999999;\
		max         = 0;         \
		timer.reset();		 \
		cnt = 1;		 \
	}
	
	#define CALC_START {		 \
		timer.reset();		 \
		timer.start();		 \
		start = DWT->CYCCNT;	 \
	}
	
	#define CALC_STOP {		 		 \
		stop         = DWT->CYCCNT;     	 \
		us           = timer.read_us(); 	 \
		avclk_old    = average_clk;              \
		delta        = stop - start;             \
		ddelta       = (long double) delta;      \
		average_clk += ((ddelta-average_clk)/cnt);\
		var         += ((ddelta-average_clk)*(ddelta-avclk_old));\
		var         /= cnt;                      \
		average_us  += (long double)(us-average_us)/cnt;\
		min          = MIN(delta,min);           \
		max          = MAX(delta,max);           \
		cnt         += 1;			 \
	}
	
	#define CALC_AVG {				 \
		std_err      = sqrt(var/cnt);            \
	}

	#define timer_read_ms(x)    chrono::duration_cast<chrono::milliseconds>((x).elapsed_time()).count()

	//set so that cycle counter can be read from  DWT->CYCCNT
	CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
	DWT->LAR = 0xC5ACCE55;
	DWT->CYCCNT = 0;
	DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
	int ret_val = 0;
	
	int i = 5;
	while(i > 0) {
		wait(1);
		pc.printf("This program runs will run in %d seconds.\n\r", i--);
		myled = !myled;
	}

	/// LAZY FALCON TESTING
	unsigned logn = 9; // set to 9 for 512 parameters, 10 for 1024
	char seed[16] = {0};
	shake256_context sc;
	shake256_init_prng_from_seed(&sc, seed, 16);

	void *pubkey, *privkey, *sig, *expkey;// *sigct, *expkey;
	size_t pubkey_len, privkey_len, sig_len, expkey_len;
	void *tmpsd, *tmpkg, *tmpmp, *tmpvv, *tmpek, *tmpst;
	size_t tmpkg_len, tmpvv_len, tmpmp_len, tmpsd_len, tmpek_len, tmpst_len;
	
	fflush(stdout);
	
	pubkey_len  = FALCON_PUBKEY_SIZE(logn);
	privkey_len = FALCON_PRIVKEY_SIZE(logn);
	sig_len     = FALCON_SIG_CT_SIZE(logn);
	expkey_len  = FALCON_EXPANDEDKEY_SIZE(logn);
	
	tmpsd_len   = FALCON_TMPSIZE_SIGNDYN(logn);
	tmpkg_len   = FALCON_TMPSIZE_KEYGEN(logn);
	tmpmp_len   = FALCON_TMPSIZE_MAKEPUB(logn);
	tmpvv_len   = FALCON_TMPSIZE_VERIFY(logn);
	tmpek_len   = FALCON_TMPSIZE_EXPANDPRIV(logn);
	tmpst_len   = FALCON_TMPSIZE_SIGNTREE(logn);

	pubkey      = xmalloc(pubkey_len);
	privkey     = xmalloc(privkey_len);
	sig         = xmalloc(sig_len);
	expkey      = xmalloc(expkey_len);
		
	tmpkg       = xmalloc(tmpkg_len);	
	tmpsd 	    = xmalloc(tmpsd_len);
	tmpmp 	    = xmalloc(tmpmp_len);
	tmpvv       = xmalloc(tmpvv_len);
	tmpek       = xmalloc(tmpek_len);
	tmpst       = xmalloc(tmpst_len);

	pc.printf("---------------------------\n\r");
	pc.printf("| START SIGNATURE TESTING |\n\r");
	pc.printf("---------------------------\n\r");

	pc.printf("------------------------\n\r");
	pc.printf("| Starting Lazy Falcon |\n\r");
	pc.printf("-----------------------\n\r");

	pc.printf("------------------------\n\r");
	pc.printf("| Doing Key Generation |\n\r");
	pc.printf("------------------------\n\r");
	
	memset(privkey, 0, privkey_len);
	memset(pubkey, 0, pubkey_len);
	CALC_RESET
	for (size_t r=0; r<BENCHMARK_ROUND/BENCHMARK_ROUND; r++) {
		DWT->CYCCNT = 0;
		CALC_START
		ret_val += falcon_keygen_make(&sc, logn, privkey, privkey_len,
			pubkey, pubkey_len, tmpkg, tmpkg_len);
		CALC_STOP
	}
	CALC_AVG
  
	pc.printf("Avg clock cycles:        %.0Lf\n\r", average_clk);
	pc.printf("Min clock cycles:        %lld\n\r", min);
	pc.printf("Max clock cycles:        %lld\n\r", max);
	pc.printf("Std dev of clock cycles: %.1Lf\n\r", (sqrt(var)));
	pc.printf("Std err of clock cycles: %.1Lf\n\r", (std_err));
    pc.printf("Avg time in millisecs:   %.1Lf\n\r", average_us/1000);
        
	memset(pubkey, 0xFF, pubkey_len);
	ret_val += falcon_make_public(pubkey, pubkey_len,
			privkey, privkey_len, tmpmp, tmpmp_len);
		
	pc.printf("----------------------\n\r");
	pc.printf("| Doing Lazy Signing |\n\r");
	pc.printf("----------------------\n\r");
	
	memset(sig, 0, sig_len);
	CALC_RESET
	for (size_t r=0; r<BENCHMARK_ROUND; r++) {
		DWT->CYCCNT = 0;
		CALC_START
		ret_val += falcon_sign_dyn_lazy(&sc, sig, &sig_len, FALCON_SIG_CT,
			pubkey, pubkey_len, privkey, privkey_len, "data1", 5, tmpsd, tmpsd_len);
		CALC_STOP
	}
	CALC_AVG
	
	pc.printf("Avg clock cycles:        %.0Lf\n\r", average_clk);
	pc.printf("Min clock cycles:        %lld\n\r", min);
	pc.printf("Max clock cycles:        %lld\n\r", max);
	pc.printf("Std dev of clock cycles: %.1Lf\n\r", (sqrt(var)));
	pc.printf("Std err of clock cycles: %.1Lf\n\r", (std_err));
    pc.printf("Avg time in millisecs:   %.1Lf\n\r", average_us/1000);



	pc.printf("-------------------\n\r");	
	pc.printf("| Doing Verifying |\n\r");
	pc.printf("-------------------\n\r");	
	
	CALC_RESET
	for (size_t r=0; r<BENCHMARK_ROUND; r++) {
		DWT->CYCCNT = 0;
		CALC_START			
		ret_val += falcon_verify(sig, sig_len, FALCON_SIG_CT, pubkey, 
			pubkey_len, "data1", 5, tmpvv, tmpvv_len);
		CALC_STOP
	}
	CALC_AVG
	
	pc.printf("Avg clock cycles:        %.0Lf\n\r", average_clk);
	pc.printf("Min clock cycles:        %lld\n\r", min);
	pc.printf("Max clock cycles:        %lld\n\r", max);
	pc.printf("Std dev of clock cycles: %.1Lf\n\r", (sqrt(var)));
	pc.printf("Std err of clock cycles: %.1Lf\n\r", (std_err));
    pc.printf("Avg time in millisecs:   %.1Lf\n\r", average_us/1000);

    pc.printf("------------------------\n\r");	
	pc.printf("| Lazy Falcon Finished |\n\r");
    pc.printf("------------------------\n\r");	
	display_mallinfo();
	fflush(stdout);


	pc.printf("-------------------\n\r");
	pc.printf("| Starting Falcon |\n\r");
	pc.printf("-------------------\n\r");

	pc.printf("------------------------\n\r");
	pc.printf("| Doing Key Generation |\n\r");
	pc.printf("------------------------\n\r");
	
	memset(privkey, 0, privkey_len);
	memset(pubkey, 0, pubkey_len);
	CALC_RESET
	for (size_t r=0; r<BENCHMARK_ROUND/10; r++) {
		DWT->CYCCNT = 0;
		CALC_START
		ret_val += falcon_keygen_make(&sc, logn, privkey, privkey_len,
			pubkey, pubkey_len, tmpkg, tmpkg_len);
		CALC_STOP
	}
	CALC_AVG
  
	pc.printf("Avg clock cycles:        %.0Lf\n\r", average_clk);
	pc.printf("Min clock cycles:        %lld\n\r", min);
	pc.printf("Max clock cycles:        %lld\n\r", max);
	pc.printf("Std dev of clock cycles: %.1Lf\n\r", (sqrt(var)));
	pc.printf("Std err of clock cycles: %.1Lf\n\r", (std_err));
    pc.printf("Avg time in millisecs:   %.1Lf\n\r", average_us/1000);
        
	memset(pubkey, 0xFF, pubkey_len);
	ret_val += falcon_make_public(pubkey, pubkey_len,
			privkey, privkey_len, tmpmp, tmpmp_len);
		
	pc.printf("-----------------------\n\r");
	pc.printf("| Doing Signing (dyn) |\n\r");
	pc.printf("-----------------------\n\r");
	
	memset(sig, 0, sig_len);
	CALC_RESET
	for (size_t r=0; r<BENCHMARK_ROUND; r++) {
		DWT->CYCCNT = 0;
		CALC_START
		ret_val += falcon_sign_dyn(&sc, sig, &sig_len, FALCON_SIG_CT,
			privkey, privkey_len, "data1", 5, tmpsd, tmpsd_len);
		CALC_STOP
	}
	CALC_AVG
	
	pc.printf("Avg clock cycles:        %.0Lf\n\r", average_clk);
	pc.printf("Min clock cycles:        %lld\n\r", min);
	pc.printf("Max clock cycles:        %lld\n\r", max);
	pc.printf("Std dev of clock cycles: %.1Lf\n\r", (sqrt(var)));
	pc.printf("Std err of clock cycles: %.1Lf\n\r", (std_err));
    pc.printf("Avg time in millisecs:   %.1Lf\n\r", average_us/1000);

	pc.printf("-------------------\n\r");	
	pc.printf("| Doing Verifying |\n\r");
	pc.printf("-------------------\n\r");	
	
	CALC_RESET
	for (size_t r=0; r<BENCHMARK_ROUND; r++) {
		DWT->CYCCNT = 0;
		CALC_START			
		ret_val += falcon_verify(sig, sig_len, FALCON_SIG_CT, pubkey, 
			pubkey_len, "data1", 5, tmpvv, tmpvv_len);
		CALC_STOP
	}
	CALC_AVG
	
	pc.printf("Avg clock cycles:        %.0Lf\n\r", average_clk);
	pc.printf("Min clock cycles:        %lld\n\r", min);
	pc.printf("Max clock cycles:        %lld\n\r", max);
	pc.printf("Std dev of clock cycles: %.1Lf\n\r", (sqrt(var)));
	pc.printf("Std err of clock cycles: %.1Lf\n\r", (std_err));
    pc.printf("Avg time in millisecs:   %.1Lf\n\r", average_us/1000);

    pc.printf("-------------------\n\r");	
	pc.printf("| Falcon Finished |\n\r");
    pc.printf("-------------------\n\r");	
	display_mallinfo();
	fflush(stdout);


	/*
 	* Dilithium Round 3 code using pqm4 as it's faster than pqclean.
 	* change Dilithium's parameters in config.h to either 2, 3, or 5.
	* ret_val outputs 0 if functions work as expected.
	* comment code out below to switch between Falcon and Dilithium.
	*/

	#define MLEN 59
	size_t mlen, smlen;
	uint8_t pk[CRYPTO_PUBLICKEYBYTES] = {0};
	uint8_t sk[CRYPTO_SECRETKEYBYTES] = {0};
	uint8_t m[MLEN + CRYPTO_BYTES];
	uint8_t m2[MLEN + CRYPTO_BYTES];
	uint8_t sm[MLEN + CRYPTO_BYTES];
	randombytes(m, MLEN);
	        
	fflush(stdout);

	pc.printf("----------------------\n\r");
	pc.printf("| Starting Dilithium |\n\r");
	pc.printf("----------------------\n\r");
	
	pc.printf("------------------------\n\r");
	pc.printf("| Doing Key Generation |\n\r");
	pc.printf("------------------------\n\r");
	
	CALC_RESET
	for (size_t r=0; r<BENCHMARK_ROUND; r++) {
		DWT->CYCCNT = 0;
		CALC_START
		ret_val = crypto_sign_keypair(pk, sk);
		CALC_STOP
	}
	CALC_AVG
	
	pc.printf("Avg clock cycles:        %.0Lf\n\r", average_clk);
	pc.printf("Min clock cycles:        %lld\n\r", min);
	pc.printf("Max clock cycles:        %lld\n\r", max);
	pc.printf("Std dev of clock cycles: %.1Lf\n\r", (sqrt(var)));
	pc.printf("Std err of clock cycles: %.1Lf\n\r", (std_err));
    pc.printf("Avg time in millisecs:   %.1Lf\n\r", average_us/1000);
	
	pc.printf("-----------------\n\r");
	pc.printf("| Doing Signing |\n\r");
	pc.printf("-----------------\n\r");
	
	CALC_RESET
	for (size_t r=0; r<BENCHMARK_ROUND; r++) {
		DWT->CYCCNT = 0;
		randombytes(m, MLEN);
		CALC_START
	ret_val = crypto_sign(sm, &smlen, m, MLEN, sk);
		CALC_STOP
	}
	CALC_AVG
	
	pc.printf("Avg clock cycles:        %.0Lf\n\r", average_clk);
	pc.printf("Min clock cycles:        %lld\n\r", min);
	pc.printf("Max clock cycles:        %lld\n\r", max);
	pc.printf("Std dev of clock cycles: %.1Lf\n\r", (sqrt(var)));
	pc.printf("Std err of clock cycles: %.1Lf\n\r", (std_err));
    pc.printf("Avg time in millisecs:   %.1Lf\n\r", average_us/1000);

	pc.printf("------------------------\n\r");
	pc.printf("| Doing Signing (open) |\n\r");
	pc.printf("------------------------\n\r");

	CALC_RESET
	for (size_t r=0; r<BENCHMARK_ROUND; r++) {
		DWT->CYCCNT = 0;		
		CALC_START
		ret_val = crypto_sign_open(m2, &mlen, sm, smlen, pk);
		CALC_STOP
	}
	CALC_AVG
	
	pc.printf("Avg clock cycles:        %.0Lf\n\r", average_clk);
	pc.printf("Min clock cycles:        %lld\n\r", min);
	pc.printf("Max clock cycles:        %lld\n\r", max);
	pc.printf("Std dev of clock cycles: %.1Lf\n\r", (sqrt(var)));
	pc.printf("Std err of clock cycles: %.1Lf\n\r", (std_err));
        pc.printf("Avg time in millisecs:   %.1Lf\n\r", average_us/1000);

	pc.printf("-------------------\n\r");
	pc.printf("| Doing Verifying |\n\r");
	pc.printf("-------------------\n\r");

	CALC_RESET
	for (size_t r=0; r<BENCHMARK_ROUND; r++) {
		DWT->CYCCNT = 0;
		CALC_START	
	ret_val = crypto_sign_verify(sm, CRYPTO_BYTES, m, MLEN, pk);
		CALC_STOP
	}
	CALC_AVG
	
	pc.printf("Avg clock cycles:        %.0Lf\n\r", average_clk);
	pc.printf("Min clock cycles:        %lld\n\r", min);
	pc.printf("Max clock cycles:        %lld\n\r", max);
	pc.printf("Std dev of clock cycles: %.1Lf\n\r", (sqrt(var)));
	pc.printf("Std err of clock cycles: %.1Lf\n\r", (std_err));
    pc.printf("Avg time in millisecs:   %.1Lf\n\r", average_us/1000);
	
	pc.printf("----------------------\n\r");
	pc.printf("| Dilithium Finished |\n\r");
	pc.printf("----------------------\n\r");
	display_mallinfo();	
	fflush(stdout);

    unsigned char public_key[32], private_key[64], seed2[32], scalar[32];
    unsigned char other_public_key[32], other_private_key[64];
    unsigned char shared_secret[32], other_shared_secret[32];
    unsigned char signature[64];

    const unsigned char message[] = "Hello, world!";
    const int message_len = strlen((char*) message);

	pc.printf("--------------------\n\r");
	pc.printf("| Starting ed25519 |\n\r");
	pc.printf("--------------------\n\r");

	pc.printf("------------------------\n\r");
	pc.printf("| Doing ed25519 keygen |\n\r");
	pc.printf("------------------------\n\r");

	CALC_RESET
	for (size_t r=0; r<BENCHMARK_ROUND; r++) {
	    /* create a random seed, and a keypair out of that seed */
		ed25519_create_seed(seed2);
		DWT->CYCCNT = 0;
		CALC_START
		ed25519_create_keypair(public_key, private_key, seed2);
		CALC_STOP
	}
	CALC_AVG
	
	pc.printf("Avg clock cycles:        %.0Lf\n\r", average_clk);
	pc.printf("Min clock cycles:        %lld\n\r", min);
	pc.printf("Max clock cycles:        %lld\n\r", max);
	pc.printf("Std dev of clock cycles: %.1Lf\n\r", (sqrt(var)));
	pc.printf("Std err of clock cycles: %.1Lf\n\r", (std_err));
    pc.printf("Avg time in millisecs:   %.1Lf\n\r", average_us/1000);

	pc.printf("----------------------\n\r");
	pc.printf("| Doing ed25519 Sign |\n\r");
	pc.printf("----------------------\n\r");
	
	CALC_RESET
	for (size_t r=0; r<BENCHMARK_ROUND; r++) {
	    /* create a random seed, and a keypair out of that seed */
		ed25519_create_seed(seed2);
		ed25519_create_keypair(public_key, private_key, seed2);
		DWT->CYCCNT = 0;
		CALC_START
		/* create signature on the message with the keypair */
		ed25519_sign(signature, message, message_len, public_key, private_key);
		CALC_STOP
	}
	CALC_AVG
	
	pc.printf("Avg clock cycles:        %.0Lf\n\r", average_clk);
	pc.printf("Min clock cycles:        %lld\n\r", min);
	pc.printf("Max clock cycles:        %lld\n\r", max);
	pc.printf("Std dev of clock cycles: %.1Lf\n\r", (sqrt(var)));
	pc.printf("Std err of clock cycles: %.1Lf\n\r", (std_err));
    pc.printf("Avg time in millisecs:   %.1Lf\n\r", average_us/1000);

	pc.printf("------------------------\n\r");
	pc.printf("| Doing ed25519 Verify |\n\r");
	pc.printf("------------------------\n\r");

	CALC_RESET
	for (size_t r=0; r<BENCHMARK_ROUND; r++) {
	    /* create a random seed, and a keypair out of that seed */
		ed25519_create_seed(seed2);
		ed25519_create_keypair(public_key, private_key, seed2);
		ed25519_sign(signature, message, message_len, public_key, private_key);
		DWT->CYCCNT = 0;
		CALC_START
		ed25519_verify(signature, message, message_len, public_key);
		CALC_STOP
	}
	CALC_AVG
	
	pc.printf("Avg clock cycles:        %.0Lf\n\r", average_clk);
	pc.printf("Min clock cycles:        %lld\n\r", min);
	pc.printf("Max clock cycles:        %lld\n\r", max);
	pc.printf("Std dev of clock cycles: %.1Lf\n\r", (sqrt(var)));
	pc.printf("Std err of clock cycles: %.1Lf\n\r", (std_err));
    pc.printf("Avg time in millisecs:   %.1Lf\n\r", average_us/1000);

	pc.printf("--------------------\n\r");	
	pc.printf("| ed25519 Finished |\n\r");
    pc.printf("--------------------\n\r");	
	display_mallinfo();
	fflush(stdout);

	pc.printf("-------------------------\n\r");
	pc.printf("| END SIGNATURE TESTING |\n\r");
	pc.printf("-------------------------\n\r");

}
