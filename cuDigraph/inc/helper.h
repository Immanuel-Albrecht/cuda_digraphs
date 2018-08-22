/* ERROR HANDLING HELPERS */

/* USAGE: checkMalloc( x = malloc(...))
      or  checkMalloc( x = calloc(...))
      etc.
*/
#define checkMalloc(ptr) do { \
	if ( ( (ptr) ) == NULL) {     \
fprintf(stderr, "%s(%i) : %s : (%d) %s.\n", __FILE__, __LINE__, "malloc failed", errno, strerror(errno)); \
 \
exit(EXIT_FAILURE);}} while (0)

#define safeFree(x) do { free(x); x = 0; } while (0)
