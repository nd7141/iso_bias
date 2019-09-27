/* This program prints generators for the automorphism group of an
   n-vertex polygon, where n is a number supplied by the user.

   This version uses dynamic allocation.
*/

#include "nauty26r11/nauty.h"
/* MAXN=0 is defined by nauty.h, which implies dynamic allocation */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct nodes_edges {
    int n_nodes, n_edges;
};

typedef struct nodes_edges Struct;

Struct read_edges(char *inf, int edges[]) {
    FILE *file = fopen (inf, "r" );

    char line [128]; /* or other suitable maximum line size */
    int linec = 0;
    char * pch;
    int total = 0;
    int m;
    int n_nodes;
    int n_edges;
    int line_words[2]; // first two words in a line
    int word_count = 0;
    Struct s;

//    get edges of a graph from the file
    while ( fgets ( line, sizeof line, file ) != NULL ) /* read a line */
    {
     if (linec == 0) { // first line contains the number of nodes and edges
        pch = strtok(line, " ");
        while (pch != NULL && word_count < 2) {
            line_words[word_count] = atoi(pch);
            word_count += 1;
            pch = strtok(NULL, " ");
        }
        n_nodes = line_words[0];
        n_edges = line_words[1];
        m = SETWORDSNEEDED(n_nodes);
     }
     else { // each line contains a pair of nodes u v
        word_count = 0;
        pch = strtok(line, " ");
        while (pch != NULL && word_count < 2) {
            line_words[word_count] = atoi(pch);
            word_count += 1;
            pch = strtok(NULL, " ");
        }
        edges[total] = line_words[0];
        edges[total + 1] = line_words[1];
        total += 2;
     }
     linec += 1;
    }
    fclose(file);

    s.n_nodes = n_nodes;
    s.n_edges = n_edges;

    return s;
};

int main(int argc, char *argv[])
{
//    printf("Number of arguments %d\n", argc);
    char *inf = argv[1];
    char *outf = argv[2];

  /* DYNALLSTAT declares a pointer variable (to hold an array when it
     is allocated) and a size variable to remember how big the array is.
     Nothing is allocated yet.  */

    DYNALLSTAT(graph,g,g_sz);
    DYNALLSTAT(int,lab,lab_sz);
    DYNALLSTAT(int,ptn,ptn_sz);
    DYNALLSTAT(int,orbits,orbits_sz);
    static DEFAULTOPTIONS_GRAPH(options);
    statsblk stats;

    int n,m,v,i;
    set *gv;

/* Default options are set by the DEFAULTOPTIONS_GRAPH macro above.
   Here we change those options that we want to be different from the
   defaults.  writeautoms=TRUE causes automorphisms to be written. */

    options.writeautoms = TRUE;
//    options.writemarkers = TRUE;

//  write group automorphism to a file
    FILE *fptr;
    fptr = fopen(outf, "w");
    options.outfile = fptr;
//
// read graph
    int edges1[100000];
    Struct s1 = read_edges(inf, edges1);
//    printf("Nodes: %d %d\n", s1.n_nodes, s1.n_edges);
    int n1 = s1.n_nodes;
    int m1 = s1.n_edges;



    n = n1;
    m = SETWORDSNEEDED(n);

    DYNALLOC2(graph,g,g_sz,m,n,"malloc");
    DYNALLOC1(int,lab,lab_sz,n,"malloc");
    DYNALLOC1(int,ptn,ptn_sz,n,"malloc");
    DYNALLOC1(int,orbits,orbits_sz,n,"malloc");

    EMPTYGRAPH(g,m,n);
    for (i = 0; i < 2*m1; i+=2) {
        ADDONEEDGE(g, edges1[i], edges1[i+1], m);
    }

    densenauty(g,lab,ptn,orbits,&options,&stats,m,n,NULL);
//    writegroupsize(stdout,stats.grpsize1,stats.grpsize2);

    fclose(fptr);

    exit(0);
}
