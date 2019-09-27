/* This program demonstrates how an isomorphism is found between
   graphs of the form in the figure above, for general size.

   This version uses dense form with dynamic allocation.
*/

#include "nauty26r11/nauty.h"
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

int read_labels(char *inf, int lab[], int ptn[]) {
    FILE *file = fopen (inf, "r");

    int line_count = 0;
    int word_count;
    int line_words[2];
    char * pch;
    char line [128];
    while ( fgets ( line, sizeof line, file ) != NULL) /* read a line */
    {
        word_count = 0;
        pch = strtok(line, " ");
        while (pch != NULL && word_count < 2) {
            line_words[word_count] = atoi(pch);
            word_count += 1;
            pch = strtok(NULL, " ");
        }
//        printf("%d %d\n", line_words[0], line_words[1]);
        lab[line_count] = line_words[0];
        ptn[line_count] = line_words[1];
        line_count += 1;
    }
    fclose(file);
    return 0;
};


int
main(int argc, char *argv[])
{
//    printf("Number of arguments %d\n", argc);
    if (argc < 4) {
//        printf("You should provide two files for graphs + output.");
        exit(1);
    }
    char *file1 = argv[1];
    char *file2 = argv[2];
    char *outf = argv[3];

//    FILE *file = fopen (outf, "a" );

//    printf("Files %s %s\n", file1, file2);


//    DYNALLSTAT(int,lab1,lab1_sz);
//    DYNALLSTAT(int,lab2,lab2_sz);
//    DYNALLSTAT(int,ptn1,ptn_sz1);
//    DYNALLSTAT(int,ptn2,ptn_sz2);
    DYNALLSTAT(int,orbits,orbits_sz);
    DYNALLSTAT(int,map,map_sz);
    DYNALLSTAT(graph,g1,g1_sz);
    DYNALLSTAT(graph,g2,g2_sz);
    DYNALLSTAT(graph,cg1,cg1_sz);
    DYNALLSTAT(graph,cg2,cg2_sz);
    static DEFAULTOPTIONS_GRAPH(options);
    statsblk stats;

    int n,m,i;

 /* Select option for canonical labelling */

    options.getcanon = TRUE;

 /* Now make the first graph */
    int edges1[100000];
    Struct s1 = read_edges(file1, edges1);
//    printf("Nodes: %d %d\n", s1.n_nodes, s1.n_edges);
    int n1 = s1.n_nodes;
    int m1 = s1.n_edges;

     /* Now make the second graph */
    int edges2[100000];
    Struct s2 = read_edges(file2, edges2);
//    printf("Nodes: %d %d\n", s2.n_nodes, s2.n_edges);
    int n2 = s2.n_nodes;
    int m2 = s2.n_edges;

    if (n1 != n2 || m1 != m2) {
//        printf("Non-isomorphic based on number of nodes/edges.\n");
//        fprintf(file, "%s %s %d\n", file1, file2, 0);
        exit(0);
    }

    n = n1;
    m = SETWORDSNEEDED(n);
    nauty_check(WORDSIZE, m, n, NAUTYVERSIONID);

//    DYNALLOC1(int,lab1,lab1_sz,n,"malloc");
//    DYNALLOC1(int,lab2,lab2_sz,n,"malloc");
//    DYNALLOC1(int,ptn1,ptn_sz1,n,"malloc");
//    DYNALLOC1(int,ptn2,ptn_sz2,n,"malloc");
    DYNALLOC1(int,orbits,orbits_sz,n,"malloc");
    DYNALLOC1(int,map,map_sz,n,"malloc");
    DYNALLOC2(graph,g1,g1_sz,n,m,"malloc");
    DYNALLOC2(graph,g2,g2_sz,n,m,"malloc");
    DYNALLOC2(graph,cg1,cg1_sz,n,m,"malloc");
    DYNALLOC2(graph,cg2,cg2_sz,n,m,"malloc");

    EMPTYGRAPH(g1,m,n);
//    printf("1. Edges 1:\n");
    for (i = 0; i < 2*s1.n_edges; i+=2) {
//        printf("%d %d\n", edges1[i], edges1[i+1]);
        ADDONEEDGE(g1, edges1[i], edges1[i+1], m);
    }

    EMPTYGRAPH(g2,m,n);
//    printf("2. Edges:\n");
    for (i = 0; i < 2*s2.n_edges; i+=2) {
//        printf("%d %d\n", edges2[i], edges2[i+1]);
        ADDONEEDGE(g2, edges2[i], edges2[i+1], m);
    }

    char *lab_fn1 = argv[4];
    char *lab_fn2 = argv[5];
    int lab1[s1.n_nodes];
    int ptn1[s1.n_nodes];
    int lab2[s2.n_nodes];
    int ptn2[s2.n_nodes];
    read_labels(lab_fn1, lab1, ptn1);
    read_labels(lab_fn2, lab2, ptn2);
//    printf("Label 1 %d %d\n", s1.n_nodes, s1.n_edges);
//    for (i=0; i < s1.n_nodes; i+=1) {
//        printf("%d %d\n", lab1[i], ptn1[i]);
//    }
//    printf("Label 2 %d %d\n", s2.n_nodes, s2.n_edges);
//    for (i=0; i < s2.n_nodes; i+=1) {
//        printf("%d %d\n", lab2[i], ptn2[i]);
//    }

 /* Label g1, result in cg1 and labelling in lab1; similarly g2.
    It is not necessary to pre-allocate space in cg1 and cg2, but
    they have to be initialised as we did above.  */

    options.defaultptn = FALSE;
    densenauty(g1,lab1,ptn1,orbits,&options,&stats,m, n,cg1);
    densenauty(g2,lab2,ptn2,orbits,&options,&stats,m, n,cg2);

 /* Compare canonically labelled graphs */
    if (memcmp(cg1,cg2,m*sizeof(graph)*n) == 0)
    {
        FILE *file = fopen (outf, "w");
        fprintf(file, "%s %s %d\n", file1, file2, 1);

//        exit(0);

//        printf("Isomorphic. %s %s\n", file1, file2);
//        if (n <= 1000)
//        {
//         /* Write the isomorphism.  For each i, vertex lab1[i]
//            of sg1 maps onto vertex lab2[i] of sg2.  We compute
//            the map in order of labelling because it looks better. */
//
//            for (i = 0; i < n; ++i) map[lab1[i]] = lab2[i];
//            for (i = 0; i < n; ++i) printf(" %d-%d",i,map[i]);
//            printf("\n");
//        }
        fclose(file);
    }
    else {
//        printf("Not isomorphic.\n");
//        fprintf(file, "%s %s %d\n", file1, file2, 0);
//        exit(0);
    }


    exit(0);
}
