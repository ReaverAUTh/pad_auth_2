#ifndef VPT_H_INCLUDED
#define VPT_H_INCLUDED

typedef struct VPtree
{
    //vantage point coordinates as an array
    double *vp;
    //median distance
    double md ;
    //vp real index
    int idx;
    //vantage point subtrees
    struct VPtree *inner;
    struct VPtree *outer;

} vptree;

struct VPtree *createVPT(double *X, int n, int d, int offset);
struct VPtree *copyTree(struct VPtree * T);
void destroy(struct VPtree *T);

// getters for ease
struct VPtree *getInner(struct VPtree * T);
struct VPtree *getOuter(struct VPtree * T);
double getMD(struct VPtree * T);
double *getVP(struct VPtree * T);
int getIDX(struct VPtree * T);

#endif // VPT_H_INCLUDED
