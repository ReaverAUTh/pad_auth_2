#ifndef SUPPLEMENTARY_H_INCLUDED
#define SUPPLEMENTARY_H_INCLUDED

// sort <arr>, mirror change its <carry>
void mergeSort_carry_one(double *arr, int *carry, int l, int r);
void merge_carry_one(double *arr, int *carry, int l, int m, int r);


double time_spent(struct timespec start_time, struct timespec end_time);
double quickselect(double A[], int left, int right, int k);
double partition_of_quick(double a[], int left, int right, int pivot);
void SWAP(double *x, double *y);


double *read_from_file(int *N, int *d);

#endif // SUPPLEMENTARY_H_INCLUDED
