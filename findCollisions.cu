#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include<string.h>
#include <math.h>
#include <cuda.h>
#include <mpi.h>

#include<sys/stat.h>
#include<sys/time.h>


#define ROOT 0
#define MAXNGPU 3
#define MAX_THREAD 512
#define M 100
#define PRIME 1567
#define DP 7

#define TIMER_DEF     struct timeval temp_1, temp_2
#define TIMER_START   gettimeofday(&temp_1, (struct timezone*)0)
#define TIMER_STOP    gettimeofday(&temp_2, (struct timezone*)0)
#define TIMER_ELAPSED ((temp_2.tv_sec-temp_1.tv_sec)*1.e6+(temp_2.tv_usec-temp_1 .tv_usec))



//nvcc -I/usr/local/openmpi-4.1.4/include -L/usr/local/openmpi-4.1.4/lib -lmpi findCollisions.cu -o findColl
// mpirun -np 8 findcoll 50 38

void cudaErrorCheck(cudaError_t error, const char * msg){
        if ( error != cudaSuccess){
        fprintf(stderr, "%s:%s\n ", msg, cudaGetErrorString(error));
        exit(EXIT_FAILURE);}}


//findPathAndDP<<<nblocks, nthreads>>>(d_x0p, d_x,lg, n_per_proc, d_DjX0, d_DjD, d_Djsteps, d_DjC, d_nDPaux);
__global__ void findPathAndDP(
        uint32_t* d_x0p,
        uint32_t* d_x,
        int lg,
        int n_per_proc,
        uint32_t* d_DjX0,
        uint32_t* d_DjD,
        uint32_t* d_Djsteps,
        uint32_t* d_DjC,
        int* d_nDPaux
) {

        int tid = threadIdx.x + blockDim.x * blockIdx.x;

        if (tid<n_per_proc) {
                //initialization of arrays
                d_nDPaux[tid]=0;
                d_DjX0[tid]=0;
                d_DjD[tid]=0;
                d_Djsteps[tid]=0;
                d_DjC[tid]=0;
                //places starting points in the right position
                d_x[tid*lg] = d_x0p[tid];
                //printf("starting point %d \n", d_x[tid*lg]);
                __syncthreads();
                //all threads start computing path
                for(int i = 0; i < lg-1; i++) {
                        d_x[tid*lg+i+1] = (d_x[tid*lg+i]*d_x[tid*lg+i]+1) % PRIME;
                        //printf("indx %d path ID %d, %d step %d \n",tid*lg+i, tid, i+1, d_x[tid*lg+i+1]);
                                //finds DPs
                                if (d_x[tid*lg+i+1] % DP == 0) {
                                        d_nDPaux[tid]=1;
                                        d_DjX0[tid] = d_x0p[tid];
                                        d_DjD[tid] = d_x[tid*lg+i+1];
                                        d_Djsteps[tid] = i+1;
                                        d_DjC[tid] = d_x[tid*lg+i];
                                        //printf("DP found %d in indx %d \n", d_DjD[tid], tid);
                                        break;
                                }
                }
                __syncthreads();
        }
}





int main (int argc, char** argv) {

TIMER_DEF;
TIMER_START;

//input validation
if(argc != 3){
        fprintf(stderr,"wrong number of inputs\n");
        return EXIT_FAILURE;}

int lg=atoi(argv[1]);

if(lg <=0){
        fprintf(stderr,"[ERROR] - lg must be > 0\n");
        return EXIT_FAILURE;}

int max=atoi(argv[2]);

 if(max <=0){
        fprintf(stderr,"[ERROR] - max must be > 0\n");
        return EXIT_FAILURE;}

 if(max>(PRIME/DP)*(PRIME/DP-1)/2){
        fprintf(stderr,"[ERROR] - required collisions too high for this method\n");
        return EXIT_FAILURE;}

//MPI initialization 
int rank, NP;
MPI_Init(&argc, &argv);

MPI_Comm_rank(MPI_COMM_WORLD, &rank);
MPI_Comm_size(MPI_COMM_WORLD, &NP);


int n_per_proc; // elements per process
n_per_proc=M/NP;

if (rank==ROOT) printf("num start point per proc %d \n", n_per_proc);

//each process selects a GPU to work on
int usableGPUs;
cudaErrorCheck(cudaGetDeviceCount(&usableGPUs),"cudaGetDevice");
if (usableGPUs>MAXNGPU) usableGPUs=MAXNGPU;
cudaErrorCheck(cudaSetDevice(rank%MAXNGPU),"cudaSetDevice");
//initialization global values and arrays
int nCollisFinal=0;

int *scA=(int*)malloc(sizeof(int)*(max+M*(M-1)/2));
if(NULL==scA){
        fprintf(stderr,"[ERROR] - Cannot allocate memory\n");
        return EXIT_FAILURE; }
int *scB=(int*)malloc(sizeof(int)*(max+M*(M-1)/2));
for (int i=0; i<max+M*(M-1)/2 ; i++){
        scA[i]=0;
        scB[i]=0;}



while (nCollisFinal < max){

        //generation and scattering of random starting points
        int nDPj=0;

        uint32_t *x0=(uint32_t*)malloc(sizeof(int)*M);

        if (NULL==x0) {
        fprintf(stderr,"[ERROR][RANK %d] Cannot allocate memory\n",rank);
        MPI_Abort(MPI_COMM_WORLD,1);}


        if (rank == ROOT){
        for (int i = 0; i < M; i++){
                x0[i]=rand()%PRIME;
                //printf("%d \n", x0[i]);
                }
        }

        uint32_t *x0p=(uint32_t*)malloc(sizeof(int)*n_per_proc);

        if(NULL==x0p){
        fprintf(stderr,"[ERROR] - Cannot allocate memory\n");
        return EXIT_FAILURE; }

        //printf("abt to scatter %d \n", n_per_proc);
        MPI_Scatter(x0, n_per_proc, MPI_INT, x0p, n_per_proc, MPI_INT, ROOT, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        if(rank==ROOT) printf("scatter success %d \n", M);

        //allocation and initialization of device arrays
        uint32_t *d_x0p;
        cudaErrorCheck(cudaMalloc(&d_x0p,sizeof(int)*n_per_proc),"cudaMalloc d_x0p");
        cudaErrorCheck(cudaMemcpy(d_x0p,x0p,sizeof(int)*n_per_proc,cudaMemcpyHostToDevice),"Memcpy d_x0p");

        uint32_t *d_x;
        cudaErrorCheck(cudaMalloc(&(d_x), sizeof(int) * lg*n_per_proc),"cudaMalloc d_x");

        uint32_t *d_DjX0;
        uint32_t *d_DjD;
        uint32_t *d_Djsteps;
        uint32_t *d_DjC;
        int *d_nDPaux;

        cudaErrorCheck(cudaMalloc(&(d_DjX0), sizeof(int) *n_per_proc),"cudaMalloc d_DjX0");
        cudaErrorCheck(cudaMalloc(&(d_DjD), sizeof(int)*n_per_proc),"cudaMalloc d_DjD");
        cudaErrorCheck(cudaMalloc(&(d_Djsteps), sizeof(int) *n_per_proc),"cudaMalloc d_Djsteps");
        cudaErrorCheck(cudaMalloc(&(d_DjC), sizeof(int) *n_per_proc),"cudaMalloc d_DjC");
        cudaErrorCheck(cudaMalloc(&(d_nDPaux), sizeof(int)*n_per_proc),"cudaMalloc d_nDPaux");

uint32_t *DjX0=(uint32_t*)malloc(sizeof(int)*n_per_proc);
        uint32_t *DjD=(uint32_t*)malloc(sizeof(int)*n_per_proc);
        uint32_t *Djsteps=(uint32_t*)malloc(sizeof(int)*n_per_proc);
        uint32_t *DjC=(uint32_t*)malloc(sizeof(int)*n_per_proc);
        int *nDPaux=(int*)malloc(sizeof(int)*n_per_proc);


        /*for (int i=0; i<n_per_proc; i++){
        DjX0[i]=0;
        DjD[i]=0;
        Djsteps[i]=0;
        DjC[i]=0;
        nDPaux[i]=0;}*/

        //invocation of CUDA function
        int nthreads=MAX_THREAD;
        int nblocks=  n_per_proc/MAX_THREAD+1 ;

        findPathAndDP<<<nblocks, nthreads>>>(d_x0p, d_x,lg, n_per_proc, d_DjX0, d_DjD, d_Djsteps, d_DjC, d_nDPaux);


        //copies from device to host
        cudaErrorCheck(cudaMemcpy(DjX0, d_DjX0,sizeof(int)*n_per_proc, cudaMemcpyDeviceToHost), "Memcpy");
        cudaErrorCheck(cudaMemcpy(DjD, d_DjD,sizeof(int)*n_per_proc, cudaMemcpyDeviceToHost), "Memcpy");
        cudaErrorCheck(cudaMemcpy(Djsteps, d_Djsteps,sizeof(int)*n_per_proc, cudaMemcpyDeviceToHost), "Memcpy");
        cudaErrorCheck(cudaMemcpy(DjC, d_DjC,sizeof(int)*n_per_proc, cudaMemcpyDeviceToHost), "Memcpy");
        cudaErrorCheck(cudaMemcpy(nDPaux, d_nDPaux,sizeof(int)*n_per_proc, cudaMemcpyDeviceToHost), "Memcpy");

        //frees on device
        cudaFree(d_x0p);
        cudaFree(d_x);
        cudaFree(d_x0p);
        cudaFree(d_DjX0);
        cudaFree(d_DjD);
        cudaFree(d_Djsteps);
        cudaFree(d_DjC);
        cudaFree(d_nDPaux);

        //frees on host
        free(x0p);

        //calculates tot number of DP per process
        for (int i=0; i<n_per_proc; i++) nDPj+=nDPaux[i];
        printf("rank %d - tot DPs found in iteration %d \n", rank, nDPj);

        //flags processes that didn't find any DP
        int flag=0;
        if(nDPj==0 && rank!=ROOT) flag=1;

        //finds collisins (per process)
        int nCollisj=0;
        uint32_t *CjA=(uint32_t*)malloc(sizeof(int)*n_per_proc*(n_per_proc-1)/2);
        uint32_t *CjB=(uint32_t*)malloc(sizeof(int)*n_per_proc*(n_per_proc-1)/2);

        int ncjaux=0;
        for (int i = 0; i < n_per_proc; i++){
        for (int k = i+1; k<n_per_proc; k++){
        if(DjD[i]==DjD[k] && DjC[i]!=DjC[k]){

                printf("rank %d collision between %d and %d on %d on indx %d \n", rank, DjC[i], DjC[k], DjD[i], i);
                if (DjC[i]<DjC[k]){
                        CjA[ncjaux]=DjC[i];
                        CjB[ncjaux]=DjC[k];}
                else{
CjA[ncjaux]=DjC[k];
                        CjB[ncjaux]=DjC[i];}
                ncjaux++;
                }
        }}


        if ( ncjaux!=0) printf("rank %d - no of collisions %d \n" , rank, ncjaux);


        //eliminates duplicates (per process)
        uint32_t *scjA=(uint32_t*)malloc(sizeof(int)*n_per_proc*(n_per_proc-1)/2);
        uint32_t *scjB=(uint32_t*)malloc(sizeof(int)*n_per_proc*(n_per_proc-1)/2);
        for (int i=0; i<n_per_proc*(n_per_proc-1)/2 ; i++){
        scjA[i]=0;
        scjB[i]=0;}



        if (ncjaux>0){
                nCollisj=1;
                scjA[0]=CjA[0];
                scjB[0]=CjB[0];
                printf("rank %d -first collis %d and %d \n", rank, scjA[0], scjB[0]);

                for (int i = 1; i < ncjaux; i++){
                int a=0;
                        for (int k = 0; k<i; k++){
                        if(CjA[i]==CjA[k] && CjB[i]==CjB[k]){
                                a++;
                                break;}}
                        if (a==0){
                                scjA[nCollisj]=CjA[i];
                                scjB[nCollisj]=CjB[i];
                                printf("rank %d -collis bw %d and %d \n", rank, scjA[nCollisj], scjB[nCollisj]);
                                nCollisj++;}
                }
                printf("rank %d -no of unique collisions in iteration %d \n" , rank, nCollisj);
        }

        int nCollisT=0;

        //each process shares with root number of collisions found and related information
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Reduce(&nCollisj, &nCollisT, 1, MPI_INT, MPI_SUM, ROOT, MPI_COMM_WORLD);

        if (rank==ROOT) printf("reduce success %d \n", nCollisT);


        uint32_t *CA=(uint32_t*)malloc(sizeof(int)*M*(n_per_proc-1)/2);
        uint32_t *CB =(uint32_t*)malloc(sizeof(int)*M*(n_per_proc-1)/2);

        if (NULL==CA) {
        fprintf(stderr,"[ERROR][RANK %d] Cannot allocate memory\n",rank);
        MPI_Abort(MPI_COMM_WORLD,1);}

        for (int i=0; i<M*(n_per_proc-1)/2 ; i++){
        CA[i]=0;
        CB[i]=0;}


        MPI_Gather(scjA, n_per_proc*(n_per_proc-1)/2, MPI_INT, CA, n_per_proc*(n_per_proc-1)/2, MPI_INT, ROOT, MPI_COMM_WORLD);
        MPI_Gather(scjB, n_per_proc*(n_per_proc-1)/2, MPI_INT, CB, n_per_proc*(n_per_proc-1)/2, MPI_INT, ROOT, MPI_COMM_WORLD);
//each process shares with root the number of DPs found

        int nDP=0;
        uint32_t *DX0=(uint32_t*)malloc(sizeof(int)*M);
        uint32_t *DD =(uint32_t*)malloc(sizeof(int)*M);
        uint32_t *Dsteps =(uint32_t*)malloc(sizeof(int)*M);
        uint32_t *DC =(uint32_t*)malloc(sizeof(int)*M);

        for (int i=0; i<M ; i++){
        DX0[i]=0;
        DD[i]=0;
        Dsteps[i]=0;
        DC[i]=0;}

        MPI_Reduce(&nDPj, &nDP, 1, MPI_INT, MPI_SUM, ROOT, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);

        int key;
        if (flag==0) key= rank;
        else key=NP-rank;

        // Split the global communicator
        MPI_Comm new_comm;
        MPI_Comm_split(MPI_COMM_WORLD, flag, key, &new_comm);

        //each prcess that found DPs>0 shares with root the realted information
        MPI_Gather(DjX0, n_per_proc, MPI_INT, DX0, n_per_proc, MPI_INT, ROOT, new_comm);
        MPI_Gather(DjD, n_per_proc, MPI_INT, DD, n_per_proc, MPI_INT, ROOT, new_comm);
        MPI_Gather(Djsteps, n_per_proc, MPI_INT, Dsteps, n_per_proc, MPI_INT, ROOT, new_comm);
        MPI_Gather(DjC, n_per_proc, MPI_INT, DC , n_per_proc, MPI_INT, ROOT, new_comm);
        MPI_Comm_free(&new_comm);

        MPI_Barrier(MPI_COMM_WORLD);

        //frees on device       
        free(DjX0);
        free(DjD);
        free(Djsteps);
        free(DjC);
        free(scjA);
        free(scjB);


        //eliminates duplicates (globally)
        if(rank==ROOT) printf("Cumulative Collis till now %d \n", nCollisFinal);

        int nCollisTot=0;

        if (nCollisT>0){

                for (int i = 0; i < M*(n_per_proc-1)/2; i++){
                int a=0;
                int b=0;

                if (CB[i]!=0){
                        for (int k = 0; k<i; k++){
                        if(CA[i]==CA[k] && CB[i]==CB[k]){
                                a++;
                                break;}}
                        //printf( "a= %d on indx %d \n", a, i);
                        if (a==0){
                                for (int h = 0;h<nCollisFinal+1;h++){
                                if(CA[i]==scA[h] && CB[i]==scB[h]){
                                        b++;
break;}}
                                if (b==0){
                                        scA[nCollisFinal+nCollisTot]=CA[i];
                                        scB[nCollisFinal+nCollisTot]=CB[i];
                                        printf("new Collis bw %d and %d on indx %d \n", scA[nCollisFinal+nCollisTot],
                                        scB[nCollisFinal+nCollisTot], i);
                                        nCollisTot++;}
                        }
                }}
                printf("nSingleRank of this iteration %d \n", nCollisTot);
        }

        //each process is updated on the number of collisions found in this iteration
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Bcast(&nCollisTot,1, MPI_INT,ROOT,MPI_COMM_WORLD);

        //looks for new "interrank" collsions
        int nCollisIr=0;
        int nCollisIrT=0;
        uint32_t *tempA=(uint32_t*)malloc(sizeof(int)*M*n_per_proc*(NP-1)/2);
        uint32_t *tempB =(uint32_t*)malloc(sizeof(int)*M*n_per_proc*(NP-1)/2);

        for (int i = 0; i <M; i++){
        if (DC[i]==0 && DD[i]!=1) break;
        for (int k =n_per_proc+i/(n_per_proc); k<M; k++){
                int a=1;
                int b=1;

                if(DC[k]==0 && DD[k]!=1) break;
                if(DD[i]==DD[k] && DC[i]!=DC[k]){
                        a=0;
                        b=0;}

                if (a==0){
                        for (int h = 0;h<nCollisFinal+nCollisTot+1;h++){
                        if((DC[i]==scA[h] && DC[k]==scB[h]) || (DC[i]==scB[h] && DC[k]==scA[h])){
                                b++;
                                break;}}
                        if (b==0){
                                if (DC[i]<DC[k]){
                                        tempA[nCollisIrT]=DC[i];
                                        tempB[nCollisIrT]=DC[k];}
                                else{
                                        tempA[nCollisIrT]=DC[k];
                                        tempB[nCollisIrT]=DC[i];}
                                printf("new interrank %d and %d Collis bw %d and %d on %d \n", i, k,
                                tempA[nCollisIrT], tempB[nCollisIrT], DD[i]);

                                nCollisIrT++;
                        }
                }
        }}

        //eliminates duplicates between new collisions
        if (nCollisIrT>0){
                scA[nCollisFinal+nCollisTot]=tempA[0];
                scB[nCollisFinal+nCollisTot]=tempB[0];
                printf("first interrank collis bw %d and %d \n", scA[nCollisFinal+nCollisTot], scB[nCollisFinal+nCollisTot]);

                for (int i = 1; i < nCollisIrT; i++){
                int a=0;
                        for (int k = 0; k<i; k++){
                        if(tempA[i]==tempA[k] && tempB[i]==tempB[k]){
                                a++;
break;}}
                        if (a==0){
                                scA[nCollisFinal+nCollisTot+nCollisIr]=tempA[i];
                                scB[nCollisFinal+nCollisTot+nCollisIr]=tempB[i];
                                printf("interrank collis bw %d and %d \n", scA[nCollisFinal+nCollisTot+nCollisIr], scB[nCollisFinal+nCollisTot+nCollisIr]);
                                nCollisIr++;}
                }
        }

         MPI_Barrier(MPI_COMM_WORLD);

        //frees on device
        free(tempA);
        free(tempB);

        free(DX0);
        free(DD);
        free(Dsteps);
        free(DC);

        free(CA);
        free(CB);

        //tot number of new Collision found in this iterarion is calculated, tot number of Collisions found is updated and shared with all processes
        nCollisTot=nCollisTot+nCollisIr;
        if(rank==ROOT) printf("nTot of this iteration %d \n", nCollisTot);
        nCollisFinal= nCollisFinal+nCollisTot;

        MPI_Bcast(&nCollisFinal,1, MPI_INT,ROOT,MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank==ROOT) printf("nFin %d \n", nCollisFinal);

}//WHILE

//for last iteration
MPI_Barrier(MPI_COMM_WORLD);
MPI_Bcast(&nCollisFinal,1, MPI_INT,ROOT,MPI_COMM_WORLD);
if (rank==ROOT) {
        printf("nDef reached %d \n", nCollisFinal);
        for (int i=0; i<nCollisFinal; i++) printf ("%d and %d \n", scA[i], scB[i]);
}

MPI_Barrier(MPI_COMM_WORLD);
TIMER_STOP;

//save in csv subroutine
if (rank==ROOT) {
        printf("running time: %f microseconds\n",TIMER_ELAPSED);
        printf("Do you want to save the ouput in a csv? (0=no/1=yes) \n");


        int answ;
        scanf("%d", &answ);
        if(answ==1){
                FILE *fp;
                char filename[100];
                printf("Type file name \n ");
                //gets(filename);
                scanf("%99s", filename);
                strcat(filename,".csv");
                fp=fopen(filename,"w+");
                for(int i = 0; i<nCollisFinal; i++){
                        fprintf(fp,"\n %d,%d", scA[i], scB[i]);
                }
 fclose(fp);
        }

}

//frees on device 
free(scA);
free(scB);

MPI_Finalize();
return 0;

}



