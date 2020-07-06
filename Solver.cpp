//////////////////////////////////////////////////////////////////////////
////This is the code implementation for GPU  conjugate gradient solver
//////////////////////////////////////////////////////////////////////////
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <thrust/inner_product.h>
#include <thrust/device_ptr.h>
using namespace std;

//////////////////////////////////////////////////////////////////////////
////This project implements the conjugate gradient solver to solve sparse linear systems
////For the mathematics, please read https://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf
////The algorithm we are implementing is in Page 50, Algorithm B.2, the standard conjugate gradient (without a preconditioner)
//////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////
////These are the global variables that define the domain of the problem to solver (for both CPU and GPU)

const int NUMTHREADS = 32;

const int grid_size=256;										////grid size, we will change this value to up to 256 to test your code, notice that we do not have padding elements
const int nt = 64;
const int nb = (int)(grid_size/nt);
const int nt1 = 128;
const int nb1 = (int)(grid_size/nt1);
const int s=grid_size*grid_size;								////array size
#define I(i,j) ((i)*grid_size+(j))								////2D coordinate -> array index
#define B(i,j) (i)<0||(i)>=grid_size||(j)<0||(j)>=grid_size		////check boundary
const bool verbose=false;										////set false to turn off print for x and residual
const int max_iter_num=1000;									////max cg iteration number
const double tolerance=1e-3;									////tolerance for the iterative solver


//////////////////////////////////////////////////////////////////////////
//Dot product of two matrices
__global__ void DOT_GPU( double* a, double* b, const int n, float* res){

	const int numThreads = nb1*nt1;
	int threadId =  blockIdx.x * (nt1) + threadIdx.x;
	int currentElement = threadId;

	__shared__ double thread_sum[nt1];
	//thread_sum[threadIdx.x] = 0;

	double ts = 0;

	__syncthreads();

	while (currentElement < s){
		double v1 = a[currentElement];
		double v2 = b[currentElement];
		double val = v1 * v2; 
		ts += val;
		__syncthreads();
		currentElement += numThreads;
	}

	thread_sum[threadIdx.x] = ts;
	__syncthreads();

	//loop unrolling
	int unrollCounter = nt1/2;
	while (unrollCounter != 0) {
		if (threadIdx.x < unrollCounter){
			thread_sum[threadIdx.x]+=thread_sum[threadIdx.x + unrollCounter];
		}
		__syncthreads();
		unrollCounter /= 2;
	}

	__syncthreads();

	//Write
	if (threadIdx.x == 0){
		float vtoadd =(float)thread_sum[0]; 
		atomicAdd(res,vtoadd);
	}
}

__global__ void DOT_GPU( double* a, const int n, float* res){

	const int numThreads = nb1*nt1;
	int threadId =  blockIdx.x * (nt1) + threadIdx.x;
	int currentElement = threadId;Q

	__shared__ double thread_sum[nt1];
	//thread_sum[threadIdx.x] = 0;
	double ts = 0;
	__syncthreads();

	while (currentElement < s){
		double v = a[currentElement];
		double val = v*v; 
		ts += val;
		__syncthreads();
		currentElement += numThreads;
	}
	thread_sum[threadIdx.x] = ts;
	__syncthreads();

	//TO DO loop unrolling
	int unrollCounter = nt1/2;
	while (unrollCounter != 0) {
		if (threadIdx.x < unrollCounter){
			thread_sum[threadIdx.x]+=thread_sum[threadIdx.x + unrollCounter];
		}
		__syncthreads();
		unrollCounter /= 2;
	}

	__syncthreads();

	//Write
	if (threadIdx.x == 0){
		float vtoadd =(float)thread_sum[0]; 
		atomicAdd(res,vtoadd);
	}
}


__global__ void Addition_GPU( double* a, double* b, const double multiplier, const int n, double* res){

	const int numThreads = nb*nt;
	int threadId =  blockIdx.x * (nt) + threadIdx.x;
	//int threadId = threadIdx.x;
	int currentElement = threadId;

	while (currentElement < n){
		double val1 = a[currentElement]; 
		__syncthreads();
		double val2 = b[currentElement]; 
		__syncthreads();
		double final = val1 + multiplier * val2;
		res[currentElement] = final; 
		__syncthreads();
		currentElement += numThreads;
	}
}


__global__ void Subtract_GPU( double* a, double* b, const double multiplier, const int n, double* res){

	const int numThreads = nb*nt;
	int threadId =  blockIdx.x * (nt) + threadIdx.x;
	int currentElement = threadId;

	while (currentElement < n){
		double val1 = a[currentElement]; 
		__syncthreads();
		double val2 = b[currentElement]; 
		__syncthreads();
		double final = val1 - multiplier * val2; 
		res[currentElement] = final;
		__syncthreads();

		currentElement += numThreads;
	}
}


__global__ void MV_GPU(/*CRS sparse matrix*/const double* val,const int* col,const int* ptr,/*number of column*/const int n,/*input vector*/const double* v,/*result*/double* mv)
{
	int numThreads = blockDim.x * blockDim.y* blockDim.z;
	int blockId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
	int threadId = threadIdx.x;

	__shared__ double sum[NUMTHREADS];

	__shared__ int ptr_start;
	__shared__ int ptr_end;
	__shared__ int numElementsInRow;

	double thread_sum;
	thread_sum = 0;

	if (threadIdx.x == 0){
		//Make this faster
		//double *temp = (double*)ptr;
		//double ptr_vals;


		//sum = 0;
		ptr_start= ptr[blockIdx.x];
		ptr_end= ptr[blockIdx.x + 1];

		//printf("%d %d %d \n", blockIdx.x,ptr_start, ptr_end );

		numElementsInRow = ptr_end - ptr_start;
	}

	__syncthreads();

	int currentElement = threadId;

	int numEls = numElementsInRow;
	int pstart = ptr_start;

	while (currentElement < numEls){
		int e = pstart +currentElement;
		int current_Col = col[e];
		double current_Val = val[e];
		double current_vector_val = v[current_Col];

		thread_sum += current_Val * current_vector_val;



		currentElement+= numThreads;
	}

	sum[threadId] = thread_sum;

	__syncthreads();

	//loop unrolling
	int unrollCounter = numThreads/2;
	while (unrollCounter != 0) {
		if (threadId < unrollCounter){
			sum[threadId]+=sum[threadId + unrollCounter];
		}
		__syncthreads();
		unrollCounter /= 2;
	}


	__syncthreads();

	//Write
	if (threadIdx.x == 0){
		mv[blockId] = sum[0];
	}



}


__global__ void B_minus_AX_GPU(const double* b, /*CRS sparse matrix*/const double* val,const int* col,const int* ptr,/*number of column*/const int n,/*input vector*/const double* v,/*result*/double* mv)
{
	int numThreads = blockDim.x * blockDim.y* blockDim.z;
	int blockId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
	int threadId = threadIdx.x;

	__shared__ double sum[NUMTHREADS];

	__shared__ int ptr_start;
	__shared__ int ptr_end;
	__shared__ int numElementsInRow;

	double thread_zero_b;
	thread_zero_b = 0;

	double thread_sum;
	thread_sum = 0;

	if (threadIdx.x == 0){
		//Make this faster
		//double *temp = (double*)ptr;
		//double ptr_vals;


		//sum = 0;
		ptr_start= ptr[blockId];
		ptr_end= ptr[blockId + 1];

		//printf("%d %d %d \n", blockIdx.x,ptr_start, ptr_end );

		thread_zero_b = b[blockId];

		numElementsInRow = ptr_end - ptr_start;
	}

	__syncthreads();


	int currentElement = threadId;

	while (currentElement < numElementsInRow){
		int current_Col = col[ptr_start +currentElement];
		double current_Val = val[ptr_start +currentElement];
		double current_vector_val = v[current_Col];

		thread_sum += current_Val * current_vector_val;



		currentElement+= numThreads;
	}

	sum[threadId] = thread_sum;

	__syncthreads();


	//loop unrolling
	int unrollCounter = numThreads/2;
	while (unrollCounter != 0) {
		if (threadId < unrollCounter){
			sum[threadId]+=sum[threadId + unrollCounter];
		}
		__syncthreads();
		unrollCounter /= 2;
	}


	__syncthreads();


	//Write
	if (threadIdx.x == 0){
		//printf("Block %d B %lf\n",blockId, thread_zero_b );
		mv[blockId] = thread_zero_b - sum[0];
	}



}



__global__ void D_dot_AX_GPU(const double* d, /*CRS sparse matrix*/const double* val,const int* col,const int* ptr,/*number of column*/const int n,/*input vector*/const double* v,/*result*/float* answer)
{
	int numThreads = blockDim.x * blockDim.y* blockDim.z;
	int blockId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
	int threadId = threadIdx.x;

	__shared__ double sum[NUMTHREADS];

	__shared__ int ptr_start;
	__shared__ int ptr_end;
	__shared__ int numElementsInRow;

	double thread_zero_d;

	double thread_sum;
	thread_sum = 0;

	if (threadIdx.x == 0){
		//Make this faster
		//double *temp = (double*)ptr;
		//double ptr_vals;


		//sum = 0;
		ptr_start= ptr[blockId];
		ptr_end= ptr[blockId + 1];

		//printf("%d %d %d \n", blockIdx.x,ptr_start, ptr_end );

		thread_zero_d = d[blockId];

		numElementsInRow = ptr_end - ptr_start;
	}

	__syncthreads();

	int currentElement = threadId;

	while (currentElement < numElementsInRow){
		int current_Col = col[ptr_start +currentElement];
		double current_Val = val[ptr_start +currentElement];
		double current_vector_val = v[current_Col];

		thread_sum += current_Val * current_vector_val;



		currentElement+= numThreads;
	}

	sum[threadId] = thread_sum;

	__syncthreads();

	//loop unrolling
	int unrollCounter = numThreads/2;
	while (unrollCounter != 0) {
		if (threadId < unrollCounter){
			sum[threadId]+=sum[threadId + unrollCounter];
		}
		__syncthreads();
		unrollCounter /= 2;
	}


	__syncthreads();

	//Write
	if (threadIdx.x == 0){
		float blockSum = (float)(thread_zero_d * sum[0]);
		atomicAdd(answer, blockSum);
	}

}


void Conjugate_Gradient_Solver_GPU(const double* val,const int* col,const int* ptr,const int n,		////A is an n x n sparse matrix stored in CRS format
								double* r,double* q,double* d,									////intermediate variables
								double* x,const double* b,										////x and b
								const int max_iter,const double tol)
{

	////declare variables
	int iter=0;
	float delta_old=0.0;
	float delta_new=0.0;
	double alpha=0.0;
	double beta=0.0;


	int nnz = ptr[n];
	double* val_gpu;
	cudaMalloc(&val_gpu, nnz*sizeof(double));
	cudaMemcpy( val_gpu,val, nnz*sizeof(double) , cudaMemcpyHostToDevice);
	int* col_gpu;
	cudaMalloc(&col_gpu, nnz*sizeof(int));
	cudaMemcpy( col_gpu,col, nnz*sizeof(int) , cudaMemcpyHostToDevice);
	int* ptr_gpu;
	cudaMalloc(&ptr_gpu, (n+1)*sizeof(int));
	cudaMemcpy( ptr_gpu,ptr, (n+1)*sizeof(int) , cudaMemcpyHostToDevice);
	double* x_gpu;
	cudaMalloc(&x_gpu, n*sizeof(double));
	cudaMemcpy( x_gpu,x, n*sizeof(double) , cudaMemcpyHostToDevice);
	double* r_gpu;
	cudaMalloc(&r_gpu, n*sizeof(double));
	cudaMemcpy( r_gpu,r, n*sizeof(double) , cudaMemcpyHostToDevice);
	double* b_gpu;
	cudaMalloc(&b_gpu, n*sizeof(double));
	cudaMemcpy( b_gpu,b, n*sizeof(double) , cudaMemcpyHostToDevice);


	////: r=b-Ax
	B_minus_AX_GPU<<<s,NUMTHREADS>>>(b_gpu, val_gpu,col_gpu,ptr_gpu,s,x_gpu,r_gpu);

	cudaMemcpy( r,r_gpu, n*sizeof(double) , cudaMemcpyDeviceToHost);


	////d=r
	memcpy(d,r,sizeof(double)*n);



	double* q_gpu;
	cudaMalloc(&q_gpu, n*sizeof(double));


	double* d_gpu;
	cudaMalloc(&d_gpu, n*sizeof(double));
	cudaMemcpy( d_gpu,d, n*sizeof(double) , cudaMemcpyHostToDevice);

	float* dTq_gpu;
	cudaMalloc(&dTq_gpu, sizeof(float));

	float* delta_new_gpu;
	cudaMalloc(&delta_new_gpu, sizeof(float));

	////: delta_new=rTr
	cudaMemset(delta_new_gpu,0x0000,sizeof(float));
	DOT_GPU<<<nb1,nt1>>>(r_gpu, r_gpu, n, delta_new_gpu);
	cudaMemcpy( &delta_new,delta_new_gpu, sizeof(float) , cudaMemcpyDeviceToHost);

	while(iter<max_iter&& delta_new>tol){	
		////: q=Ad
		//memset(q,0x0000,sizeof(double)*n);
		float dTq;
		MV_GPU<<<s,NUMTHREADS>>>(val_gpu,col_gpu,ptr_gpu,s,d_gpu,q_gpu);

		cudaMemset(dTq_gpu,0x0000,sizeof(float));
		DOT_GPU<<<nb1,nt1>>>(d_gpu, q_gpu, n, dTq_gpu);
		cudaMemcpy( &dTq,dTq_gpu, sizeof(float) , cudaMemcpyDeviceToHost);

		//D_dot_AX_GPU<<<s,32>>>(d_gpu, val_gpu,col_gpu,ptr_gpu,n,q_gpu,dTq_gpu);
		//cudaMemcpy( &dTq,dTq_gpu, sizeof(float) , cudaMemcpyDeviceToHost);

		////: alpha=delta_new/d^Tq
		alpha = delta_new/dTq;

		//printf("Alpha %d %lf\n",iter, alpha );

		////: x=x+alpha*d
		Addition_GPU<<<nb,nt>>>(x_gpu,d_gpu, alpha,n, x_gpu);


		if(iter%50==0&&iter>1){
			////: r=b-Ax
			B_minus_AX_GPU<<<s,NUMTHREADS>>>(b_gpu, val_gpu,col_gpu,ptr_gpu,s,x_gpu,r_gpu);
			//cudaMemcpy( r,r_gpu, n*sizeof(double) , cudaMemcpyDeviceToHost);
		}else{
			////: r=r-alpha*q
			Subtract_GPU<<<nb,nt>>>(r_gpu,q_gpu,alpha,n,r_gpu);
			//cudaMemcpy( r,r_gpu, n*sizeof(double) , cudaMemcpyDeviceToHost);
		}

		////: delta_old=delta_new
		delta_old = delta_new;

		////: delta_new=r^Tr

		//delta_new = Dot(r, r, n);
		cudaMemset(delta_new_gpu,0x0000,sizeof(float));
		DOT_GPU<<<nb1,nt1>>>(r_gpu, n, delta_new_gpu);
		cudaMemcpy( &delta_new,delta_new_gpu, sizeof(float) , cudaMemcpyDeviceToHost);

		////: beta=delta_new/delta_old
		beta = delta_new/delta_old;
		
		////: d=r+beta*d
		Addition_GPU<<<nb,nt>>>(r_gpu,d_gpu,beta,n,d_gpu);
		//cudaMemcpy( d,d_gpu, n*sizeof(double) , cudaMemcpyDeviceToHost);


		//printf("New Delta %lf\n",delta_new );

		////: increase the counter
		iter++;
	}
	// if(iter<max_iter)
	// 	cout<<"G conjugate gradient solver converges after "<<iter<<" iterations with residual "<<(delta_new)<<endl;
	// else 
	// 	cout<<"GPU conjugate gradient solver does not converge after "<<max_iter<<" iterations with residual "<<(delta_new)<<endl;

	cudaMemcpy( x,x_gpu, n*sizeof(double) , cudaMemcpyDeviceToHost);
	cudaMemcpy( r,r_gpu, n*sizeof(double) , cudaMemcpyDeviceToHost);

}

//////////////////////////////////////////////////////////////////////////


////calculate mv=M*v, here M is a square matrix
void MV(/*CRS sparse matrix*/const double* val,const int* col,const int* ptr,/*number of column*/const int n,/*input vector*/const double* v,/*result*/double* mv)
{
	/*Your implementation starts*/

	//Sparse matrix Dense vector
	//Transverse all the nonzeroes in row i and get their value and column idices
	for (int row=0; row <n; row++){
		int startIndex = ptr[row];
		int endIndex = ptr[row+1];
		// printf("%d\n",endIndex - startIndex );
		for (int i= startIndex; i<endIndex; i++){
			int colIndex = col[i];
			double value = val[i];
			mv[row] += value*v[colIndex];
		}

	}

	/*Your implementation ends*/
}

////return the dot product between a and b
double Dot(const double* a,const double* b,const int n)
{
	/*Your implementation starts*/
	double d = 0;
	for (int i=0; i <n; i++){
		d+= a[i]*b[i];
	}
	return d;
	/*Your implementation ends*/
}

void Subtract(const double* a,const double* b,const double multiplier, const int n, double* res)
{

	for (int i = 0; i < n; i++){
	        res[i] = a[i] - multiplier * b[i];  
	}

	return;
}
void Addition( double* a, double* b, double multiplier, const int n, double* res)
{

	for (int i = 0; i < n; i++){
	        res[i] = a[i] + multiplier * b[i];    
	}

	return;
}

//////////////////////////////////////////////////////////////////////////
//// 2: Warm up practice 2 -- implement a CPU-based conjugate gradient solver based on the painless PCG course notes to solve Ax=b
////Please read the notes and implement all the s in the function

void Conjugate_Gradient_Solver(const double* val,const int* col,const int* ptr,const int n,		////A is an n x n sparse matrix stored in CRS format
								double* r,double* q,double* d,									////intermediate variables
								double* x,const double* b,										////x and b
								const int max_iter,const double tol)							////solver parameters
{
	////declare variables
	int iter=0;
	double delta_old=0.0;
	double delta_new=0.0;
	double alpha=0.0;
	double beta=0.0;

	////: r=b-Ax
	//r = Ax
	memset(r,0x0000,sizeof(double)*n);
	MV(val,col,ptr,n,x,r);

	//r = b - Ax
	Subtract(b,r,1, n,r);

	////: d=r
	memcpy(d,r,sizeof(double)*n);
	
	////: delta_new=rTr
	delta_new=Dot(r, r, n);

	
	////Here we use the absolute tolerance instead of a relative one, which is slightly different from the notes
	while(iter<max_iter&& delta_new>tol){	
		////: q=Ad
		memset(q,0x0000,sizeof(double)*n);
		MV(val, col, ptr, n, d, q);


		////: alpha=delta_new/d^Tq
		double dTq;
		dTq = Dot(d, q, n);
		alpha = delta_new/dTq;
		//printf("\nCPU: alpha: %lf\n", alpha);

		////: x=x+alpha*d
		Addition(x,d, alpha,n, x);

		if(iter%50==0&&iter>1){
			////: r=b-Ax
			memset(r,0x0000,sizeof(double)*n);
			MV(val,col,ptr,n,x,r);
			Subtract(b,r,1,n,r);
		}
		else{
			////: r=r-alpha*q
			//printf("%s\n","here" );
			Subtract(r,q,alpha,n,r);
		}

		////: delta_old=delta_new
		delta_old = delta_new;

		////: delta_new=r^Tr
		delta_new = Dot(r, r, n);
		//printf("Delta %lf\n",delta_new );

		////: beta=delta_new/delta_old
		beta = delta_new/delta_old;
		
		////: d=r+beta*d
		Addition(r,d, beta,n, d);


		////: increase the counter
		iter++;
	}

	if(iter<max_iter)
		cout<<"CPU conjugate gradient solver converges after "<<iter<<" iterations with residual "<<(delta_new)<<endl;
	else 
		cout<<"CPU conjugate gradient solver does not converge after "<<max_iter<<" iterations with residual "<<(delta_new)<<endl;
}



ofstream out;

//////////////////////////////////////////////////////////////////////////
////Test functions
////Here we setup a test example by initializing the same Poisson problem as in the last competition: -laplace(p)=b, with p=x^2+y^2 and b=-4.
////The boundary conditions are set on the one-ring ghost cells of the grid
////There is nothing you need to implement in this function

void Initialize_2D_Poisson_Problem(vector<double>& val,vector<int>& col,vector<int>& ptr,vector<double>& b)
{
	////assemble the CRS sparse matrix
	////The grid dimension is grid_size x grid_size. 
	////The matrix's dimension is s x s, with s= grid_size*grid_size.
	////We also initialize the right-hand vector b

	val.clear();
	col.clear();
	ptr.resize(s+1,0);
	b.resize(s,-4.);

	for(int i=0;i<grid_size;i++){
		for(int j=0;j<grid_size;j++){
			int r=I(i,j);
			int nnz_for_row_r=0;

			////set (i,j-1)
			if(!(B(i,j-1))){
				int c=I(i,j-1);
				val.push_back(-1.);
				col.push_back(c);
				nnz_for_row_r++;
			}
			else{
				double boundary_val=(double)(i*i+(j-1)*(j-1));	
				b[r]+=boundary_val;
			}

			////set (i-1,j)
			if(!(B(i-1,j))){
				int c=I(i-1,j);
				val.push_back(-1.);
				col.push_back(c);
				nnz_for_row_r++;
			}
			else{
				double boundary_val=(double)((i-1)*(i-1)+j*j);
				b[r]+=boundary_val;
			}

			////set (i+1,j)
			if(!(B(i+1,j))){
				int c=I(i+1,j);
				val.push_back(-1.);
				col.push_back(c);
				nnz_for_row_r++;
			}
			else{
				double boundary_val=(double)((i+1)*(i+1)+j*j);
				b[r]+=boundary_val;
			}

			////set (i,j+1)
			if(!(B(i,j+1))){
				int c=I(i,j+1);
				val.push_back(-1.);
				col.push_back(c);
				nnz_for_row_r++;
			}
			else{
				double boundary_val=(double)(i*i+(j+1)*(j+1));
				b[r]+=boundary_val;
			}

			////set (i,j)
			{
				val.push_back(4.);
				col.push_back(r);
				nnz_for_row_r++;
			}
			ptr[r+1]=ptr[r]+nnz_for_row_r;
		}
	}
}

//////////////////////////////////////////////////////////////////////////
////CPU test function
////There is nothing you need to implement in this function
void Test_CPU_Solvers()
{
	vector<double> val;
	vector<int> col;
	vector<int> ptr;
	vector<double> b;
	Initialize_2D_Poisson_Problem(val,col,ptr,b);

	vector<double> x(s,0.);
	vector<double> r(s,0.);
	vector<double> q(s,0.);
	vector<double> d(s,0.);
	
	auto start=chrono::system_clock::now();

	Conjugate_Gradient_Solver(&val[0],&col[0],&ptr[0],s,
								&r[0],&q[0],&d[0],
								&x[0],&b[0],
								max_iter_num,tolerance);

	auto end=chrono::system_clock::now();
	chrono::duration<double> t=end-start;
	double cpu_time=t.count()*1000.;	

	if(verbose){
		cout<<"\n\nx for CG on CPU:\n";
		for(int i=0;i<s;i++){
			cout<<x[i]<<", ";
		}	
	}
	cout<<"\n\n";

	//////calculate residual
	MV(&val[0],&col[0],&ptr[0],s,&x[0],&r[0]);
	for(int i=0;i<s;i++)r[i]=b[i]-r[i];
	double residual=Dot(&r[0],&r[0],s);
	cout<<"\nCPU time: "<<cpu_time<<" ms"<<endl;
	cout<<"Residual for your CPU solver: "<<residual<<endl;

	out<<"R0: "<<residual<<endl;
	out<<"T0: "<<cpu_time<<endl;
}

//////////////////////////////////////////////////////////////////////////
////GPU test function
void Test_GPU_Solver()
{
	vector<double> val;
	vector<int> col;
	vector<int> ptr;
	vector<double> b;
	Initialize_2D_Poisson_Problem(val,col,ptr,b);

	vector<double> x(s,0.);
	vector<double> r(s,0.);
	vector<double> q(s,0.);
	vector<double> d(s,0.);


	cudaEvent_t start,end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	float gpu_time=0.0f;
	cudaDeviceSynchronize();
	cudaEventRecord(start);

	//////////////////////////////////////////////////////////////////////////
	//// 4: call your GPU functions here
	////Requirement: You need to copy data from the CPU arrays, conduct computations on the GPU, and copy the values back from GPU to CPU
	

Conjugate_Gradient_Solver_GPU(&val[0],&col[0],&ptr[0],s,		////A is an n x n sparse matrix stored in CRS format
								&r[0],&q[0],&d[0],									////intermediate variables
								&x[0],&b[0],										////x and b
								max_iter_num,tolerance);


	////The final variables should be stored in the same place as the CPU function, i.e., the array of x
	////The correctness of your simulation will be evaluated by the residual (<1e-3)
	//////////////////////////////////////////////////////////////////////////

	cudaEventRecord(end);
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&gpu_time,start,end);
	cudaEventDestroy(start);
	cudaEventDestroy(end);
	//////////////////////////////////////////////////////////////////////////


	//  check for error
	  cudaError_t error = cudaGetLastError();
	  if(error != cudaSuccess)
	  {
	    // print the CUDA error message and exit
	    printf("CUDA error: %s\n", cudaGetErrorString(error));
	    exit(-1);
	  }

	if(verbose){
		cout<<"\n\nx for CG on GPU:\n";
		for(int i=0;i<s;i++){
			cout<<x[i]<<", ";
		}	
	}
	cout<<"\n\n";

	//////calculate residual
	MV(&val[0],&col[0],&ptr[0],s,&x[0],&r[0]);
	for(int i=0;i<s;i++)r[i]=b[i]-r[i];
	double residual=Dot(&r[0],&r[0],s);
	cout<<"\nGPU time: "<<gpu_time<<" ms"<<endl;
	cout<<"Residual for your GPU solver: "<<residual<<endl;

	out<<"R1: "<<residual<<endl;
	out<<"T1: "<<gpu_time<<endl;
}

int main()
{

	Test_CPU_Solvers();
	Test_GPU_Solver();

	return 0;
}
