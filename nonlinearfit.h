/*
 * nonlinearfit.h


 *
 *  Created on: Jan 26, 2015
 *      Author: Brian Donohue
 */


#ifndef NONLINEARFIT_H_
#define NONLINEARFIT_H_
template <class value_type> class nonlinearfit{

	//This will stores the Jacobian functions
	typedef value_type (*functions)(value_type, value_type*);




	public:

	//Number of data points
	  unsigned int n;

	//Number of params
	  unsigned int p;

	//Function to be fitted. Be sure to assign this a function prior to calling solve
	  value_type (*F)(value_type, value_type*);

	//Jacobian functions used for fit. Be sure to assign this functions prior to calling solve
	  functions * Jacobianfunctions;

	  /*Initial mu values as well as the values
	  that mu reverts to when an iteration fails*/
	  value_type mu1Inc;
	  value_type mu2Inc;

	  //Scalar that multplies the diag(JTJ) matrix by
	  value_type  lambda1;

	  //Scalar that multiplies the JT(Y-f(B) vector by
	  value_type  lambda2;

	  //The mu values control the rate at which the lambdas increase
	  value_type  mu1;

	  value_type  mu2;




	//Y values
	  value_type * Y;

	//X values
	  value_type * X;

	  //Holds the function parameter values
	  value_type * B;

	  //Holds the increment increase of parameter values after an iteration
	  value_type * delta;



	  __device__ nonlinearfit(unsigned int nD, unsigned int  nP){
		  n = nD;
		  p = nP;




		  Jacobianfunctions = new functions[nP];

		  X = new value_type[nD];
		  Y = new value_type[nD];
		  B = new value_type[nP];

		  delta = new value_type[nP];


	  }


	 __device__  ~nonlinearfit(){
		  delete [] X;
		  delete [] B;
		  delete [] delta;
		  delete Jacobianfunctions;


	  }

	  //Initializes the rate of convergence scalars
	 __device__  void setInitialIterIncr(value_type lambda1Ini, value_type lambda2Ini,
			  value_type mu1Ini, value_type mu2Ini, value_type mu1Incn, value_type mu2Incn){


		  lambda1 = lambda1Ini;
		  lambda2 = lambda2Ini;

		  mu1 = mu1Ini;
		  mu2 = mu2Ini;
		  mu1Inc = mu1Incn;
		  mu2Inc = mu2Incn;

	  }

	  //Calculate chi ,which determines whether an iteration is successful as well as controlling when iterations should stop
	 __device__  value_type calcChi(value_type * rB){
		  value_type sum;
		  for(int i = 0 ; i < n; i++){
			  sum += Y[i]*Y[i] - 2.f*Y[i]*F(X[i], rB) + F(X[i], rB)*F(X[i], rB);
		  }
		  return sum;

	  }
	  //Performs a single solver iteration
	 __device__  bool solveIter(){



		  value_type **JTJ;
		  value_type * JTY;
		  JTY = new value_type[p];
		  JTJ = new value_type*[p];
		  for(int i = 0 ; i < p ; i++){
			  JTJ[i] = new value_type[p];
		  }

		  generateMatrix(JTJ, JTY);

		  solveMatrix(JTJ, JTY, delta);
		  delete [] JTJ;
		  delete [] JTY;
		  value_type iniChi = calcChi(B);

		  value_type * nB = new value_type[p];
		  for(int m = 0 ; m < p ; m++){
			  nB[m] = B[m] + delta[m];
		  }

		  value_type newChi = calcChi(nB);

		  if(newChi < iniChi){
			  lambda1 = lambda1*mu1;
			  lambda2 = lambda2*mu2;
			  mu1 = mu1*mu1Inc;
			  mu2 = mu2*mu2Inc;

			  for(int m = 0 ; m < p ; m++){
				  B[m] = nB[m];

			  }

			  delete [] nB;
			  return true;
		  }else{

			  lambda1 = lambda1/mu1;
			  lambda2 = lambda2/mu2;
			  mu1 = mu1Inc;
			  mu2 = mu2Inc;
			  delete [] nB;
			  return false;
		  }

	  }

	  //Performs an entire fit until precision is reached or the number of iterations reaches max
	 __device__  bool solve(value_type precision, int max){
		  int iter = 0;
		  bool success;
		  do{
			  success = solveIter();
			  iter += 1;
		  }while((iter < max) && (calcChi(B) > precision));


		  return success;
	  }



	private:

	  //Solves the matrix equation using LU factorization
	 __device__  void solveMatrix(value_type ** A, value_type * b, value_type * x){
			float L[p][p];
			float U[p][p];



			for(int r = 0; r < p; r++){
				for(int c = 0 ; c < p ; c++){
					if(r >= c){
					if(c == r){
						L[r][c] = 1.f;
					}else{
						float sum = A[r][c];
						for(int s = 0; s < c; s++){
							sum += -L[r][s]*U[s][c];
						}
						sum = sum/U[c][c];
						L[r][c] = sum;
					}
					}else{
						L[r][c] = 0.f;
					}
					if(c >= r){
						float sum = A[r][c];
						for(int s = 0 ; s < r ; s++){
							sum += -L[r][s]*U[s][c];
						}

						U[r][c] = sum;
					}else{
						U[r][c] = 0.f;
					}

				}

			}
			float y[3];

			for(int r = 0; r < p; r++){
				float sum = b[r];
				for(int c = 0; c < r ; c++){
					sum += -y[c]*L[r][c];
				}
				y[r] = sum;
			}

			for(int r = p - 1; r >= 0 ; r = r - 1){
				float sum = y[r];
				for(int c = p - 1 ; c > r  ; c = c - 1){
					sum += -x[c]*U[r][c];
				}
				sum = sum/U[r][r];
				x[r] = sum;
			}
	  }

	  //Creates the matrix equation for an iteration
	 __device__  void generateMatrix(value_type ** A, value_type * b){
		  for(int r = 0 ; r < p ; r++){
			  for(int c = 0 ; c < p ; c++){
				  value_type JTJsum = 0.f;
				  value_type JTYsum = 0.f;
				  for(int w = 0; w < n; w++){
					  if(r == 0){
						  JTYsum += (Y[w] - F(X[w], B))*Jacobianfunctions[c](X[w], B);
					  }
					  JTJsum += Jacobianfunctions[c](X[w], B)*Jacobianfunctions[r](X[w], B);
				  }

				  if(r == 0){
					  b[c] = lambda2*JTYsum;
				  }
				  A[r][c] = JTJsum;
			  }

		  }

		  A[0][0] += A[0][0]*lambda1;
		  A[1][1] += A[1][1]*lambda1;
		  A[2][2] += A[2][2]*lambda1;
	  }







};


#endif /* NONLINEARFIT_H_ */
