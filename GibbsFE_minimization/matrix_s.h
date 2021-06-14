//
//  matrix.cpp
//  Define Class for Vector & Matrix
//
//  Created by Yoshi Miyazaki on 2015/04/11.
//

#include "matrix.h"
/*----------------------------------------
 Vector Types Constructers
 ---------------------------------------*/
template<class T>
Vector1d<T>::Vector1d(){
    n = 0;
    v = 0;
}
template<class T>
Vector1d<T>::Vector1d(int nn){
    n = nn;
    v = new T[n];
}
template<class T>
Vector1d<T>::Vector1d(const T& a, int nn){
    n = nn;
    v = new T[nn];
    for (int i=0; i<nn; i++){
        v[i] = a;
    }
}
template<class T>
Vector1d<T>::Vector1d(const T* a, int nn){
    n = nn;
    v = new T[n];
    for (int i=0; i<nn; i++){
        v[i] = *a++;
    }
}
template<class T>
Vector1d<T>::Vector1d(const Vector1d<T> &copy){
    n = copy.n;
    v = new T[n];
    for (int i=0; i<n; i++){
        v[i] = copy[i];
    }
}
/*----------------------------------------
 Operater
 ---------------------------------------*/
template<class T>                                            // Substitution
Vector1d<T>& Vector1d<T>::operator=(const Vector1d<T> &copy){
    if (this != &copy){
        if (n != copy.n){
            if (v != 0) delete[] v;
            n = copy.n;
            v = new T[n];
        }
        for (int i=0; i<n; i++){
            v[i] = copy[i];
        }
    }
    return *this;
}
template<class T>                                            // i'th element
Vector1d<T>& Vector1d<T>::operator=(const T &a){
    for (int i=0; i<n; i++){
        v[i] = a;
    }
    return *this;
}
template<class T>
const bool Vector1d<T>::operator==(const Vector1d<T>& rhs) const{
    if (n != rhs.n){
        return 0;
    }
    else{
        bool b = 1;
	for (int i=0; i<n; i++){
	    if (v[i] != rhs[i]){
	        b = 0;
		break; 
	    }
	}
	return b;
    }
}
template<class T>
void Vector1d<T>::resize(int nn){
    if (n != nn){
        if (v != 0){
            delete[] v;
        }
        n = nn;
        v = new T[n];
    }
}
template<class T>
void Vector1d<T>::resize(const T& a, int nn){
    if (n != nn){
        if (v != 0){
            delete[] v;
        }
        n = nn;
        v = new T[n];
    }
    for (int i=0; i<n; i++){
        v[i] = a;
    }
}


/*----------------------------------------
 Mathematical Operater
 ---------------------------------------*/
template<class T>
const T Vector1d<T>::norm() const{
    T norm = 0;
    for (int i=0; i<n; i++){
        norm += v[i]*v[i];
    }
    return sqrt(norm);
}
template<class T>
const T Vector1d<T>::maxv() const{
    T maxv = v[0];
    for (int i=1; i<n; i++){
	if (maxv < v[i]){maxv = v[i];}
    }
    return maxv;
}
template<class T>
const T Vector1d<T>::minv() const{
    T minv = v[0];
    for (int i=1; i<n; i++){
	if (minv > v[i]){minv = v[i];}
    }
    return minv;
}
template<class T>
const T Vector1d<T>::average() const{
    T ave = 0;
    for (int i=0; i<n; i++){
	ave += v[i];
    }
    return ave/double(n);
}
template<class T> /* maximum of abs(v[i]) */
const T Vector1d<T>::absmaxv() const{
    T maxv = abs(v[0]);
    for (int i=1; i<n; i++){
	if (maxv < abs(v[i])){maxv = abs(v[i]);}
    }
    return maxv;
}
template<class T> /* minimum of abs(v[i]) */
const T Vector1d<T>::absminv() const{
    T minv = abs(v[0]);
    for (int i=1; i<n; i++){
	if (minv > abs(v[i])){minv = abs(v[i]);}
    }
    return minv;
}
template<class T> /* minimum of abs(v[i]) */
const T Vector1d<T>::absnon0minv() const{
    T minv = 1e100;
    for (int i=0; i<n; i++){
      if ((minv > abs(v[i])) && (v[i] != 0)){minv = abs(v[i]);}
    }
    return minv;
}
template<class T> /* average of abs(v[i]) */
const T Vector1d<T>::absaverage() const{
    T ave = 0;
    for (int i=0; i<n; i++){
	ave += (v[i]>0 ? v[i] : -1.0*v[i]);
    }
    return ave/double(n);
}
template<class T> /* dot product */
const T Vector1d<T>::operator*(const Vector1d<T>& A){
    int nA;    nA = A.size();
    T   dotp = 0;

    if (nA != n){
        cout << "size of vectors don't match. Revise your input." << endl;
        exit(7);
    }
    else{
        for (int i=0; i<n; i++){
            dotp += v[i]*A[i];
        }
        return dotp;
    }
}
template<class T>
Vector1d<T> Vector1d<T>::operator+(const Vector1d<T>& A){
    int nA;
    nA = A.size();
    
    if (nA != n){
        cout << "size of vectors don't match. Revise your input." << endl;
        exit(7);
    }
    else{
        Vector1d<double> sum(n);
        for (int i=0; i<n; i++){
            sum[i] = v[i] + A[i];
        }
        return sum;
    }
}
template<class T>
Vector1d<T> Vector1d<T>::operator-(const Vector1d<T>& A){
    int nA;
    nA = A.size();
    
    if (nA != n){
        cout << "size of vectors don't match. Revise your input." << endl;
        exit(7);
    }
    else{
        Vector1d<double> sum(n);
        for (int i=0; i<n; i++){
            sum[i] = v[i] - A[i];
        }
        return sum;
    }
}
template<class T>
Vector1d<T> Vector1d<T>::operator+(const T& A){
    Vector1d<double> sum(n);
    for (int i=0; i<n; i++){
        sum[i] = v[i] + A;
    }
    return sum;
}
template<class T>
Vector1d<T> Vector1d<T>::operator-(const T& A){
    Vector1d<double> sum(n);
    for (int i=0; i<n; i++){
        sum[i] = v[i] - A;
    }
    return sum;
}
template<class T>
Vector1d<T> Vector1d<T>::operator*(const T& A){
    Vector1d<double> product(n);
    for (int i=0; i<n; i++){
        product[i] = v[i] * A;
    }
    return product;
}
template<class T>
Vector1d<T> Vector1d<T>::operator/(const T& A){
    Vector1d<double> quotient(n);
    for (int i=0; i<n; i++){
        quotient[i] = v[i] / A;
    }
    return quotient;
}
template<class T>
Vector1d<T>& Vector1d<T>::operator+=(const Vector1d<T>& A){
    int nA;
    nA = A.size();
    
    if (nA != n){
        cout << "size of vectors don't match. Revise your input." << endl;
        exit(7);
    }
    else{
        for (int i=0; i<n; i++){
            v[i] += A[i];
        }
        return *this;
    }
}
template<class T>
Vector1d<T>& Vector1d<T>::operator+=(const T& a){
    for (int i=0; i<n; i++){
        v[i] += a;
    }
    return *this;
}
template<class T>
Vector1d<T>& Vector1d<T>::operator-=(const Vector1d<T>& A){
    int nA;
    nA = A.size();
    
    if (nA != n){
        cout << "size of vectors don't match. Revise your input." << endl;
        exit(7);
    }
    else{
        for (int i=0; i<n; i++){
            v[i] -= A[i];
        }
        return *this;
    }
}
template<class T>
Vector1d<T>& Vector1d<T>::operator-=(const T& a){
    for (int i=0; i<n; i++){
        v[i] -= a;
    }
    return *this;
}
template<class T>
Vector1d<T>& Vector1d<T>::operator*=(const T& a){
    for (int i=0; i<n; i++){
        v[i] *= a;
    }
    return *this;
}
template<class T>
Vector1d<T>& Vector1d<T>::operator/=(const T& a){
    for (int i=0; i<n; i++){
        v[i] /= a;
    }
    return *this;
}

/*----------------------------------------
 Destructers
 ---------------------------------------*/
template<class T>
Vector1d<T>::~Vector1d<T>(){
    if (v != 0){
        delete[] (v);
    }
}


/*----------------------------------------
 Matrix Types Constructers
 ---------------------------------------*/
template<class T>
Matrix<T>::Matrix(){
    n = 0;    m = 0;
    v = 0;
}
template<class T>
Matrix<T>::Matrix(int nn, int mm){
    n = nn;    m = mm;
    v = new T*[n];
    v[0] = new T[m*n];
    for (int i=1; i<n; i++){
        v[i] = v[i-1] + m;
    }
}
template<class T>
Matrix<T>::Matrix(const T &a, int nn, int mm){
    n = nn;    m = mm;
    v = new T*[n];
    v[0] = new T[m*n];
    for (int i=1; i<n; i++){
        v[i] = v[i-1] + m;
    }
    for (int i=0; i<n; i++){
        for (int j=0; j<m; j++){
            v[i][j] = a;
        }
    }
}
template<class T>
Matrix<T>::Matrix(const T *a, int nn, int mm){
    n = nn;    m = mm;
    v = new T*[n];
    v[0] = new T[m*n];
    for (int i=1; i<n; i++){
        v[i] = v[i-1] + m;
    }
    for (int i=0; i<n; i++){
        for (int j=0; j<m; j++){
            v[i][j] = *a++;
        }
    }
}
template<class T>
Matrix<T>::Matrix(const Matrix &copy){
    n = copy.n; m = copy.m;
    v = new T*[n];
    v[0] = new T[m*n];
    for (int i=1; i<n; i++){
        v[i] = v[i-1] + m;
    }
    for (int i=0; i<n; i++){
        for (int j=0; j<m; j++){
            v[i][j] = copy[i][j];
        }
    }
}

/*----------------------------------------
 Operater
 ---------------------------------------*/
template<class T>
Matrix<T>& Matrix<T>:: operator=(const Matrix<T> &copy){
    if (this != &copy){
        if (n != copy.n || m != copy.m){
            if (v != 0){
                delete v[0];
                delete v;
            }
            n = copy.n;
            m = copy.m;
            v = new T*[n];
            v[0] = new T[n*m];
        }
        for (int i=1; i<n; i++){
            v[i] = v[i-1] + m;
        }
        for (int i=0; i<n; i++){
            for (int j=0; j<m; j++){
                v[i][j] = copy[i][j];
            }
        }
    }
    return *this;
}
template<class T>
Matrix<T>& Matrix<T>:: operator=(const T &r){
    for (int i=0; i<n; i++){
        for (int j=0; j<m; j++){
            v[i][j] = r;
        }
    }
    return *this;
}
template<class T>
void Matrix<T>::resize(int nn, int mm){
    if (n != nn || m != mm){
        if (v != 0){
            delete v[0];
            delete v;
        }
        n = nn;
        m = mm;
        v = new T*[n];
        v[0] = new T[n*m];
    }
    for (int i=1; i<n; i++){
        v[i] = v[i-1] + m;
    }
}
template<class T>
void Matrix<T>::resize(const T& a, int nn, int mm){
    if (n != nn || m != mm){
        if (v != 0){
            delete v[0];
            delete v;
        }
        n = nn;
        m = mm;
        v = new T*[n];
        v[0] = new T[n*m];
    }
    for (int i=1; i<n; i++){
        v[i] = v[i-1] + m;
    }
    for (int i=0; i<n; i++){
        for (int j=0; j<m; j++){
            v[i][j] = a;
        }
    }
}

/*----------------------------------------
 Return row & column vector
 ---------------------------------------*/
template<class T>
Vector1d<T> Matrix<T>::colvector(const int j){
    Vector1d<T> rowv(n);
    for (int i=0; i<n; i++){
	rowv[i] = v[i][j];
    }
    return rowv;
}
template<class T>
Vector1d<T> Matrix<T>::rowvector(const int i){
    Vector1d<T> colv(m);
    for (int j=0; j<m; j++){
	colv[j] = v[i][j];
    }
    return colv;
}

/*----------------------------------------
 Mathematical Operater
 ---------------------------------------*/
template<class T>
Matrix<T> Matrix<T>::transpose(){
    Matrix<T> tran(m,n);  int i,j;
    for (i=0; i<n; i++){
	for (j=0; j<m; j++){
	    tran[j][i] = v[i][j];
	}
    }
    return tran;
}
template<class T>
Matrix<T> Matrix<T>::lu_decomp(){
    if (m != n){
	cout << "unable to calculate the inverse" << endl;
	exit(25);
    }
    Matrix<T> lu(m,m);
    /* LU decomposition */
    for (int i=0; i<m; i++){
	/* calculate l_ij */
	for (int j=i; j<m; j++){
	    lu[j][i] = v[j][i];
	    for (int k=0; k<i; k++){
		lu[j][i] -= lu[k][i]*lu[j][k];
	    } 
	}
	/* calculate u_ij */
	for (int j=i+1; j<m; j++){
	    lu[i][j] = v[i][j];
	    for (int k=0; k<i; k++){
		lu[i][j] -= lu[k][j]*lu[i][k];
	    }
	    lu[i][j] /= lu[i][i];
	}
    }
    return lu;
}
template<class T>
void Matrix<T>::lu_linear(Vector1d<T>& A){
    /* calculate solution */
    for (int i=0; i<n; i++){
	for (int k=0; k<i; k++){ A[i] -= v[i][k]*A[k]; }
	A[i] /= v[i][i];
    }
    for (int i=n-1; i>=0; i--){
	for (int k=i+1; k<n; k++){
	    A[i] -= v[i][k]*A[k];
	}
    }
}
template<class T>
Matrix<T> Matrix<T>::lu_inverse(){
    /* matrix should already been LU decomposed */
    if (m != n){
	cout << "unable to calculate the inverse" << endl;
	exit(25);
    }
    /* prepare identiy matrix */
    Matrix<T> inv(0.0,m,m);
    for (int i=0; i<m; i++){
	inv[i][i] = 1.0;
    }
    /* calculate inverse */
    for (int j=0; j<m; j++){
	for (int i=0; i<n; i++){
	    for (int k=0; k<i; k++){ inv[i][j] -= v[i][k]*inv[k][j]; }
	    inv[i][j] /= v[i][i];
	}
	for (int i=n-1; i>=0; i--){
	    for (int k=i+1; k<n; k++){
		inv[i][j] -= v[i][k]*inv[k][j];
	    }
	}
    }
    return inv;
}
template<class T>
Matrix<T>& Matrix<T>::numeric0(double LIM){
    /* find abs max value in matrix */
    T absmaxv = 0.0;
    for (int i=0; i<n; i++){
	for (int j=0; j<m; j++){
	    if (abs(v[i][j]) > absmaxv) {absmaxv = abs(v[i][j]);}
	}
    }
    /* drop off all numeric error */
    T eps = absmaxv*LIM*16;
    for (int i=0; i<n; i++){
	for (int j=0; j<m; j++){
	    if (abs(v[i][j]) < eps && v[i][j] != 0){ v[i][j] = 0; }
	}
    }
    return *this;
}
template<class T>
Matrix<T>& Matrix<T>::operator+=(const Matrix<T>& B){
    int nB = B.nrows();
    int mB = B.mcols();

    if ((nB != n) || (mB != m)){
	cout << "size of matrixes don't match. Revise your input." << endl;
	exit(7);
    }
    else {
	for (int i=0; i<n; i++){
	    for (int j=0; j<m; j++){
		v[i][j] += B[i][j];
	    }
	}
	return *this;
    }
}
template<class T>
Matrix<T>& Matrix<T>::operator-=(const Matrix<T>& B){
    int nB = B.nrows();
    int mB = B.mcols();

    if ((nB != n) || (mB != m)){
	cout << "size of matrixes don't match. Revise your input." << endl;
	exit(7);
    }
    else {
	for (int i=0; i<n; i++){
	    for (int j=0; j<m; j++){
		v[i][j] -= B[i][j];
	    }
	}
	return *this;
    }
}
template<class T>
Vector1d<T> Matrix<T>::operator*(Vector1d<T> &A){
    int nA;
    nA = A.size();
    // cout << n << m << nB << mB << endl;
    if (nA != m){
        cout << "size of matrix & vector don't match. Revise your input. sizes: " << m << " & " << nA << endl;
        exit(7);
    }
    else{
        Vector1d<T> product(n);
        for (int i=0; i<n; i++){
            product[i] = 0;
            for (int k=0; k<m; k++){
                product[i] += v[i][k]*A[k];
            }
        }
        return product;
    }
}
template<class T>
Matrix<T> Matrix<T>::operator*(Matrix<T> &B){
    int nB, mB;
    nB = B.nrows(); mB = B.mcols();
    // cout << n << m << nB << mB << endl;
    if (nB != m){
        cout << "size of 2 matricies don't match. Revise your matrix." << endl;
        exit(7);
    }
    else{
        Matrix<T> product(n,mB); int i,j,k;
	// int NUM_THREADS=omp_get_num_procs();
	// omp_set_num_threads(NUM_THREADS);
	// #pragma omp parallel for private(j,k)
        for (i=0; i<n; i++){
            for (j=0; j<mB; j++){
                product[i][j] = 0;
                for (k=0; k<m; k++){
                    product[i][j] += v[i][k]*B[k][j];
                }
            }
        }
        return product;
    }
}

/*----------------------------------------
 Destructers
 ---------------------------------------*/
template<class T>
Matrix<T>::~Matrix<T>(){
    if (v!=0){
        if (m!=0){
            delete[] v[0];
        }
        delete[] v;
    }
}
