#ifndef MATRIX_H
#define MATRIX_H

#include <iostream>
#include <vector>
#include <string>
#include <map>

#include <iostream>
#include <vector>
#include <string>

typedef long long ll;

class Rational;

template <unsigned K>
class Finite;

#include <cmath>

namespace extras
{
    ///___________________________twoDegree______________________________///
    template <unsigned Num>
    struct twoDegree {
        static const unsigned value = twoDegree<Num / 2>::value * 2;
    };

    template <>
    struct twoDegree <0> {
        static const unsigned value = 1;
    };

    template <unsigned N, unsigned M, unsigned K>
    struct max {
        static const unsigned value = (N > K ? (N > M ? N : M) : (K > M ? K : M));
    };

    ///___________________________Prime-compile-check____________________///
    template <unsigned Div, unsigned Num>
    struct prime_handler {
        static const bool value = prime_handler<Div - 1, Num>::value && (Num % Div != 0);
    };

    template <unsigned Num>
    struct prime_handler <1, Num> {
        static const bool value = true;
    };

    template <unsigned Num>
    struct prime {
        static const bool value = prime_handler<static_cast<unsigned>(sqrt(Num)), Num>::value;
    };

    ///_______________________Interface__________________________///
    template <unsigned N>
    void crasher();

    template <>
    void crasher<true>() {};

    template <unsigned Number>
    void primeTest() {
        crasher<prime<Number>::value>();
    }

    ///_______________________EQ________________________________///
    template <unsigned A, unsigned B>
    struct equal {
        static const bool value = (A == B);
    };

    ///______________________Is_Same____________________________///
    template <typename U>
    struct finiteCheck {
        template <bool isPrime = true, typename Assistant = void>
        struct helper { static void check() {} };
    };

    template <typename U, U n>
    struct finiteCheck<Finite<n>> {
        template <bool isPrime = extras::prime<n>::value, typename Assistant = void>
        struct helper {
            static void check();
        };

        template <typename Assistant>
        struct helper<true, Assistant> {
            static void check() {}
        };
    };
}

template <unsigned K>
class Finite {
public:
    Finite();
    Finite(const int num);
    Finite(const Finite& right);
    ~Finite();

    Finite& operator=(const Finite& right);

    explicit operator bool() const;

    template <unsigned T>
    friend const Finite<T> operator+(const Finite<T>& left, const Finite<T>& right);
    template <unsigned T>
    friend const Finite<T> operator-(const Finite<T>& left, const Finite<T>& right);
    template <unsigned T>
    friend const Finite<T> operator*(const Finite<T>& left, const Finite<T>& right);
    template <unsigned T>
    friend const Finite<T> operator/(const Finite<T>& left, const Finite<T>& right);

    template <unsigned T>
    friend const Finite<T>& operator+(const Finite<T>& number);
    template <unsigned T>
    friend const Finite<T> operator-(const Finite<T>& number);

    template <unsigned T>
    friend Finite<T>& operator++(Finite<T>& number);
    template <unsigned T>
    friend Finite<T>& operator--(Finite<T>& number);
    template <unsigned T>
    friend Finite<T> operator++(Finite<T>& number, int);
    template <unsigned T>
    friend Finite<T> operator--(Finite<T>& number, int);

    template <unsigned T>
    friend Finite<T>& operator+=(Finite<T>& left, const Finite<T>& right);
    template <unsigned T>
    friend Finite<T>& operator-=(Finite<T>& left, const Finite<T>& right);
    template <unsigned T>
    friend Finite<T>& operator*=(Finite<T>& left, const Finite<T>& right);
    template <unsigned T>
    friend Finite<T>& operator/=(Finite<T>& left, const Finite<T>& right);

    template <unsigned T>
    friend bool operator==(const Finite<T>& left, const Finite<T>& right);
    template <unsigned T>
    friend bool operator!=(const Finite<T>& left, const Finite<T>& right);

    template <unsigned T>
    friend std::istream& operator>> (std::istream &in, Finite<T> &number);
    template <unsigned T>
    friend std::ostream& operator<< (std::ostream &out, const Finite<T> &number);

    const Finite pow(unsigned int deg) const;
    const Finite inv() const;
    unsigned int value() const;

private:
    unsigned int value_;
};

///____________________constructors_destructor_______________///
template <unsigned K>
Finite<K>::Finite() :
    value_(0) {}

template <unsigned K>
Finite<K>::Finite(const int num) {
    int temp = num % static_cast<int>(K);
    value_ = temp + (temp < 0 ? K : 0);
}

template <unsigned K>
Finite<K>::Finite(const Finite<K>& right) :
    value_(right.value_) {}

template <unsigned K>
Finite<K>::~Finite() {};

///____________________Assignment_____________________________///
template <unsigned K>
Finite<K>& Finite<K>::operator=(const Finite<K>& right) {
    if (this == &right) {
        return *this;
    }
    value_ = right.value_;
    return *this;
}

///____________________cast_operators_and_functions_____________///
template <unsigned K>
Finite<K>::operator bool() const {
    return value_ != 0;
}

template <unsigned K>
unsigned int  Finite<K>::value() const {
    return value_;
}
///____________________binary_operators________________________///
template <unsigned T>
const Finite<T> operator+(const Finite<T>& left, const Finite<T>& right) {
    Finite<T> sum = left;
    return sum += right;
}

template <unsigned T>
const Finite<T> operator-(const Finite<T>& left, const Finite<T>& right) {
    Finite<T> dif = left;
    return dif -= right;
}

template <unsigned T>
const Finite<T> operator*(const Finite<T>& left, const Finite<T>& right) {
    Finite<T> multy = left;
    return multy *= right;
}

template <unsigned T>
const Finite<T> operator/(const Finite<T>& left, const Finite<T>& right) {
    Finite<T> div = left;
    return div /= right;
}
///____________________________________________________________///
template <unsigned T>
Finite<T>& operator+=(Finite<T>& left, const Finite<T>& right) {
    left.value_ += right.value_;
    left.value_ %= T;
    return left;
}

template <unsigned T>
Finite<T>& operator-=(Finite<T>& left, const Finite<T>& right) {
    if (left.value_ < right.value_) {
        left.value_ += T;
    }
    left.value_ -= right.value_;
    return left;
}

template <unsigned T>
Finite<T>& operator*=(Finite<T>& left, const Finite<T>& right) {
    unsigned long long temp = left.value_;
    temp *= right.value_;
    temp %= T;
    return left = static_cast<unsigned int>(temp);
}

template <unsigned T>
Finite<T>& operator/=(Finite<T>& left, const Finite<T>& right) {
    return left *= (right.inv());
}

///_________________________________________________________///
template <unsigned T>
bool operator==(const Finite<T>& left, const Finite<T>& right) {
    return left.value_ == right.value_;
}

template <unsigned T>
bool operator!=(const Finite<T>& left, const Finite<T>& right) {
    return !(left == right);
}


///_________________________Unary_operators_________________///
template <unsigned T>
const Finite<T>& operator+(const Finite<T>& number)
{
    return number;
}

template <unsigned T>
const Finite<T> operator-(const Finite<T>& number)
{
    Finite<T> oppos = number;
    if (oppos){
        oppos.value_ = T - oppos.value_;
    }
    return oppos;
}

template <unsigned T>
Finite<T>& operator++(Finite<T>& number)
{
    return number += Finite<T>(1);
}

template <unsigned T>
Finite<T>& operator--(Finite<T>& number)
{
    return number -= Finite<T>(1);
}

template <unsigned T>
Finite<T> operator++(Finite<T>& number, int)
{
    Finite<T> old_number = number;
    ++number;
    return old_number;
}

template <unsigned T>
Finite<T> operator--(Finite<T>& number, int)
{
    Finite<T> old_number = number;
    --number;
    return old_number;
}

///_________________________IO______________________________///
template <unsigned T>
std::istream& operator>> (std::istream& in, Finite<T>& number) {
    int inInt;
    in >> inInt;
    inInt %= static_cast<int>(T);
    number.value_ = inInt + (inInt < 0 ? T : 0);
    return in;
}

template <unsigned T>
std::ostream& operator<< (std::ostream& out, const Finite<T>& number) {
    out << number.value_;
    return out;
}

///________________________Functions_________________________///
template <unsigned K>
const Finite<K> Finite<K>::pow(unsigned int deg) const {
    if (deg == 0) {
        return 1;
    }
    if (deg % 2 == 0) {
        Finite<K> a = (this->pow(deg / 2));
        return a * a;
    } else {
        return (*this) * (this->pow(deg - 1));
    }
}


template <unsigned K>
const Finite<K> Finite<K>::inv() const {
    extras::primeTest<K>();
    return this->pow(K - 2);
}

#include <vector>

template<unsigned N, unsigned M, typename Field=Rational>
class Matrix
{
public:
    Matrix();
    Matrix(const Matrix& right);
    Matrix(bool rand);
    Matrix(const std::initializer_list<std::initializer_list<Field>>&);
    ~Matrix();

    Matrix& operator=(const Matrix& right);
    Field det() const;
    unsigned rank() const;

    Matrix<M, N, Field> transposed() const;

    void invert();
    Matrix inverted() const;

    Field trace() const;

    std::vector<Field> getRow(const unsigned n) const;
    std::vector<Field> getColumn(const unsigned n) const;

    template <unsigned N_, unsigned M_, typename Field_>
    friend const Matrix<N_, M_, Field_> operator+(const Matrix<N_, M_, Field_>& left, const Matrix<N_, M_, Field_>& right);
    template <unsigned N_, unsigned M_, typename Field_>
    friend const Matrix<N_, M_, Field_> operator-(const Matrix<N_, M_, Field_>& left, const Matrix<N_, M_, Field_>& right);
    template <unsigned N_, unsigned M_, unsigned K_, typename Field_>
    friend const Matrix<N_, K_, Field_> operator*(const Matrix<N_, M_, Field_>& left, const Matrix<M_, K_, Field_>& right);///!!!!!!!!!!!!

    template <unsigned N_, unsigned M_, typename Field_>
    friend const Matrix<N_, M_, Field_> operator*(const Matrix<N_, M_, Field_>& left, const Field_& right);
    template <unsigned N_, unsigned M_, typename Field_>
    friend const Matrix<N_, M_, Field_> operator*(const Field_& left, const Matrix<N_, M_, Field_>& right);

    template <unsigned N_, unsigned M_, typename Field_>
    friend Matrix<N_, M_, Field_>& operator*=(Matrix<N_, M_, Field_>& left, const Field_& right);

    template <unsigned N_, unsigned M_, typename Field_>
    friend Matrix<N_, M_, Field_>& operator+=(Matrix<N_, M_, Field_>& left, const Matrix<N_, M_, Field_>& right);
    template <unsigned N_, unsigned M_, typename Field_>
    friend Matrix<N_, M_, Field_>& operator-=(Matrix<N_, M_, Field_>& left, const Matrix<N_, M_, Field_>& right);
    template <unsigned N_, typename Field_>
    friend Matrix<N_, N_, Field_>& operator*=(Matrix<N_, N_, Field_>& left, const Matrix<N_, N_, Field_>& right);

    template <unsigned N_, unsigned M_, typename Field_>
    friend std::istream& operator>> (std::istream &in, Matrix<N_, M_, Field_> &number);
    template <unsigned N_, unsigned M_, typename Field_>
    friend std::ostream& operator<< (std::ostream &out, const Matrix<N_, M_, Field_> &number);

    template <unsigned N_, unsigned M_, typename Field_>
    friend bool operator==(const Matrix<N_, M_, Field_>& left, const Matrix<N_, M_, Field_>& right);
    template <unsigned N_, unsigned M_, typename Field_>
    friend bool operator!=(const Matrix<N_, M_, Field_>& left, const Matrix<N_, M_, Field_>& right);

    const std::vector<Field>& operator[](const size_t ind) const;
    std::vector<Field>& operator[](const size_t ind);
private:
    std::vector<std::vector <Field>> data_;

    static void checkFinite_() { extras::finiteCheck<Field>::template helper<>::check(); }
    static void checkSize_() { extras::crasher<extras::equal<N, M>::value>(); }
};

template<unsigned N, typename Field=Rational>
using SquareMatrix = Matrix<N, N, Field>;


///______________________Constructor_Destructor_____________________///
template <unsigned N, unsigned M, typename Field>
Matrix<N, M, Field>::Matrix() {
    checkFinite_();
    data_.resize(N, std::vector<Field>(M));
}

template <unsigned N, unsigned M, typename Field>
Matrix<N, M, Field>::Matrix(bool rand) {
    checkFinite_();
    rand = true;
    rand %= 2;
    data_.resize(N, std::vector<Field>(M));
    for (size_t i = 0; i < N; ++i) {
        for (size_t c = 0; c < M; ++c) {
            data_[i][c] = std::rand();
        }
    }
}

template <unsigned N, unsigned M, typename Field>
Matrix<N, M, Field>::Matrix(const Matrix<N, M, Field>& right) :
    data_(right.data_) {}

template <unsigned N, unsigned M, typename Field>
Matrix<N, M, Field>::Matrix(const std::initializer_list<std::initializer_list<Field>>& init) : Matrix() {
    checkFinite_();
    size_t i = 0,
           j = 0;
    for (auto it = init.begin(); it != init.end(); it++, i++) {
        j = 0;
        for (auto jt = it->begin(); jt != it->end(); jt++, j++) {
            data_[i][j] = *jt;
        }
    }
}

template<unsigned N, unsigned M, typename Field>
Matrix<N, M, Field>::~Matrix() {}

///____________________Assignment_____________________________///
template <unsigned N, unsigned M, typename Field>
Matrix<N, M, Field>& Matrix<N, M, Field>::operator=(const Matrix<N, M, Field>& right) {
    if (this == &right) {
        return *this;
    }
    data_ = right.data_;
    return *this;
}

template <unsigned N, unsigned M, typename Field>
const std::vector<Field>& Matrix<N, M, Field>::operator[](const size_t ind) const {
    return data_[ind];
}

template <unsigned N, unsigned M, typename Field>
std::vector<Field>& Matrix<N, M, Field>::operator[](const size_t ind) {
    return data_[ind];
}
///_______________________IO________________________________________///
template <unsigned N_, unsigned M_, typename Field_>
std::istream& operator>> (std::istream &in, Matrix<N_, M_, Field_> &matrix) {
    for (size_t i = 0; i < N_; ++i) {
        for (size_t c = 0; c < M_; ++c) {
            in >> matrix.data_[i][c];
        }
    }
    return in;
}

template <unsigned N_, unsigned M_, typename Field_>
std::ostream& operator<< (std::ostream &out, const Matrix<N_, M_, Field_> &matrix) {
    for (size_t i = 0; i < N_; ++i) {
        for (size_t c = 0; c < M_; ++c) {
            out << matrix.data_[i][c] << ' ';
        }
        out << std::endl;
    }
    return out;
}

///_______________________Binary________________________________________///
template <unsigned N_, unsigned M_, typename Field_>
const Matrix<N_, M_, Field_> operator+(const Matrix<N_, M_, Field_>& left, const Matrix<N_, M_, Field_>& right) {
    Matrix<N_, M_, Field_> sum = left;
    return sum += right;
}

template <unsigned N_, unsigned M_, typename Field_>
const Matrix<N_, M_, Field_> operator-(const Matrix<N_, M_, Field_>& left, const Matrix<N_, M_, Field_>& right) {
    Matrix<N_, M_, Field_> dif = left;
    return dif -= right;
}

template <unsigned N_, unsigned M_, unsigned K_, typename Field_>
const Matrix<N_, K_, Field_> operator*(const Matrix<N_, M_, Field_>& left, const Matrix<M_, K_, Field_>& right) {
    Matrix <N_, K_, Field_> multy;
    for (size_t i = 0; i < N_; ++i) {
        for (size_t c = 0; c < K_; ++c) {
            for (size_t k = 0; k < M_; ++k) {
                multy[i][c] += left[i][k] * right[k][c];
            }
        }
    }
    return multy;
}

template <unsigned N_, unsigned M_, typename Field_>
const Matrix<N_, M_, Field_> operator*(const Matrix<N_, M_, Field_>& left, const Field_& right) {
    Matrix<N_, M_, Field_> multy = left;
    return multy *= right;
}

template <unsigned N_, unsigned M_, typename Field_>
const Matrix<N_, M_, Field_> operator*(const Field_& left, const Matrix<N_, M_, Field_>& right) {
    Matrix<N_, M_, Field_> multy = right;
    return multy *= left;
}

///_____________________________________________________________________///
template <unsigned N_, unsigned M_, typename Field_>
Matrix<N_, M_, Field_>& operator+=(Matrix<N_, M_, Field_>& left, const Matrix<N_, M_, Field_>& right) {
    for (size_t i = 0; i < N_; ++i) {
        for (size_t c = 0; c < M_; ++c) {
            left[i][c] += right[i][c];
        }
    }
    return left;
}

template <unsigned N_, unsigned M_, typename Field_>
Matrix<N_, M_, Field_>& operator-=(Matrix<N_, M_, Field_>& left, const Matrix<N_, M_, Field_>& right) {
    for (size_t i = 0; i < N_; ++i) {
        for (size_t c = 0; c < M_; ++c) {
            left[i][c] -= right[i][c];
        }
    }
    return left;
}

template <unsigned N_, typename Field_>
Matrix<N_, N_, Field_>& operator*=(Matrix<N_, N_, Field_>& left, const Matrix<N_, N_, Field_>& right)
{
    left = left * right;
    return left;
}

template <unsigned N_, unsigned M_, typename Field_>
Matrix<N_, M_, Field_>& operator*=(Matrix<N_, M_, Field_>& left, const Field_& right) {
    for (size_t i = 0; i < N_; ++i) {
        for (size_t c = 0; c < M_; ++c) {
            left[i][c] *= right;
        }
    }
    return left;
}

///_______________________Methods_______________________________________///
template<unsigned N, unsigned M, typename Field>
std::vector<Field> Matrix<N, M, Field>::getRow(const unsigned n) const {
    return data_[n];
}

template<unsigned N, unsigned M, typename Field>
std::vector<Field> Matrix<N, M, Field>::getColumn(const unsigned n) const {
    std::vector<Field> col(N);
    for (size_t i = 0; i < N; ++i) {
        col[i] = data_[i][n];
    }
    return col;
}

template<unsigned N, unsigned M, typename Field>
Field Matrix<N, M, Field>::trace() const {
    checkSize_();
    Field trace = 0;
    for (size_t i = 0; i < N; ++i) {
        trace += data_[i][i];
    }
    return trace;
}

template<unsigned N, unsigned M, typename Field>
Matrix<M, N, Field> Matrix<N, M, Field>::transposed() const {
    Matrix<M, N, Field> transposed;
    for (size_t i = 0; i < N; ++i) {
        for (size_t c = 0; c < M; ++c) {
            transposed[c][i] = data_[i][c];
        }
    }
    return transposed;
}

template<unsigned N, unsigned M, typename Field>
unsigned Matrix<N, M, Field>::rank() const {
    bool dummyB;
    Matrix<N, M, Field> dummy;
    Matrix<N, M, Field> diagonal = methodGaus(*this, dummy, dummyB);
    unsigned rank = 0;
    std::vector<Field> zeroes(M);
    for (size_t i = 0; i < N; ++i) {
        if (diagonal[i] != zeroes) {
            rank++;
        }
    }
    return rank;
}
template<unsigned N, unsigned M, typename Field>
Field Matrix<N, M, Field>::det() const {
    checkSize_();
    bool sign_ = false;
    Matrix<N, M, Field> dummy;
    Matrix<N, M, Field> triangle = methodGaus(*this, dummy, sign_);
    Field det = 1;
    for (size_t i = 0; i < N; ++i) {
        det *= triangle[i][i];
    }
    det *= sign_ ? Field(-1) : Field(1);
    return det;
}

template<unsigned N, unsigned M, typename Field>
void Matrix<N, M, Field>::invert() {
    *this = this->inverted();
}

template<unsigned N, unsigned M, typename Field>
Matrix<N, M, Field> Matrix<N, M, Field>::inverted() const {
    checkSize_();
    if (this->det() == Field(0)){
        throw std::bad_cast();
    }
    bool sign_ = false;
    SquareMatrix<N, Field> inverted;
    for (size_t i = 0; i < N; ++i){
        inverted[i][i] = 1;
    }
    Matrix<N, M, Field> firstStep = methodGaus(*this, inverted, sign_);
    for (size_t i = 0; i < N; ++i){
        Field coef = firstStep[i][i];
        divElem(inverted[i], coef);
    }
    return inverted;
}

template <unsigned N_, unsigned M_, typename Field_>
bool operator==(const Matrix<N_, M_, Field_>& left, const Matrix<N_, M_, Field_>& right) {
    return left.data_ == right.data_;
}

template <unsigned N_, unsigned M_, typename Field_>
bool operator!=(const Matrix<N_, M_, Field_>& left, const Matrix<N_, M_, Field_>& right) {
    return !(left == right);
}

///_________________________________________________///
template<typename Field>
void sumElem(std::vector<Field>& v1, const std::vector<Field> v2, const Field& coef = 1) {
    if (v1.size() != v2.size()) {
        throw std::bad_cast();
    }
    for (size_t i = 0; i < v1.size(); ++i) {
        v1[i] += v2[i] * coef;
    }
}

template<typename Field>
void divElem(std::vector<Field>& v, const Field& coef) {
    for (size_t i = 0; i < v.size(); ++i) {
        v[i] /= coef;
    }
}

template<unsigned N, unsigned M, typename Field>
Matrix<N, M, Field> methodGaus(const Matrix<N, M, Field>& matrix, Matrix<N, M, Field>& inverted, bool& sign_) {
    Matrix<N, M, Field> diagonal = matrix;
    for (size_t i = 0; i < std::min(N, M); ++i) {
        bool zero = true;
        for (size_t c = i; c < N; ++c) {
            if (diagonal[c][i] != Field(0)) {
                std::swap(diagonal[i], diagonal[c]);
                std::swap(inverted[i], inverted[c]);
                if (i != c) sign_ ^= true;
                zero = false;
                break;
            }
        }
        if (zero) {
            continue;
        }
        for (size_t c = 0; c < N; ++c) {
            if (c == i) {
                continue;
            }
            Field coef = diagonal[c][i] / diagonal[i][i] * Field(-1);
            sumElem(diagonal[c], diagonal[i], coef);
            sumElem(inverted[c], inverted[i], coef);
        }
    }
    return diagonal;
}

template <unsigned N, unsigned M, unsigned K, unsigned T, typename Field>
void copyMatrixValue(Matrix<N, M, Field>& direction, const Matrix<K, T, Field>& source, size_t X = 0, size_t Y = 0) {
    int a = int(N) - int(K),
        b = int(M) - int(T);
    if (a * b < 0) {
        throw std::bad_cast();
    }
    bool flag = (N > K || M > T);
    for (size_t i = 0; i < std::min(size_t(std::min(K, N)), std::max(K, N) - X); ++i) {
        for (size_t c = 0; c < std::min(size_t(std::min(M, T)), std::max(M, T) - Y); ++c) {
            if (flag) {
                direction[i + X][c + Y] = source[i][c];
            } else {
                direction[i][c] = source[i + X][c + Y];
            }
        }
    }
}

template <unsigned N_, unsigned M_, unsigned K_, typename Field_>
Matrix<N_, K_, Field_> strassen(const Matrix<N_, M_, Field_>& left, const Matrix<M_, K_, Field_>& right) {
    static const unsigned twoDegree = extras::twoDegree<extras::max<N_ - 1, M_ - 1, K_ - 1>::value>::value;
    SquareMatrix<twoDegree, Field_> A, B, C;

    copyMatrixValue(A, left);
    copyMatrixValue(B, right);
    C = strassen_(A, B);

    Matrix<N_, K_, Field_> answer;
    copyMatrixValue(answer, C);

    return answer;
}

template <unsigned N, typename Field>
void copyMatrixStrassen_(std::vector<std::vector<SquareMatrix<N / 2, Field>>> dests, const SquareMatrix<N, Field>& A) {
    for (char i = 0; i <= 1; ++i) {
        for (char j = 0; j <= 1; ++j) {
            copyMatrixValue(dests[i][j], A, i * N / 2, j * N / 2);
        }
    }
}

template <unsigned N, typename Field>
const SquareMatrix<N, Field> strassen_(const SquareMatrix<N, Field>& A, const SquareMatrix<N, Field>& B) {
    if (N <= 64) {
        return A * B;
    }
    std::vector<std::vector<SquareMatrix<N / 2, Field>>> a(2, std::vector<SquareMatrix<N / 2, Field>>(2)),
                                                         b(2, std::vector<SquareMatrix<N / 2, Field>>(2)),
                                                         c(2, std::vector<SquareMatrix<N / 2, Field>>(2));
    std::vector<SquareMatrix<N / 2, Field>> p(7);

    copyMatrixStrassen_(a, A);
    copyMatrixStrassen_(b, B);

    p[0] = strassen_(a[0][0] + a[1][1], b[0][0] + b[1][1]);
    p[1] = strassen_(a[1][0] + a[1][1], b[0][0]);
    p[2] = strassen_(a[0][0], b[0][1] - b[1][1]);
    p[3] = strassen_(a[1][1], b[1][0] - b[0][0]);
    p[4] = strassen_(a[0][0] + a[0][1], b[1][1]);
    p[5] = strassen_(a[1][0] - a[0][0], b[0][0] + b[0][1]);
    p[6] = strassen_(a[0][1] - a[1][1], b[1][0] + b[1][1]);

    c[0][0] = p[0] + p[3] - p[4] + p[6];
    c[0][1] = p[2] + p[4];
    c[1][0] = p[1] + p[3];
    c[1][1] = p[0] - p[1] + p[2] + p[5];

    SquareMatrix<N, Field> C;
    copyMatrixStrassen_(c, C);
    return C;
}

#include <iostream>
#include <vector>
#include <string>

enum SIGN {
    MINUS = -1,
    PLUS = 1
};

void AddZeros(std::string &str, size_t number_of_zeros) {
    str += std::string(number_of_zeros, '0');
}

class BigInteger;

bool operator<(const BigInteger &lhs, const BigInteger &rhs);

bool operator==(const BigInteger &lhs, const BigInteger &rhs);

bool operator>(const BigInteger &lhs, const BigInteger &rhs);

bool operator<=(const BigInteger &lhs, const BigInteger &rhs);

bool operator>=(const BigInteger &lhs, const BigInteger &rhs);

bool operator!=(const BigInteger &lhs, const BigInteger &rhs);

const BigInteger operator+(const BigInteger &lhs, const BigInteger &rhs);

const BigInteger operator-(const BigInteger &lhs, const BigInteger &rhs);

const BigInteger operator*(const BigInteger &lhs, const BigInteger &rhs);

const BigInteger operator/(const BigInteger &lhs, const BigInteger &rhs);

const BigInteger operator%(const BigInteger &lhs, const BigInteger &rhs);

BigInteger operator ""_bi(const char *str);

class BigInteger {
private:
    static const int NORMAL_MULTIPLY_LIMIT = 32;
    static const int BASE = 1000000000;
    static const int BASE_DIGITS = 9;

    std::vector<int> _digits;
    SIGN sign = PLUS;

public:

    BigInteger() = default;

    BigInteger(const BigInteger &big_int) = default;

    BigInteger(BigInteger &&big_int) = default;

    BigInteger &operator=(BigInteger &&big_int) = default;

    BigInteger &operator=(const BigInteger &big_int);

    ~BigInteger() = default;

    BigInteger(long long number);

    explicit BigInteger(const std::string &str);

    BigInteger &operator=(long long number);

private:
    void ReadFromString(const std::string &str);

public:

    friend std::istream &operator>>(std::istream &in, BigInteger &big_int);

    friend std::ostream &operator<<(std::ostream &out, const BigInteger &big_int);

    std::string toString() const;

public:
    static int CompareAbs(const BigInteger &lhs, const BigInteger &rhs);

private:
    const BigInteger UnsignedAdd(const BigInteger &big_int);

    const BigInteger UnsignedSub(const BigInteger &big_int);

public:
    const BigInteger operator-() const;

    BigInteger &operator+=(const BigInteger &big_int);

    BigInteger &operator-=(const BigInteger &big_int);

    BigInteger &operator++();

    BigInteger &operator--();

    const BigInteger operator++(int);

    const BigInteger operator--(int);

private:
    std::pair<const BigInteger, const BigInteger> DivWithMod(const BigInteger &big_int) const;

    const BigInteger MultiplyN_2(const BigInteger &big_int) const;

    static std::vector<long long>
    MultiplyKaratsuba(const std::vector<long long> &lhs, const std::vector<long long> &rhs);

    const BigInteger Multiply_N_1_58(const BigInteger &big_int) const;

    static std::vector<int> ChangeBase(const std::vector<int> &_old, int old_base_digits, int new_base_digits);


public:
    BigInteger &operator/=(const BigInteger &big_int);

    BigInteger &operator%=(const BigInteger &big_int);

    BigInteger &operator/=(long long number);

    BigInteger &operator%=(long long number);

    const BigInteger operator/(long long number) const;

    const BigInteger operator%(long long number) const;

    BigInteger &operator*=(long long number);

    const BigInteger operator*(long long number) const;

    BigInteger &operator*=(const BigInteger &big_int);

    static void MultiplyByBase(BigInteger &big_int);

public:
    explicit operator bool() const;

    explicit operator long long() const;

public:
    static BigInteger GCD(BigInteger lhs, BigInteger rhs);

    size_t Length() const;

    SIGN GetSign() const;

    static int GetBase();

    static int GetBaseDigits();

    const BigInteger Abs() const;

private:
    bool IsZero() const;

    static size_t NumberOfZeros(int digit);

    static void FillWithZeros(std::vector<long long> &digits, size_t size_to_fill);

    static int CheckNearestTwoDegree(int number);

    void Shrink();
};

class Rational;

bool operator<(const Rational &lhs, const Rational &rhs);

bool operator>(const Rational &lhs, const Rational &rhs);

bool operator<=(const Rational &lhs, const Rational &rhs);

bool operator>=(const Rational &lhs, const Rational &rhs);

bool operator==(const Rational &lhs, const Rational &rhs);

bool operator!=(const Rational &lhs, const Rational &rhs);


const Rational operator+(const Rational &lhs, const Rational &rhs);

const Rational operator-(const Rational &lhs, const Rational &rhs);

const Rational operator*(const Rational &lhs, const Rational &rhs);

const Rational operator/(const Rational &lhs, const Rational &rhs);

class Rational {
private:
    BigInteger _numerator{0};
    BigInteger _denominator{1};
public:
    Rational() = default;

    Rational(const Rational &rational) = default;

    Rational &operator=(const Rational &rational);

    Rational(Rational &&rational) = default;

    Rational &operator=(Rational &&rational) = default;

    Rational(BigInteger numerator, BigInteger denominator = 1);

    Rational(long long numerator, long long denominator = 1);

    ~Rational() = default;

public:
    static int CompareAbs(const Rational &lhs, const Rational &rhs);

public:

    std::string toString() const;

    friend std::istream &operator>>(std::istream &in, Rational &rational);

    friend std::ostream &operator<<(std::ostream &out, const Rational &rational);

public:

    const Rational operator-() const;

    Rational &operator+=(const Rational &rational);

    Rational &operator-=(const Rational &rational);

    Rational &operator*=(const Rational &rational);

    Rational &operator/=(const Rational &rational);
public:

    std::string asDecimal(size_t precision = 0) const;

    explicit operator double() const;
private:
    void Normalize();

    static void RemoveTrailingZeros(std::string &str, size_t &curr_precision);

public:
    SIGN GetSign() const;
};

BigInteger &BigInteger::operator=(const BigInteger &big_int) {
    if (this != &big_int) {
        _digits = big_int._digits;
        sign = big_int.sign;
    }
    return *this;
}

BigInteger::BigInteger(long long number) {
    *this = number;
    sign = number >= 0 ? PLUS : MINUS;
}


BigInteger::BigInteger(const std::string &str) {
    ReadFromString(str);
}

BigInteger &BigInteger::operator=(long long number) {
    sign = number < 0 ? MINUS : PLUS;
    if (sign == MINUS) {
        number = -number;
    }
    _digits.clear();
    while (number > 0) {
        _digits.push_back(number % BASE);
        number /= BASE;
    }
    return *this;
}

void BigInteger::ReadFromString(const std::string &str) {
    sign = PLUS;
    _digits.clear();
    if (str.empty()) {
        return;
    }
    int sign_pos = 0;
    if (str[sign_pos] == '+' || str[sign_pos] == '-') {
        sign = (str[sign_pos] == '-') ? MINUS : PLUS;
        ++sign_pos;
    }

    for (int cur_pos = static_cast<int>(str.size() - 1); cur_pos >= sign_pos; cur_pos -= BASE_DIGITS) {
        int curr_digit{0};
        for (int i = std::max(sign_pos, cur_pos - BASE_DIGITS + 1); i <= cur_pos; ++i) {
            curr_digit *= 10;
            if (str[i] >= '0' && str[i] <= '9') {
                curr_digit += (str[i] - '0');
            } else {
                *this = 0;
            }
        }
        _digits.push_back(curr_digit);
    }
    Shrink();
}

std::istream &operator>>(std::istream &in, BigInteger &big_int) {
    std::string str;
    in >> str;
    big_int.ReadFromString(str);
    return in;
}

std::ostream &operator<<(std::ostream &out, const BigInteger &big_int) {
    return out << big_int.toString();
}

std::string BigInteger::toString() const {
    std::string result{};
    result.reserve(_digits.size() + 1);
    if (sign == MINUS && !IsZero()) {
        result += '-';
    }
    result += (_digits.empty() ? "0" : std::to_string(_digits.back()));
    for (int cur_pos = static_cast<int>(_digits.size() - 2); cur_pos >= 0; --cur_pos) {
        AddZeros(result, NumberOfZeros(_digits[cur_pos]));
        result += std::to_string(_digits[cur_pos]);
    }
    return result;
}

BigInteger operator "" _bi(const char *str) {
    BigInteger result((std::string(str)));
    return result;
}

////////////////// * READING AND WRITING END * //////////////////


////////////////// * COMPARE BEGIN * //////////////////

int BigInteger::CompareAbs(const BigInteger &lhs, const BigInteger &rhs) {
    if (lhs._digits.size() != rhs._digits.size()) {
        return lhs._digits.size() < rhs._digits.size() ? -1 : 1;
    }
    for (int i = static_cast<int>(lhs._digits.size() - 1); i >= 0; --i) {
        if (lhs._digits[i] != rhs._digits[i]) {
            return lhs._digits[i] < rhs._digits[i] ? -1 : 1;
        }
    }
    return 0;
}

bool operator<(const BigInteger &lhs, const BigInteger &rhs) {
    if (lhs.GetSign() != rhs.GetSign()) {
        return lhs.GetSign() < rhs.GetSign();
    }
    return BigInteger::CompareAbs(lhs, rhs) * lhs.GetSign() == -1;
}

bool operator==(const BigInteger &lhs, const BigInteger &rhs) {
    return lhs.GetSign() == rhs.GetSign() && BigInteger::CompareAbs(lhs, rhs) == 0;
}

bool operator>(const BigInteger &lhs, const BigInteger &rhs) {
    return rhs < lhs;
}

bool operator<=(const BigInteger &lhs, const BigInteger &rhs) {
    return !(rhs < lhs);
}

bool operator>=(const BigInteger &lhs, const BigInteger &rhs) {
    return !(lhs < rhs);
}

bool operator!=(const BigInteger &lhs, const BigInteger &rhs) {
    return !(lhs == rhs);
}

const BigInteger BigInteger::UnsignedAdd(const BigInteger &big_int) {
    if (_digits.size() < big_int._digits.size()) {
        _digits.resize(big_int._digits.size(), 0);
    }
    for (int pos = 0, to_add = 0; pos < static_cast<int>(_digits.size()) || to_add != 0; ++pos) {
        if (pos == static_cast<int>(_digits.size())) {
            _digits.push_back(0);
        }
        _digits[pos] += to_add + (pos < static_cast<int>(big_int._digits.size()) ? big_int._digits[pos] : 0);
        to_add = _digits[pos] >= BASE ? 1 : 0;
        if (to_add == 1) {
            _digits[pos] -= BASE;
        }
    }
    return *this;
}

const BigInteger BigInteger::UnsignedSub(const BigInteger &big_int) {
    for (int pos = 0, to_sub = 0; pos < static_cast<int>(big_int._digits.size()) || to_sub != 0; ++pos) {
        _digits[pos] -= to_sub + (pos < static_cast<int>(big_int._digits.size()) ? big_int._digits[pos] : 0);
        to_sub = _digits[pos] < 0 ? 1 : 0;
        if (to_sub == 1) {
            _digits[pos] += BASE;
        }
    }
    Shrink();
    return *this;
}

const BigInteger BigInteger::operator-() const {
    if (IsZero()) {
        return *this;
    }
    BigInteger res = *this;
    res.sign = static_cast<SIGN>(-1 * sign);
    return res;
}

BigInteger &BigInteger::operator+=(const BigInteger &big_int) {
    if (sign == big_int.sign) {
        UnsignedAdd(big_int);
    } else {
        if (CompareAbs(*this, big_int) >= 0) {
            UnsignedSub(big_int);
        } else {
            BigInteger other = big_int;
            *this = other.UnsignedSub(*this);
        }
    }
    return *this;
}

BigInteger &BigInteger::operator-=(const BigInteger &big_int) {
    if (sign == big_int.sign) {
        if (CompareAbs(*this, big_int) >= 0) {
            UnsignedSub(big_int);
        } else {
            BigInteger other = big_int;
            *this = other.UnsignedSub(*this);
            sign = static_cast<SIGN>(-1 * sign);
        }
    } else {
        /* a - (-b) = a + b */
        UnsignedAdd(big_int);
    }
    return *this;
}

const BigInteger operator+(const BigInteger &lhs, const BigInteger &rhs) {
    BigInteger lhs_copy = lhs;
    return lhs_copy += rhs;
}

const BigInteger operator-(const BigInteger &lhs, const BigInteger &rhs) {
    BigInteger lhs_copy = lhs;
    return lhs_copy -= rhs;
}

BigInteger &BigInteger::operator++() {
    return *this += 1;
}

BigInteger &BigInteger::operator--() {
    return *this -= 1;
}

const BigInteger BigInteger::operator++(int) {
    BigInteger old(*this);
    ++(*this);
    return old;
}

const BigInteger BigInteger::operator--(int) {
    BigInteger old(*this);
    --(*this);
    return old;
}


std::pair<const BigInteger, const BigInteger> BigInteger::DivWithMod(const BigInteger &big_int) const {
    int norm = BigInteger::BASE / (big_int._digits.back() + 1);
    BigInteger _new_lhs = Abs() * norm;
    BigInteger _new_rhs = big_int.Abs() * norm;
    BigInteger q = 0, r = 0;
    q._digits.resize(_new_lhs._digits.size());
    for (int i = static_cast<int>(_new_lhs._digits.size() - 1); i >= 0; --i) {
        r *= BigInteger::BASE;
        r += _new_lhs._digits[i];
        long long from_r_1 = r._digits.size() <= _new_rhs._digits.size() ? 0 : r._digits[_new_rhs._digits.size()];
        long long from_r_2 =
                static_cast<int>(r._digits.size()) <= static_cast<int>(_new_rhs._digits.size() - 1) ? 0 : r._digits[
                        _new_rhs._digits.size() - 1];
        long long digit = (static_cast<long long>(BigInteger::BASE) * from_r_1 + from_r_2) / _new_rhs._digits.back();
        r -= _new_rhs * digit;
        while (r < 0) {
            r += _new_rhs;
            --digit;
        }
        q._digits[i] = digit;
    }
    q.sign = (sign == big_int.sign) ? PLUS : MINUS;
    r.sign = sign;
    q.Shrink();
    r.Shrink();
    return std::make_pair(q, r /= norm);
}

void BigInteger::MultiplyByBase(BigInteger &big_int) {
    big_int._digits.emplace(big_int._digits.begin(), 0);
}

const BigInteger BigInteger::MultiplyN_2(const BigInteger &big_int) const {
    BigInteger result{};
    result.sign = (sign == big_int.sign) ? PLUS : MINUS;
    result._digits.resize(_digits.size() + big_int._digits.size(), 0);
    for (int this_pos = 0; this_pos < static_cast<int>(_digits.size()); ++this_pos) {
        if (_digits[this_pos] != 0) {
            for (int other_pos = 0, to_add = 0;
                 other_pos < static_cast<int>(big_int._digits.size()) || to_add != 0; ++other_pos) {
                long long current =
                        result._digits[this_pos + other_pos] + static_cast<long long>(_digits[this_pos]) *
                                                               (other_pos < static_cast<int>(big_int._digits.size())
                                                                ? big_int._digits[other_pos] : 0) + to_add;
                to_add = static_cast<int>(current / BASE);
                result._digits[this_pos + other_pos] = static_cast<int>(current % BASE);
            }
        }
    }
    result.Shrink();
    return result;
}

std::vector<long long>
BigInteger::MultiplyKaratsuba(const std::vector<long long> &lhs, const std::vector<long long> &rhs) {
    int size = lhs.size();
    std::vector<long long> result(2 * size);
    //Числа меньше умножаем наивным способом, можем не переносить
    if (size <= 32) {
        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < size; ++j) {
                result[i + j] += lhs[i] * rhs[j];
            }
        }
        return result;
    }
    int half_size = size / 2;
    std::vector<long long> l1(lhs.begin(), lhs.begin() + half_size);
    std::vector<long long> l2(lhs.begin() + half_size, lhs.end());
    std::vector<long long> r1(rhs.begin(), rhs.begin() + half_size);
    std::vector<long long> r2(rhs.begin() + half_size, rhs.end());

    std::vector<long long> l1_r1 = MultiplyKaratsuba(l1, r1);
    std::vector<long long> l2_r2 = MultiplyKaratsuba(l2, r2);

    for (int i = 0; i < half_size; ++i) {
        l2[i] += l1[i];
        r2[i] += r1[i];
    }
    std::vector<long long> temp_res = MultiplyKaratsuba(l2, r2);
    for (int pos = 0; pos < static_cast<int>(l1_r1.size()); ++pos) {
        temp_res[pos] -= l1_r1[pos] + l2_r2[pos];
    }
    for (int pos = 0; pos < size; ++pos) {
        result[pos] += l1_r1[pos];
        result[pos + half_size] += temp_res[pos];
        result[pos + size] += l2_r2[pos];
    }
    return result;
}

/* алгоритм карацубы с переводом в другую систему счисления */
const BigInteger BigInteger::Multiply_N_1_58(const BigInteger &big_int) const {
    /* необходимость перейти в миллионную систему связана с алгоритмом Карацубы,
     * который не гарантирует переполнения цифр */
    const int new_base = 1000000;
    const int new_base_digits = 6;
    std::vector<int> lhs_base_6 = ChangeBase(_digits, BASE_DIGITS, new_base_digits);
    std::vector<int> rhs_base_6 = ChangeBase(big_int._digits, BASE_DIGITS, new_base_digits);
    /* конструктор по итераторам, т.к. std::vector<int> по другому не превратить в std::vector<long long>
     * а переделать вовзращаемый тип ChangeBase нельзя, т.к. после ещё обратное преобразование */
    std::vector<long long> lhs(lhs_base_6.begin(), lhs_base_6.end());
    std::vector<long long> rhs(rhs_base_6.begin(), rhs_base_6.end());

    /* заполняем нулями до ближайшей степени двойки */
    size_t size_to_have = CheckNearestTwoDegree(std::max(lhs.size(), rhs.size()));
    FillWithZeros(lhs, size_to_have);
    FillWithZeros(rhs, size_to_have);

    std::vector<long long> temp_res = MultiplyKaratsuba(lhs, rhs);
    BigInteger result{};
    result.sign = (sign == big_int.sign) ? PLUS : MINUS;
    long long to_add = 0;
    for (long long cur_value : temp_res) {
        long long current = cur_value + to_add;
        result._digits.push_back(static_cast<int>(current % new_base));
        to_add = current / new_base;
    }
    result._digits = ChangeBase(result._digits, new_base_digits, BASE_DIGITS);
    result.Shrink();
    return result;
}


std::vector<int> BigInteger::ChangeBase(const std::vector<int> &_old, int old_base_digits, int new_base_digits) {
    //вектор с нужными степенями десятки
    std::vector<long long> power(std::max(old_base_digits, new_base_digits) + 1);
    power[0] = 1;
    for (int i = 1; i < static_cast<int>(power.size()); i++) {
        power[i] = power[i - 1] * 10;
    }
    std::vector<int> result{};
    int current_digits{0};
    long long current{0};
    for (int digit : _old) {
        current += digit * power[current_digits];
        current_digits += old_base_digits;
        while (current_digits >= new_base_digits) {
            result.push_back(static_cast<int>(current % power[new_base_digits]));
            current /= power[new_base_digits];
            current_digits -= new_base_digits;
        }
    }
    result.push_back(static_cast<int>(current));
    while (!result.empty() && result.back() == 0) {
        result.pop_back();
    }
    return result;
}

const BigInteger operator/(const BigInteger &lhs, const BigInteger &rhs) {
    BigInteger lhs_copy = lhs;
    return lhs_copy /= rhs;
}

const BigInteger operator%(const BigInteger &lhs, const BigInteger &rhs) {
    BigInteger lhs_copy = lhs;
    return lhs_copy %= rhs;
}

BigInteger &BigInteger::operator/=(const BigInteger &big_int) {
    *this = this->DivWithMod(big_int).first;
    return *this;
}

BigInteger &BigInteger::operator%=(const BigInteger &big_int) {
    *this = this->DivWithMod(big_int).second;
    return *this;
}

BigInteger &BigInteger::operator/=(long long number) {
    if (llabs(number) >= BASE) {
        *this /= BigInteger(number);
        return *this;
    }
    if (number < 0) {
        sign = static_cast<SIGN>(-1 * sign);
        number = -number;
    }
    for (int i = static_cast<int>(_digits.size() - 1), rem = 0; i >= 0; --i) {
        long long current = _digits[i] + rem * static_cast<long long>(BASE);
        _digits[i] = static_cast<int>(current / number);
        rem = static_cast<int>(current % number);
    }
    Shrink();
    return *this;
}

const BigInteger BigInteger::operator/(long long number) const {
    if (llabs(number) >= BASE) {
        return *this / BigInteger(number);
    }
    BigInteger result = *this;
    return result /= number;
}

BigInteger &BigInteger::operator%=(long long number) {
    return *this %= BigInteger(number);
}

const BigInteger BigInteger::operator%(long long number) const {
    if (number >= BASE) {
        return *this % BigInteger(number);
    }
    long long result = 0;
    for (int i = static_cast<int>(_digits.size() - 1); i >= 0; --i) {
        result = (_digits[i] + result * static_cast<long long>(BASE)) % number;
    }
    return result * sign;
}

BigInteger &BigInteger::operator*=(long long number) {
    if (llabs(number) >= BASE) {
        *this *= BigInteger(number);
        return *this;
    }
    if (number < 0) {
        sign = static_cast<SIGN>(-1 * sign);
        number = -number;
    }
    for (int pos = 0, to_add = 0; pos < static_cast<int>(_digits.size()) || to_add != 0; ++pos) {
        if (pos == static_cast<int>(_digits.size())) {
            _digits.push_back(0);
        }
        long long current = _digits[pos] * static_cast<long long>(number) + to_add;
        to_add = static_cast<int>(current / BASE);
        _digits[pos] = static_cast<int>(current % BASE);
    }
    Shrink();
    return *this;
}

const BigInteger BigInteger::operator*(long long number) const {
    BigInteger this_copy = *this;
    return this_copy *= number;
}

BigInteger &BigInteger::operator*=(const BigInteger &big_int) {
    if (_digits.size() <= NORMAL_MULTIPLY_LIMIT || big_int._digits.size() <= NORMAL_MULTIPLY_LIMIT) {
        *this = MultiplyN_2(big_int);
    } else {
        *this = Multiply_N_1_58(big_int);
    }
    return *this;
}

const BigInteger operator*(const BigInteger &lhs, const BigInteger &rhs) {
    BigInteger lhs_copy = lhs;
    lhs_copy *= rhs;
    return lhs_copy;
}

BigInteger::operator bool() const {
    return !IsZero();
}

BigInteger::operator long long() const {
    long long result{0};
    for (int i = static_cast<int>(_digits.size() - 1); i >= 0; --i) {
        result *= BASE;
        result += _digits[i];
    }
    return result;
}

BigInteger BigInteger::GCD(BigInteger lhs, BigInteger rhs) {
    while (!lhs.IsZero() && !rhs.IsZero()) {
        bool is_smaller = lhs < rhs;
        lhs = is_smaller ? lhs : lhs % rhs;
        rhs = is_smaller ? rhs % lhs : rhs;
    }
    return lhs + rhs;
}

size_t BigInteger::Length() const {
    size_t length{0};
    if (IsZero()) {
        return 1;
    }
    length += (static_cast<int>(_digits.size()) - 1) * BASE_DIGITS + std::to_string(_digits.back()).size();
    return length;
}

SIGN BigInteger::GetSign() const {
    return sign;
}

int BigInteger::GetBase() {
    return BASE;
}

int BigInteger::GetBaseDigits() {
    return BASE_DIGITS;
}


const BigInteger BigInteger::Abs() const {
    BigInteger result = *this;
    result.sign = PLUS;
    return result;
}

bool BigInteger::IsZero() const {
    return _digits.empty() || (_digits.size() == 1 && _digits[0] == 0);
}

size_t BigInteger::NumberOfZeros(int digit) {
    return static_cast<size_t>(BASE_DIGITS - static_cast<int>(std::to_string(digit).size()));
}

void BigInteger::FillWithZeros(std::vector<long long> &digits, size_t size_to_fill) {
    digits.resize(size_to_fill, 0);
}


int BigInteger::CheckNearestTwoDegree(int number) {
    if ((number & (number - 1)) == 0) {
        return number;
    }
    int nearest = 1;
    while (nearest < number) {
        nearest *= 2;
    }
    return nearest;
}

void BigInteger::Shrink() {
    while (!_digits.empty() && _digits.back() == 0) {
        _digits.pop_back();
    }
    if (_digits.empty()) {
        sign = PLUS;
    }
}

Rational &Rational::operator=(const Rational &rational) {
    if (this != &rational) {
        _numerator = rational._numerator;
        _denominator = rational._denominator;
    }
    return *this;
}

Rational::Rational(BigInteger numerator, BigInteger denominator) : _numerator(std::move(numerator)),
                                                                   _denominator(std::move(denominator)) {
    Normalize();
}

Rational::Rational(long long numerator, long long denominator) : _numerator(numerator), _denominator(denominator) {
    Normalize();
}

int Rational::CompareAbs(const Rational &lhs, const Rational &rhs) {
    return BigInteger::CompareAbs(lhs._numerator * rhs._denominator, lhs._denominator * rhs._numerator);
}

bool operator<(const Rational &lhs, const Rational &rhs) {
    if (lhs.GetSign() != rhs.GetSign()) {
        return lhs.GetSign() < rhs.GetSign();
    }
    return Rational::CompareAbs(lhs, rhs) * lhs.GetSign() == -1;
}

bool operator>(const Rational &lhs, const Rational &rhs) {
    return rhs < lhs;
}

bool operator>=(const Rational &lhs, const Rational &rhs) {
    return !(lhs < rhs);
}

bool operator<=(const Rational &lhs, const Rational &rhs) {
    return !(rhs < lhs);
}

bool operator==(const Rational &lhs, const Rational &rhs) {
    return lhs.GetSign() == rhs.GetSign() && Rational::CompareAbs(lhs, rhs) == 0;
}

bool operator!=(const Rational &lhs, const Rational &rhs) {
    return !(lhs == rhs);
}

std::string Rational::toString() const {
    std::string result;
    result += _numerator.toString();
    if (_denominator != 1) {
        result += '/';
        result += _denominator.toString();
    }
    return result;
}

std::istream &operator>>(std::istream &in, Rational &rational) {
    int x;
    in >> x;
    rational = x;
    rational.Normalize();
    return in;
}

std::ostream &operator<<(std::ostream &out, const Rational &rational) {
    out << double(rational);
    return out;
}

const Rational Rational::operator-() const {
    Rational result = *this;
    result._numerator = -result._numerator;
    return result;
}

Rational &Rational::operator+=(const Rational &rational) {
    _numerator *= rational._denominator;
    _numerator += rational._numerator * _denominator;
    _denominator *= rational._denominator;
    Normalize();
    return *this;
}

Rational &Rational::operator-=(const Rational &rational) {
    _numerator *= rational._denominator;
    _numerator -= rational._numerator * _denominator;
    _denominator *= rational._denominator;
    Normalize();
    return *this;
}

const Rational operator+(const Rational &lhs, const Rational &rhs) {
    Rational lhs_copy = lhs;
    return lhs_copy += rhs;
}

const Rational operator-(const Rational &lhs, const Rational &rhs) {
    Rational lhs_copy = lhs;
    return lhs_copy -= rhs;
}

Rational &Rational::operator*=(const Rational &rational) {
    _numerator *= rational._numerator;
    _denominator *= rational._denominator;
    Normalize();
    return *this;
}

Rational &Rational::operator/=(const Rational &rational) {
    _numerator *= rational._denominator;
    _denominator *= rational._numerator;
    Normalize();
    return *this;
}

const Rational operator*(const Rational &lhs, const Rational &rhs) {
    Rational lhs_copy = lhs;
    return lhs_copy *= rhs;
}

const Rational operator/(const Rational &lhs, const Rational &rhs) {
    Rational lhs_copy = lhs;
    return lhs_copy /= rhs;
}


std::string Rational::asDecimal(size_t precision) const {
    if (_denominator == 1) {
        return _numerator.toString();
    }
    size_t digits_precision = 0;
    Rational this_copy = *this;
    BigInteger remainder = this_copy._numerator.Abs() % this_copy._denominator;
    BigInteger int_part = this_copy._numerator.Abs() / this_copy._denominator;
    BigInteger digits{0};
    while (digits_precision < precision + 1) {
        while (remainder < this_copy._denominator) {
            BigInteger::MultiplyByBase(remainder);
            digits_precision += BigInteger::GetBaseDigits();
            BigInteger::MultiplyByBase(digits);
        }
        digits += remainder / this_copy._denominator;
        remainder %= this_copy._denominator;
    }
    while (digits_precision != precision + 1) {
        digits /= 10;
        --digits_precision;
    }
    BigInteger back = digits % 10;
    digits /= 10;
    --digits_precision;
    size_t old_length = digits.Length();
    if (back >= 5) {
        digits += 1;
    }
    if (digits.Length() != old_length) {
        int_part += 1;
        digits = 0;
    }
    std::string result{};
    if (this_copy._numerator.GetSign() == MINUS && !(int_part == 0 && digits == 0)) {
        result += '-';
    }
    result += int_part.toString();
    if (precision != 0) {
        result += ".";
        AddZeros(result, digits_precision - digits.Length());
        result += digits.toString();
        RemoveTrailingZeros(result, digits_precision);
        if (result.back() == '.') {
            result.pop_back();
        }
    }
    return result;
}

Rational::operator double() const {
    return std::stod(asDecimal(16));
}


void Rational::Normalize() {
    if (_denominator == 0) {
        throw (std::runtime_error("Denominator cannot be 0"));
    }
    if (_denominator.GetSign() == MINUS) {
        _denominator = -_denominator;
        _numerator = -_numerator;
    }
    BigInteger gcd = BigInteger::GCD(_denominator.Abs(), _numerator.Abs());
    _denominator /= gcd;
    _numerator /= gcd;
}

void Rational::RemoveTrailingZeros(std::string &str, size_t &curr_precision) {
    while (str.back() == '0') {
        str.pop_back();
        --curr_precision;
    }
}

SIGN Rational::GetSign() const {
    return _numerator.GetSign();
}

#endif // MATRIX_H
