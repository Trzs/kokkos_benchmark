#include <Kokkos_Core.hpp>
#include <iostream>

using range2d_policy = Kokkos::MDRangePolicy<Kokkos::Rank<2>>;

template <typename T, int N, int M>
struct test_suite
{
  using view_t = Kokkos::View<T***>;

  view_t A = view_t ("A", N, N, M);
  view_t B = view_t ("B", N, N, M);
  view_t C = view_t ("C", N, N, M);

  void init_views()
  {
    auto tmp_A = A;
    auto tmp_B = B;
    auto tmp_C = C;
    Kokkos::parallel_for("init_views", range2d_policy ({0, 0}, {N, N}), KOKKOS_LAMBDA (const int i, const int j) {
      for (int k=0; k<M; k++)
      {
        tmp_A(i,j,k) = 1;
        tmp_B(i,j,k) = i/1000 - 0.5;
        tmp_C(i,j,k) = j/1000;
      }
    });
  }

  void add()
  {
    auto tmp_B = B;
    auto tmp_C = C;
    Kokkos::parallel_for("add", range2d_policy ({0, 0}, {N, N}), KOKKOS_LAMBDA (const int i, const int j) {
      for (int k=0; k<M; k++)
      {
        tmp_C(i, j, k) = tmp_B(i, j, k) + tmp_C(i, j, k);
      }
    });
  }

  void sub()
  {
    auto tmp_A = A;
    auto tmp_B = B;
    Kokkos::parallel_for("sub", range2d_policy ({0, 0}, {N, N}), KOKKOS_LAMBDA (const int i, const int j) {
      for (int k=0; k<M; k++)
      {
        tmp_A(i, j, k) = tmp_B(i, j, k) - tmp_A(i, j, k);
      }
    });
  }

  void mult()
  {
    auto tmp_B = B;
    auto tmp_C = C;
    Kokkos::parallel_for("mult_three", range2d_policy ({0, 0}, {N, N}), KOKKOS_LAMBDA (const int i, const int j) {
      for (int k=0; k<M; k++)
      {
        tmp_C(i, j, k) = tmp_C(i, j, k) * tmp_B(i, j, k);
      }
    });
  }

  void div()
  {
    auto tmp_B = B;
    auto tmp_C = C;
    Kokkos::parallel_for("div_three", range2d_policy ({0, 0}, {N, N}), KOKKOS_LAMBDA (const int i, const int j) {
      for (int k=0; k<M; k++)
      {
        tmp_B(i, j, k) = tmp_B(i, j, k) / tmp_C(i, j, k);
      }
    });
  }

  void exp()
  {
    auto tmp_A = A;
    auto tmp_C = C;
    Kokkos::parallel_for("exp", range2d_policy ({0, 0}, {N, N}), KOKKOS_LAMBDA (const int i, const int j) {
      for (int k=0; k<M; k++)
      {
        tmp_A(i, j, k) = Kokkos::Experimental::exp( tmp_C(i, j, k) );
      }
    });
  }

  void sin()
  {
    auto tmp_A = A;
    auto tmp_C = C;
    Kokkos::parallel_for("sin", range2d_policy ({0, 0}, {N, N}), KOKKOS_LAMBDA (const int i, const int j) {
      for (int k=0; k<M; k++)
      {
        tmp_A(i, j, k) = Kokkos::Experimental::sin( tmp_C(i, j, k) );
      }
    });
  }

  void expsin()
  {
    auto tmp_A = A;
    auto tmp_C = C;
    Kokkos::parallel_for("sin", range2d_policy ({0, 0}, {N, N}), KOKKOS_LAMBDA (const int i, const int j) {
      for (int k=0; k<M; k++)
      {
        tmp_A(i, j, k) = Kokkos::Experimental::exp( Kokkos::Experimental::sin( tmp_C(i, j, k)) );
      }
    });
  }

};
