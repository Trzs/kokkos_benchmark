#include <iostream>
#include <iomanip>
#include <Kokkos_Core.hpp>

#include "test_suite.h"

#define RUN_TEST( function ) timer.reset(); test_double.function(); Kokkos::fence(); t_double=timer.seconds(); \
                             timer.reset(); test_float.function(); Kokkos::fence(); t_float=timer.seconds(); \
                             std::cout << std::setw(15) << "T[" #function "] " \
                                       << std::setw(10) << t_float << "s | " \
                                       << std::setw(10) << t_double << "s." << std::endl;

int main() {

  Kokkos::initialize();
  {
    const int size = 1000;
    const int loops = 1000;

    test_suite<double, size, loops> test_double;
    test_suite<float, size, loops> test_float;

    Kokkos::Timer timer;
    double t_double = 0;
    double t_float = 0;

    std::cout << " Array size: " << size << " x " << size << " x " << loops << std::endl; 
    std::cout << std::endl;
    std::cout << "   FUNCTION        FLOAT   |    DOUBLE  " << std::endl;
    std::cout << "---------------------------+------------" << std::endl;
    RUN_TEST( init_views );
    RUN_TEST( add );
    RUN_TEST( sub );
    RUN_TEST( mult );
    RUN_TEST( div );
    RUN_TEST( exp );
    RUN_TEST( sin );
    RUN_TEST( expsin );

  }
  Kokkos::finalize();

  return 0;
}
