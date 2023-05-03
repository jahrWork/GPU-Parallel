// main.cpp . 

#include <iostream>


extern "C" float add_cpp(float, float);


int main()
{
 

    printf("C++ sum = % f", add_cpp(3.4, 4.3) ); 

}
