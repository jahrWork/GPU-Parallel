# include <stdlib.h>
# include <stdio.h>



extern "C" void print_C(char* text)

{

    printf("%s\n", text);

}

extern "C" float add_cpp(float a, float b)
{
    printf("%10.8f\n", a);
    printf("%10.8f\n", b);
    return a + b;
}

extern "C" int add_cpp2(int a, int b)
{
    return a + b;
}



extern "C" int mul_cpp3(float a, float b)
{
    return a + b;
}
