#include <iostream>

void matrixFunction() 
{
    const int n1 = 10;
    float a[n1][n1] = {};
    float b[n1] = {};
    float c[n1] = {};
    for (int i = 0; i < n1; i++)
    {
        for (int j = 0; j < n1; j++)
        {
            a[i][j] = j + 1.1f;
        }
        b[i] = i + 1.1f;
    }
    for (int i = 0; i < n1; i++)
    {
        for (int j = 0; j < n1; j++)
        {
            c[i] = c[i] + a[i][j]*b[j];
        }
    }
    std::cout << "a = \n";
    for (int i = 0; i < n1; i++)
    {
        for (int j = 0; j < n1; j++)
        {
            std::cout << a[i][j] << "   ";
        }
        std::cout << "\n";
    }
    std::cout << "b = \n";
    for (int i = 0; i < n1; i++)
    {
        std::cout << b[i] << "\n";
    }
    std::cout << "c = \n";
    for (int i = 0; i < n1; i++)
    {
        std::cout << c[i] << "\n";
    }
}
