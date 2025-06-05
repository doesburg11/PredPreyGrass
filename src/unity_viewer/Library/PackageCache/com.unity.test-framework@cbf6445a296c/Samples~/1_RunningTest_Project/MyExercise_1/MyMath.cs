
namespace MyExercise_1
{
    public static class MyMath
    {
        public static int Add(int a, int b)
        {
            return a + b;
        }
    
        public static int Subtract(int a, int b)
        {
            return a - b + a; // The code is wrong. Let's see if your test can see that.
        }
    }
}
