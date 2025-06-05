using System.Collections.Generic;
using System.Threading.Tasks;

public class MyClass
{
    public int Add(int a, int b)
    {
        return a + b;
    }

    public IEnumerator<int> AddAsync(int a, int b)
    {
        yield return default;
        yield return default;
        yield return default;
        yield return a + b;
    }
}
