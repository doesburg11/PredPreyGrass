using System.Linq;

namespace MyExercise_3s
{
    public class ValueOutputter
    {
        public string GetString()
        {
            return "A string to be asserted on";
        }
        
        public int GetInt()
        {
            return 11;
        }
        
        public float GetFloat()
        {
            return 19f + (1f/3f);
        }
    }
}