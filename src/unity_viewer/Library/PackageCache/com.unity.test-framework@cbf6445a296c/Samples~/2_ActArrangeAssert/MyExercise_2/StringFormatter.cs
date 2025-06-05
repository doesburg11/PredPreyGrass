using System.Linq;

namespace MyExercise_2
{
    public class StringFormatter
    {
        private string m_joinDelimiter = ",";
        
        public void Configure(string joinDelimiter)
        {
            m_joinDelimiter = joinDelimiter;
        }

        public string Join(object[] args)
        {
            return string.Join(m_joinDelimiter, args.Select(arg => arg.ToString()).ToArray());
        }
    }
}